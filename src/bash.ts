import * as fs from "node:fs/promises";
import { createWriteStream, mkdirSync, type WriteStream } from "node:fs";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import { randomBytes } from "node:crypto";
import type { AgentToolProgress } from "./types.ts";
import { errnoCode } from "./utils.ts";

export const BASH_HEAD_BYTES = 16 * 1024;
export const BASH_TAIL_BYTES = 32 * 1024;
// 输出超出头尾摘要时尽力保存输出日志；单个日志的磁盘上限，防止持续刷屏的命令写满磁盘
export const MAX_BASH_LOG_BYTES = 64 * 1024 * 1024;
const KILL_GRACE_MS = 2_000;
const IO_DRAIN_GRACE_MS = 200;
const GROUP_POLL_MS = 40;
// bash 执行中进度报告的节流间隔与单次增量的字符上限（超出时只保留末尾并注明跳过量）
const BASH_PROGRESS_INTERVAL_MS = 250;
const BASH_PROGRESS_MAX_CHARS = 8_192;

function isUtf8ContinuationByte(byte: number): boolean {
  return (byte & 0xc0) === 0x80;
}

// 向下对齐到合法的 UTF-8 字符边界（避免在前缀截断处切开多字节字符）
function alignUtf8PrefixEnd(buf: Buffer, maxLen: number): number {
  let end = Math.min(buf.length, maxLen);
  while (end > 0 && isUtf8ContinuationByte(buf[end])) {
    end--;
  }
  return end;
}

// 向上对齐到合法的 UTF-8 字符边界（避免在后缀截断起点处残留半个多字节字符）
function alignUtf8SuffixStart(buf: Buffer, startIdx: number): number {
  let start = Math.max(0, startIdx);
  while (start < buf.length && isUtf8ContinuationByte(buf[start])) {
    start++;
  }
  return start;
}

// 节流的增量文本进度：push 累积新文本，每个间隔最多报告一次；累积超过上限时只保留末尾并注明跳过量。
// flush 立即报告剩余内容并停止计时（工具返回前调用）。
export function createThrottledTextReporter(report: (progress: AgentToolProgress) => void): {
  push: (text: string) => void;
  flush: () => void;
} {
  let pending = "";
  let skipped = 0;
  let timer: NodeJS.Timeout | undefined;
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (!pending && !skipped) return;
    const text = skipped > 0 ? `[... ${skipped} chars skipped ...]\n${pending}` : pending;
    pending = "";
    skipped = 0;
    report({ text });
  };
  const push = (text: string) => {
    pending += text;
    if (pending.length > BASH_PROGRESS_MAX_CHARS) {
      let cut = pending.length - BASH_PROGRESS_MAX_CHARS;
      // 不从代理对中间切开
      const code = pending.charCodeAt(cut);
      if (code >= 0xdc00 && code <= 0xdfff) cut++;
      skipped += cut;
      pending = pending.slice(cut);
    }
    timer ??= setTimeout(flush, BASH_PROGRESS_INTERVAL_MS);
  };
  return { push, flush };
}

// 头尾缓冲区不变式：head 始终是原输出（解码后）的连续前缀；tail 始终是原输出的连续后缀。
// 一旦有任何内容进入 tail，head 即永久封口——字符边界对齐留下的少量空位宁可闲置，
// 也不允许后到的内容补进 head 而排到先到内容之前。
export class HeadTailBuffer {
  private head = Buffer.alloc(0);
  private tail = Buffer.alloc(0);
  private headSealed = false;
  private totalBytes = 0;
  private decodedBytes = 0;

  constructor(
    private readonly headLimit: number = BASH_HEAD_BYTES,
    private readonly tailLimit: number = BASH_TAIL_BYTES
  ) {}

  // text 必须是已解码的完整字符（多个流各自用独立的 StringDecoder 解码）；rawByteLength 为对应的原始字节数
  append(text: string, rawByteLength: number): void {
    this.totalBytes += rawByteLength;
    if (!text) return;
    const chunk = Buffer.from(text, "utf-8");
    this.decodedBytes += chunk.length;

    if (!this.headSealed && this.head.length < this.headLimit) {
      const needed = alignUtf8PrefixEnd(chunk, this.headLimit - this.head.length);
      if (needed > 0) {
        this.head = Buffer.concat([this.head, chunk.subarray(0, needed)]);
      }
      const rest = chunk.subarray(needed);
      if (rest.length > 0) {
        this.headSealed = true;
        this.appendTail(rest);
      }
    } else {
      this.headSealed = true;
      this.appendTail(chunk);
    }
  }

  private appendTail(chunk: Buffer): void {
    if (chunk.length >= this.tailLimit) {
      const start = alignUtf8SuffixStart(chunk, chunk.length - this.tailLimit);
      this.tail = Buffer.from(chunk.subarray(start));
    } else {
      const combined = Buffer.concat([this.tail, chunk]);
      if (combined.length > this.tailLimit) {
        const start = alignUtf8SuffixStart(combined, combined.length - this.tailLimit);
        this.tail = Buffer.from(combined.subarray(start));
      } else {
        this.tail = combined;
      }
    }
  }

  // 是否有内容被省略（中间部分未进入 head/tail）
  get isTruncated(): boolean {
    return this.decodedBytes > this.head.length + this.tail.length;
  }

  format(savedLog?: { path: string; complete: boolean }): string {
    const stripTrailingNewline = (s: string) => s.replace(/\r?\n$/, "");
    if (!this.isTruncated) {
      return stripTrailingNewline(Buffer.concat([this.head, this.tail]).toString("utf-8"));
    }
    // 省略量按解码后的字节计算（与 head/tail 同一口径）；total 为原始字节数
    const omitted = Math.max(1, this.decodedBytes - this.head.length - this.tail.length);
    const logNote = savedLog
      ? ` ${savedLog.complete ? "Full output" : `Output prefix (up to ${MAX_BASH_LOG_BYTES} bytes)`} saved to: ${savedLog.path} ` +
        `(inspect it with bash, e.g. rg/grep -n, sed -n 'X,Yp', tail).`
      : "";
    return stripTrailingNewline(
      [
        this.head.toString("utf-8"),
        `\n... [Omitted ${omitted} bytes in the middle (total output: ${this.totalBytes} bytes).${logNote}] ...\n`,
        this.tail.toString("utf-8"),
      ].join("")
    );
  }
}

// 完整输出日志：输出接近头尾摘要容量前只在内存暂存（不超过约 48 KiB），超过后才创建日志文件，
// 先写入暂存内容，再流式追加后续输出。写入的是与 HeadTailBuffer 相同的已解码文本，
// 达到 MAX_BASH_LOG_BYTES 后停止写入。任何日志 I/O 失败只会让日志不可用，不影响命令结果。
class BashOutputLog {
  private pending: string[] = [];
  private pendingBytes = 0;
  private writtenBytes = 0;
  private stream: WriteStream | undefined;
  private filePath: string | undefined;
  private failed = false;
  private capped = false;
  private created = false;

  constructor(
    private readonly resolveDir: (() => string) | undefined,
    private readonly openThreshold: number
  ) {}

  append(text: string): void {
    if (!text || !this.resolveDir || this.failed) return;
    if (!this.stream) {
      this.pending.push(text);
      this.pendingBytes += Buffer.byteLength(text, "utf-8");
      if (this.pendingBytes <= this.openThreshold) return;
      if (!this.open()) return;
      text = this.pending.join("");
      this.pending = [];
    }
    this.write(text);
  }

  private open(): boolean {
    try {
      const dir = this.resolveDir!();
      mkdirSync(dir, { recursive: true, mode: 0o700 });
      this.filePath = path.join(dir, `bash-${Date.now()}-${randomBytes(4).toString("hex")}.log`);
      this.stream = createWriteStream(this.filePath, { flags: "wx", mode: 0o600 });
      this.stream.once("open", () => {
        this.created = true;
      });
      this.stream.on("error", () => {
        this.failed = true;
      });
      return true;
    } catch {
      this.failed = true;
      this.pending = [];
      return false;
    }
  }

  private write(text: string): void {
    if (this.capped || !this.stream) return;
    let buf = Buffer.from(text, "utf-8");
    const room = MAX_BASH_LOG_BYTES - this.writtenBytes;
    if (buf.length > room) {
      buf = buf.subarray(0, alignUtf8PrefixEnd(buf, room));
      this.capped = true;
    }
    if (buf.length > 0) {
      this.stream.write(buf);
      this.writtenBytes += buf.length;
    }
  }

  // 结束写入并等待文件真正关闭（"close" 事件），再按流自身的错误状态判断日志是否可用：
  // destroyed 只表示开始销毁，error 事件可能尚未派发；end() 回调也不代表文件已关闭。
  // keep 为 false 或写入失败时删除文件——但只删除本次成功创建的文件（"wx" 撞名失败时路径属于他人）。不会抛错。
  async close(keep: boolean): Promise<{ path: string; complete: boolean } | undefined> {
    this.pending = [];
    const { stream, filePath } = this;
    if (!stream || !filePath) return undefined;
    await new Promise<void>((resolve) => {
      if (stream.closed) return resolve();
      stream.once("close", () => resolve());
      if (!stream.destroyed) stream.end();
    });
    if (!keep || this.failed || stream.errored) {
      if (this.created) await fs.rm(filePath, { force: true }).catch(() => {});
      return undefined;
    }
    return { path: filePath, complete: !this.capped };
  }
}

const activeProcessGroups = new Set<number>();
process.once("exit", () => {
  for (const pid of activeProcessGroups) sendProcessGroupSignal(pid, "SIGKILL");
});

function isProcessGroupAlive(pid: number): boolean {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (err) {
    return errnoCode(err) === "EPERM";
  }
}

function sendProcessGroupSignal(pid: number | undefined, sig: NodeJS.Signals): void {
  if (!pid) return;
  try {
    process.kill(-pid, sig);
  } catch {}
}

export interface BashRunOptions {
  // 在继承 process.env 的基础上覆盖这些变量；传入 {} 不会得到干净环境，不能当作白名单
  env?: NodeJS.ProcessEnv;
  signal?: AbortSignal;
  // 输出超出头尾摘要时保存完整日志的目录（按需调用，仅在确实需要创建日志时才求值）；
  // 省略则不保存日志。日志保留到宿主清理为止；取消或启动失败时不保留。
  logDir?: () => string;
  // 每段已解码输出（stdout/stderr 按到达顺序）的同步观察回调，异常被隔离；用于进度展示，不影响结果
  onOutput?: (text: string) => void;
}

export interface BashRunResult {
  exitCode: number | null;
  exitSignal: NodeJS.Signals | null;
  output: string;
  // 已保存的输出日志（仅在输出被截断且日志可用时存在）
  log?: { path: string; complete: boolean };
}

export function runBashProcessGroup(
  command: string,
  cwd: string,
  timeoutMs: number,
  options: BashRunOptions = {}
): Promise<BashRunResult> {
  const { env: customEnv, signal, logDir, onOutput } = options;
  return new Promise((resolve, reject) => {
    if (process.platform === "win32") {
      return reject(new Error("bash tool requires a POSIX platform (/bin/bash and process groups)."));
    }
    if (signal?.aborted) {
      return reject(signal.reason);
    }

    const child = spawn(command, {
      cwd,
      shell: "/bin/bash",
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        ...customEnv,
        CI: "1",
        GIT_PAGER: "cat",
        GIT_TERMINAL_PROMPT: "0",
      },
    });

    const pid = child.pid;
    if (pid) activeProcessGroups.add(pid);

    // stdout 与 stderr 各自使用独立的 StringDecoder 解码，避免双流字节交错切碎多字节 UTF-8 字符
    const buffer = new HeadTailBuffer();
    // 提前少量开启日志：字符边界对齐可能让 head/tail 实际容量略小于上限；最终未截断时日志会被删除
    const log = new BashOutputLog(logDir, BASH_HEAD_BYTES + BASH_TAIL_BYTES - 8);
    const stdoutDecoder = new StringDecoder("utf-8");
    const stderrDecoder = new StringDecoder("utf-8");
    const collect = (text: string, rawByteLength: number) => {
      buffer.append(text, rawByteLength);
      log.append(text);
      if (onOutput && text) {
        try {
          onOutput(text);
        } catch {}
      }
    };

    child.stdout?.on("data", (chunk: Buffer) => {
      collect(stdoutDecoder.write(chunk), chunk.length);
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      collect(stderrDecoder.write(chunk), chunk.length);
    });

    let timedOut = false;
    let aborted = false;
    let terminating = false;
    let shellExited = false;
    let exitCode: number | null = null;
    let exitSignal: NodeJS.Signals | null = null;
    let settled = false;

    let timeoutTimer: NodeJS.Timeout | undefined;
    let killEscalationTimer: NodeJS.Timeout | undefined;
    let postKillTimer: NodeJS.Timeout | undefined;
    let groupPollTimer: NodeJS.Timeout | undefined;
    let drainTimer: NodeJS.Timeout | undefined;

    const finalize = (spawnErr?: Error) => {
      if (settled) return;
      settled = true;
      if (pid) activeProcessGroups.delete(pid);

      clearTimeout(timeoutTimer);
      clearTimeout(killEscalationTimer);
      clearTimeout(postKillTimer);
      clearInterval(groupPollTimer);
      clearTimeout(drainTimer);
      drainTimer = undefined;
      signal?.removeEventListener("abort", onAbort);

      // 冲刷两个解码器中可能残留的末尾字节
      const remOut = stdoutDecoder.end();
      if (remOut) collect(remOut, 0);
      const remErr = stderrDecoder.end();
      if (remErr) collect(remErr, 0);

      child.stdout?.destroy();
      child.stderr?.destroy();
      sendProcessGroupSignal(pid, "SIGKILL");

      // 结果此刻已确定；等待日志落盘只是为了在返回前关闭文件（close 不会抛错）
      const keepLog = !spawnErr && !aborted && buffer.isTruncated;
      void log.close(keepLog).then((savedLog) => {
        const output = buffer.format(savedLog);
        if (spawnErr) return reject(spawnErr);
        if (aborted) return reject(signal?.reason); // 原样传递取消原因（可能是 null 等任意值）
        if (timedOut) {
          return reject(new Error(`Command timed out after ${timeoutMs}ms.\n${output}`));
        }
        resolve({ exitCode, exitSignal, output, ...(savedLog ? { log: savedLog } : {}) });
      });
    };

    const scheduleDrainAndFinalize = () => {
      if (drainTimer || settled) return;
      drainTimer = setTimeout(() => finalize(), IO_DRAIN_GRACE_MS);
    };

    // 终止流程中：shell 已退出且进程组已无存活进程时，停止轮询与强杀计时，进入输出排空
    const checkGroupDead = () => {
      if (shellExited && (!pid || !isProcessGroupAlive(pid))) {
        clearInterval(groupPollTimer);
        clearTimeout(killEscalationTimer);
        scheduleDrainAndFinalize();
      }
    };

    const startTerminationSequence = () => {
      if (terminating || settled) return;
      terminating = true;
      clearTimeout(timeoutTimer);
      clearTimeout(drainTimer);
      drainTimer = undefined;

      sendProcessGroupSignal(pid, "SIGTERM");

      groupPollTimer = setInterval(checkGroupDead, GROUP_POLL_MS);
      groupPollTimer.unref?.();

      // 注意：killEscalationTimer 保持 ref 状态（不调用 unref），防止 CLI 在等待强杀完成前提前退出事件循环
      killEscalationTimer = setTimeout(() => {
        clearInterval(groupPollTimer);
        sendProcessGroupSignal(pid, "SIGKILL");
        postKillTimer = setTimeout(() => finalize(), GROUP_POLL_MS);
      }, KILL_GRACE_MS);
    };

    timeoutTimer = setTimeout(() => {
      timedOut = true;
      startTerminationSequence();
    }, timeoutMs);

    const onAbort = () => {
      // shell 已退出时结果已确定：取消只提前结束输出排空，不把已完成的命令改报为取消
      if (shellExited && !terminating) return finalize();
      aborted = true;
      startTerminationSequence();
    };
    signal?.addEventListener("abort", onAbort, { once: true });

    child.on("error", (err) => finalize(err));

    child.on("exit", (code, sig) => {
      shellExited = true;
      exitCode = code;
      exitSignal = sig;
      if (terminating) return checkGroupDead();
      // timeoutMs 只约束 shell 本身；退出后的输出排空另有 IO_DRAIN_GRACE_MS 上限，不再触发超时
      clearTimeout(timeoutTimer);
      scheduleDrainAndFinalize();
    });

    child.on("close", (code, sig) => {
      shellExited = true;
      if (exitCode === null) exitCode = code;
      if (exitSignal === null) exitSignal = sig;
      if (terminating) {
        if (!pid || !isProcessGroupAlive(pid)) {
          finalize();
        }
        return;
      }
      finalize();
    });
  });
}
