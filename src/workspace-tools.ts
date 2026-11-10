import { z } from "zod";
import * as fs from "node:fs/promises";
import { mkdtempSync } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { randomBytes } from "node:crypto";
import type { AgentTool, AgentToolContext, AgentToolResult, ToolOutput } from "./types.ts";
import {
  BASH_HEAD_BYTES,
  BASH_TAIL_BYTES,
  MAX_BASH_LOG_BYTES,
  createThrottledTextReporter,
  runBashProcessGroup,
  type BashRunResult,
} from "./bash.ts";
import { isSupportedImage, MAX_IMAGE_EDGE, readImage } from "./image.ts";
import { errnoCode, sliceStringSafe } from "./utils.ts";

const MAX_READ_CHARS = 30_000;
const MAX_READ_BYTES = 5 * 1024 * 1024;
// 图片单独的磁盘大小上限：read 会先缩放再发送，内存由 image.ts 的 MAX_IMAGE_PIXELS 约束
const MAX_IMAGE_FILE_BYTES = 50 * 1024 * 1024;

// ignoreBOM: true 表示“不要吞掉 BOM”，解码结果会保留开头的 U+FEFF，
// 从而 edit 写回时 BOM 字节原样保留（默认 false 会静默剥离 BOM）。
const utf8FatalDecoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

// 将常量字节数格式化为文案（仅用于整 KiB/MiB 的限额常量）
function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${bytes / (1024 * 1024)} MiB` : `${bytes / 1024} KiB`;
}

async function resolveExistingRealPath(targetPath: string): Promise<string> {
  let current = path.resolve(targetPath);
  const tailParts: string[] = [];

  while (true) {
    try {
      const realCurrent = await fs.realpath(current);
      return tailParts.length > 0 ? path.join(realCurrent, ...tailParts.reverse()) : realCurrent;
    } catch (err) {
      if (errnoCode(err) === "ENOENT") {
        const st = await fs.lstat(current).catch(() => null);
        if (st?.isSymbolicLink()) {
          throw new Error(`Dangling symlink blocked: ${targetPath}`);
        }
        const parent = path.dirname(current);
        if (parent === current) return path.resolve(targetPath);
        tailParts.push(path.basename(current));
        current = parent;
      } else {
        throw err;
      }
    }
  }
}

function decodeStrictUtf8(buf: Buffer, filePath: string): string {
  if (buf.subarray(0, 1024).includes(0)) {
    throw new Error(`Binary file detected: ${filePath}. Cannot process as UTF-8 text.`);
  }
  try {
    return utf8FatalDecoder.decode(buf);
  } catch {
    throw new Error(`File is not valid UTF-8 text: ${filePath}.`);
  }
}

async function readFileHead(filePath: string, length: number): Promise<Buffer> {
  const handle = await fs.open(filePath, "r");
  try {
    const { bytesRead, buffer } = await handle.read(Buffer.alloc(length), 0, length, 0);
    return buffer.subarray(0, bytesRead);
  } finally {
    await handle.close();
  }
}

// 原子提交写入：同目录临时文件 → 完整写入并 fsync → 关闭 → rename 覆盖目标。
// - 提交边界是 rename：提交前观察到的失败或取消会清理临时文件，原目标不变；
//   rename 一旦开始即可能完成，此后按成功返回（不会在提交后再报告取消，避免“报失败、实际已改”）。
// - 只提供运行期间的原子替换与提交前失败保护；不承诺系统崩溃或掉电后的结果（未对目录 fsync）。
// - 只保留普通 rwx 权限位；setuid/setgid/sticky、属主/属组、ACL、xattr 不保留。
// - 目标已存在但当前进程不可写（如 0444）时拒绝，保持与直接写入相同的语义——rename 只看目录权限，
//   否则会悄悄覆盖用户设为只读的文件。该检查与 rename 之间仍有时间窗口，仅防误操作。
// - targetPath 应为已解析的真实路径：工作区内的 symlink 因此被保留；硬链接会被解绑（目标路径指向新 inode）。
async function atomicWriteFile(targetPath: string, content: string, signal?: AbortSignal): Promise<void> {
  const existing = await fs.stat(targetPath).catch((err: unknown) => {
    if (errnoCode(err) === "ENOENT") return null;
    throw err;
  });
  if (existing && !existing.isFile()) {
    throw new Error(`Cannot overwrite non-regular file: ${targetPath}`);
  }
  if (existing) {
    await fs.access(targetPath, fs.constants.W_OK).catch(() => {
      throw new Error(`Permission denied: file is not writable: ${targetPath}`);
    });
  }

  // 临时文件名不包含原文件名：原文件名可能已接近文件系统 255 字节上限（如长中文文件名）
  const dir = path.dirname(targetPath);
  const tmpPath = path.join(dir, `.agent.${process.pid}.${randomBytes(6).toString("hex")}.tmp`);
  const mode = existing ? existing.mode & 0o777 : 0o666;

  let handle: fs.FileHandle | undefined;
  let created = false;
  let committed = false;
  try {
    signal?.throwIfAborted();
    handle = await fs.open(tmpPath, "wx", mode);
    created = true;
    if (existing) await handle.chmod(mode); // open() 的 mode 受 umask 影响，这里显式还原原文件的 rwx 位
    await handle.writeFile(content, { encoding: "utf-8", signal });
    await handle.sync();
    await handle.close();
    handle = undefined;
    signal?.throwIfAborted();
    await fs.rename(tmpPath, targetPath);
    committed = true;
  } finally {
    if (handle) await handle.close().catch(() => {});
    // 只清理本次成功创建的临时文件："wx" 撞名失败时该路径属于他人，不能删除
    if (created && !committed) await fs.rm(tmpPath, { force: true }).catch(() => {});
  }
}

function stripLeadingBom(text: string): string {
  return text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
}

// 查找所有不把 "\r\n" 从中间劈开的有效匹配位置（上限 2 处即可判断唯一性）
function findMatchesWithoutSplittingCrLf(content: string, needle: string): number[] {
  const matches: number[] = [];
  let searchFrom = 0;
  while (searchFrom <= content.length - needle.length) {
    const idx = content.indexOf(needle, searchFrom);
    if (idx === -1) break;

    const splitsLeadingCrLf = needle.startsWith("\n") && idx > 0 && content[idx - 1] === "\r";
    const splitsTrailingCrLf =
      needle.endsWith("\r") && idx + needle.length < content.length && content[idx + needle.length] === "\n";

    if (!splitsLeadingCrLf && !splitsTrailingCrLf) {
      matches.push(idx);
      if (matches.length > 1) break;
    }
    searchFrom = idx + 1;
  }
  return matches;
}

// 用 Zod 对象定义 AgentTool：JSON Schema 由 Zod 生成，parse 负责运行时校验
function defineTool<S extends z.ZodObject>(spec: {
  name: string;
  description: string;
  inputSchema: S;
  run: (input: z.infer<S>, context: AgentToolContext) => Promise<ToolOutput | AgentToolResult>;
}): AgentTool<z.infer<S>> {
  return {
    name: spec.name,
    description: spec.description,
    input_schema: z.toJSONSchema(spec.inputSchema, { reused: "ref" }) as AgentTool["input_schema"],
    parse: (input) => spec.inputSchema.parse(input),
    run: spec.run,
  };
}

// 返回绑定到同一工作区根目录的 read / write / edit / bash 工具。
// 工具本身与代码无关（不做语法分析），以 workspace 为边界；read 支持文本与图片，write/edit 面向 UTF-8 文本。
// 取消信号由每次调用的 AgentToolContext.signal 提供。
// bashLogDir：bash 输出被截断时保存完整日志的目录（相对路径基于 workspace）。
// - 省略：首次需要时在系统临时目录下创建本实例专用目录（agent-bash-logs-XXXXXX）；该目录在工作区外，
//   read 工具会拒绝读取，模型需用 bash 检索；若希望 read 可读，宿主可将其设为工作区内的目录。
// - null：不保存日志，仅返回头尾摘要。
// 日志不会自动删除，清理由宿主负责。
export function createWorkspaceTools(
  workspace: string,
  options?: { env?: NodeJS.ProcessEnv; bashLogDir?: string | null }
): AgentTool[] {
  const root = path.resolve(workspace);
  const logDirOption = options?.bashLogDir;
  let defaultLogDir: string | undefined;
  const resolveBashLogDir: (() => string) | undefined =
    logDirOption === null
      ? undefined
      : logDirOption !== undefined
        ? () => path.resolve(root, logDirOption)
        : () => (defaultLogDir ??= mkdtempSync(path.join(os.tmpdir(), "agent-bash-logs-")));

  const isInsideRoot = (rootDir: string, targetPath: string): boolean => {
    const rel = path.relative(rootDir, targetPath);
    return rel !== ".." && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel);
  };

  // 路径检查用于防止误操作（目录穿越、指向工作区外的 symlink），不是安全沙箱：
  // 检查与后续按路径打开之间存在 TOCTOU 窗口，硬链接也不受 realpath 约束，且 bash 工具本身可执行任意命令。
  // 返回已解析 symlink 的真实路径，后续读写均基于该路径，避免二次跟随链接，并让原子写入保留工作区内的 symlink。
  const resolveWorkspacePath = async (filePath: string): Promise<string> => {
    const resolved = path.resolve(root, filePath);
    if (!isInsideRoot(root, resolved)) {
      throw new Error(`Path is outside workspace (${root}): ${filePath}`);
    }

    const realRoot = await resolveExistingRealPath(root);
    const realTarget = await resolveExistingRealPath(resolved);
    if (!isInsideRoot(realRoot, realTarget)) {
      throw new Error(`Symlink escape blocked: ${filePath} resolves outside workspace (${realTarget})`);
    }
    return realTarget;
  };

  const readTool = defineTool({
    name: "read",
    description:
      `Read a UTF-8 text file (up to ${formatBytes(MAX_READ_BYTES)}) or a JPEG, PNG, GIF, or WebP image (up to ${formatBytes(MAX_IMAGE_FILE_BYTES)}). ` +
      "Text supports 1-based line offset and limit. Images ignore offset/limit, are auto-oriented and resized to fit " +
      `${MAX_IMAGE_EDGE}x${MAX_IMAGE_EDGE} without cropping or enlargement; only the first frame is sent. Images require a vision-capable model.`,
    inputSchema: z.object({
      file_path: z.string().min(1),
      offset: z.number().int().min(1).optional(),
      limit: z.number().int().min(1).max(5000).optional(),
    }),
    run: async ({ file_path, offset, limit }, { signal: sig }) => {
      sig?.throwIfAborted();
      const fullPath = await resolveWorkspacePath(file_path);
      const stat = await fs.stat(fullPath);
      if (!stat.isFile()) {
        throw new Error(`Cannot read non-regular file: ${file_path}`);
      }
      // 文本与图片的大小上限不同：读入整个文件前先按文件头判断类型；读入后再按实际内容复核一次
      const assertWithinLimit = (isImage: boolean, size: number) => {
        if (isImage && size > MAX_IMAGE_FILE_BYTES) {
          throw new Error(
            `Image file size (${size} bytes) exceeds ${formatBytes(MAX_IMAGE_FILE_BYTES)} limit: ${file_path}. Resize it before reading.`
          );
        }
        if (!isImage && size > MAX_READ_BYTES) {
          throw new Error(
            `File size (${size} bytes) exceeds ${formatBytes(MAX_READ_BYTES)} limit. Use bash tools (head/tail/rg) instead.`
          );
        }
      };
      assertWithinLimit(isSupportedImage(await readFileHead(fullPath, 12)), stat.size);

      const buf = await fs.readFile(fullPath, { signal: sig });
      const isImage = isSupportedImage(buf);
      assertWithinLimit(isImage, buf.length);
      if (isImage) return readImage(buf, file_path, sig);

      // 展示时去掉 UTF-8 BOM（不可见字符，避免模型将其复制进 old_text）；edit 仍会原样保留文件中的 BOM
      const text = stripLeadingBom(decodeStrictUtf8(buf, file_path));
      if (text.length === 0) {
        return "(empty file)";
      }

      const lines = text.split(/\r?\n/);
      // 文件末尾的换行符不额外产生一行；没有换行符结尾的最后一行仍计入
      if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
      const start = (offset ?? 1) - 1;
      if (start >= lines.length) {
        return `(offset ${offset} is beyond end of file; total lines: ${lines.length})`;
      }

      const end = limit ? Math.min(start + limit, lines.length) : lines.length;
      // 单次循环按输出预算构造结果：超出 MAX_READ_CHARS 后不再格式化后续行；
      // 被截断的那一行保留预算内的部分内容（不切开代理对）
      const parts: string[] = [];
      let used = 0;
      let completeLines = 0;
      let truncated = false;
      for (let i = start; i < end; i++) {
        const piece = `${i > start ? "\n" : ""}${String(i + 1).padStart(4, " ")} | ${lines[i]}`;
        if (used + piece.length > MAX_READ_CHARS) {
          parts.push(sliceStringSafe(piece, MAX_READ_CHARS - used));
          truncated = true;
          break;
        }
        parts.push(piece);
        used += piece.length;
        completeLines++;
      }
      const numbered = parts.join("");

      if (truncated) {
        const hint =
          completeLines > 0
            ? `Lines ${start + 1}-${start + completeLines} shown completely; continue with offset=${start + completeLines + 1}.`
            : `Line ${start + 1} alone exceeds the limit and cannot be read further with this tool; use bash (e.g. \`sed -n '${start + 1}p' FILE | cut -c 1-20000\`) to read it in segments.`;
        return numbered + `\n... [Truncated at ${MAX_READ_CHARS} chars. File has ${lines.length} total lines. ${hint}]`;
      }
      if (end < lines.length) {
        const remaining = lines.length - end;
        return (
          numbered +
          `\n... [Showing lines ${start + 1}-${end} of ${lines.length}; ${remaining} more line${remaining === 1 ? "" : "s"}. ` +
          `Continue with offset=${end + 1}.]`
        );
      }
      return numbered;
    },
  });

  const writeTool = defineTool({
    name: "write",
    description: "Create a new file or overwrite an existing regular file completely.",
    inputSchema: z.object({
      file_path: z.string().min(1),
      content: z.string(),
    }),
    run: async ({ file_path, content }, { signal: sig }) => {
      sig?.throwIfAborted();
      const fullPath = await resolveWorkspacePath(file_path);
      await fs.mkdir(path.dirname(fullPath), { recursive: true });
      await atomicWriteFile(fullPath, content, sig);
      return `Successfully wrote ${Buffer.byteLength(content, "utf8")} bytes to ${file_path}`;
    },
  });

  // 换行策略（逐项判断）：
  // 1. 纯 CRLF 文件：无论走精确匹配还是 LF→CRLF 回退匹配，new_text 中的换行一律转为 CRLF，保持文件风格；
  // 2. LF 文件或混合换行文件：精确匹配时按原样替换（尊重调用方给出的换行）；
  //    精确匹配未命中而 CRLF 归一化后命中时，被替换区域本身是 CRLF，new_text 同步转为 CRLF。
  // 任何情况下都不会把已有的 "\r\n" 从中间劈开。
  // 批量语义：所有 edits 都在同一份原文上定位，每项必须唯一、区域互不重叠（相邻可以）；
  // 任一项失败则整体不写入，全部校验通过后一次原子提交。不做引号/空白/Unicode 归一化等模糊匹配。
  const editTool = defineTool({
    name: "edit",
    description:
      "Replace one or more exact, unique text blocks in an existing UTF-8 file in a single atomic write. " +
      "Every old_text is matched against the original file content (not against the result of earlier edits), " +
      "must match exactly once, and must not overlap any other edit; if any edit fails, nothing is written. " +
      "In files that use CRLF line endings exclusively, newlines in new_text are written as CRLF.",
    inputSchema: z.object({
      file_path: z.string().min(1),
      edits: z
        .array(
          z.object({
            old_text: z.string().min(1, "old_text cannot be empty"),
            new_text: z.string(),
          })
        )
        .min(1, "edits must contain at least one edit"),
    }),
    run: async ({ file_path, edits }, { signal: sig }) => {
      sig?.throwIfAborted();
      const fullPath = await resolveWorkspacePath(file_path);
      const stat = await fs.stat(fullPath);
      if (!stat.isFile()) {
        throw new Error(`Cannot edit non-regular file: ${file_path}`);
      }
      if (stat.size > MAX_READ_BYTES) {
        throw new Error(`File size (${stat.size} bytes) exceeds ${formatBytes(MAX_READ_BYTES)} limit.`);
      }

      const buf = await fs.readFile(fullPath, { signal: sig });
      const rawContent = decodeStrictUtf8(buf, file_path);

      const hasCrLf = rawContent.includes("\r\n");
      const hasBareLf = rawContent.replace(/\r\n/g, "").includes("\n");
      const isPureCrLfFile = hasCrLf && !hasBareLf;
      const label = (i: number) => (edits.length === 1 ? "'old_text'" : `edits[${i}].old_text`);

      const located = edits.map(({ old_text, new_text }, i) => {
        let targetOld = old_text;
        let matches = findMatchesWithoutSplittingCrLf(rawContent, targetOld);
        let matchedViaCrLfFallback = false;

        if (matches.length === 0 && hasCrLf) {
          // 精确匹配未命中且文件含 CRLF 时，尝试将 old_text 按 CRLF 归一化后匹配
          const crlfOld = old_text.replace(/\r?\n/g, "\r\n");
          if (crlfOld !== old_text) {
            const crlfMatches = findMatchesWithoutSplittingCrLf(rawContent, crlfOld);
            if (crlfMatches.length > 0) {
              targetOld = crlfOld;
              matches = crlfMatches;
              matchedViaCrLfFallback = true;
            }
          }
        }

        if (matches.length === 0) {
          throw new Error(`${label(i)} not found in ${file_path}. Read the file first to verify exact text. No edits were applied.`);
        }
        if (matches.length > 1) {
          throw new Error(
            `${label(i)} matched multiple locations in ${file_path}. Include more surrounding lines. No edits were applied.`
          );
        }

        const targetNew =
          isPureCrLfFile || matchedViaCrLfFallback ? new_text.replace(/\r?\n/g, "\r\n") : new_text;
        return { index: i, start: matches[0], end: matches[0] + targetOld.length, text: targetNew };
      });

      located.sort((a, b) => a.start - b.start);
      for (let k = 1; k < located.length; k++) {
        const prev = located[k - 1];
        const cur = located[k];
        if (cur.start < prev.end) {
          throw new Error(
            `edits[${prev.index}] and edits[${cur.index}] overlap in ${file_path}. ` +
              `Merge them into one edit or choose non-overlapping old_text. No edits were applied.`
          );
        }
      }

      const parts: string[] = [];
      let cursor = 0;
      for (const { start, end, text } of located) {
        parts.push(rawContent.slice(cursor, start), text);
        cursor = end;
      }
      parts.push(rawContent.slice(cursor));
      const updated = parts.join("");

      if (updated === rawContent) {
        throw new Error(`No changes made to ${file_path}: the combined edits produce identical content.`);
      }

      // rawContent 保留了 BOM（见 utf8FatalDecoder），写回时原样保留
      await atomicWriteFile(fullPath, updated, sig);
      // 首个实际变化所在的行号：取原文与新内容公共前缀的结束处（匹配文本可能含未变化的上下文，首项也可能是无变化编辑）。
      // 上面已排除 updated === rawContent，因此必有差异位置。
      let firstChangedLine = 1;
      for (let i = 0; i < rawContent.length && i < updated.length && rawContent[i] === updated[i]; i++) {
        if (rawContent.charCodeAt(i) === 10) firstChangedLine++;
      }
      return {
        content:
          edits.length === 1
            ? `Successfully edited ${file_path}`
            : `Successfully applied ${edits.length} edits to ${file_path}`,
        details: { path: file_path, edits: edits.length, firstChangedLine },
      };
    },
  });

  const bashTool = defineTool({
    name: "bash",
    description:
      "Execute a foreground bash command in the workspace. Background child processes in the same process group are terminated when the command exits. " +
      `Long output is reduced to its first ${formatBytes(BASH_HEAD_BYTES)} and last ${formatBytes(BASH_TAIL_BYTES)}` +
      (resolveBashLogDir
        ? `. Output logging is best-effort and limited to ${formatBytes(MAX_BASH_LOG_BYTES)}; when a log is available, ` +
          "its path and whether it is complete are included in the result."
        : "."),
    inputSchema: z.object({
      command: z.string().min(1),
      timeout_ms: z.number().int().min(100).max(300_000).optional(),
    }),
    // 非零退出不抛错：返回 isError 结果，模型看到的文本与原先一致，宿主另从 details 取得退出码与日志位置。
    // 执行期间由 bash.ts 的进度报告器节流，把新增输出作为 tool_progress 的 text 报告（增量，过多时只保留末尾）。
    run: async ({ command, timeout_ms }, { signal: sig, reportProgress }) => {
      const progress = reportProgress ? createThrottledTextReporter(reportProgress) : undefined;
      let result: BashRunResult;
      try {
        result = await runBashProcessGroup(command, root, timeout_ms ?? 30_000, {
          env: options?.env,
          signal: sig,
          logDir: resolveBashLogDir,
          onOutput: progress?.push,
        });
      } finally {
        progress?.flush();
      }
      const { exitCode, exitSignal, output, log } = result;
      const details = { exitCode, exitSignal, ...(log ? { log } : {}) };
      if (exitCode !== 0) {
        const statusDesc =
          exitCode !== null
            ? `exit code ${exitCode}`
            : `signal ${exitSignal ?? "UNKNOWN"}`;
        const body = output || `(Command terminated by ${statusDesc} with no output)`;
        return { content: `Command failed with ${statusDesc}:\n${body}`, isError: true, details };
      }
      return { content: output || "(Command exited with code 0)", details };
    },
  });

  return [readTool, writeTool, editTool, bashTool];
}
