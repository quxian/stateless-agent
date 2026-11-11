import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import * as fs from "node:fs/promises";
import { createWriteStream, mkdirSync, mkdtempSync, type WriteStream } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import { randomBytes } from "node:crypto";

/**
 * 无状态的模型—工具调用循环：会话历史、工具装配与生命周期均由宿主管理。
 * 核心流程：准备本轮请求（prepareMessages）→ 请求模型 → 串行执行工具 → 保存结果 → 继续或结束。
 *
 * 文件结构分层：
 * - 上半部分（第 1–3 节）：无状态 Agent 核心（公开类型契约、工具定义辅助、历史回合/缓存断点协议、runAgent 主循环）。
 * - 下半部分（第 4–5 节）：内置本地工具实现（头尾缓冲区与受控进程组执行器、路径/图片/原子写/CRLF 编辑，以及独立导出的 createReadTool / createWriteTool / createEditTool / createBashTool）。
 *
 * 运行契约（宿主职责边界）：
 * - 会话历史由调用方传入、由返回值/onMessage 带出；runAgent 不持有跨调用状态。
 * - 工具由宿主显式传入与组装（默认无工具）；自定义工具可用 defineTool 定义，内置本地工具（createReadTool / createWriteTool / createEditTool / createBashTool）按需独立创建，工作区只属于具体工具实例。
 *   历史中含 tool_use/tool_result 时，请求必须定义相应工具（API 要求），宿主应保持工具集与历史一致。
 * - 上下文整理（裁剪、摘要、检索注入、图片替换）由宿主通过 prepareMessages 完成，只影响本次请求，不改写完整历史；
 *   约束见 AgentOptions.prepareMessages。
 * - 单次运行内工具严格串行执行。但工作区按“单写入者”设计：多个 Agent 实例、并发直接调用工具或外部编辑器
 *   同时修改同一文件时，可能发生“最后提交者覆盖”的丢更新；多写入者场景需要宿主提供文件级协调或版本冲突检测。
 * - 路径检查只用于防误操作，不是安全沙箱（存在 TOCTOU 窗口、硬链接不受约束、bash 可执行任意命令）。
 *   用于不可信代码或多租户环境时，隔离必须由操作系统权限、容器等执行环境承担。
 * - write/edit 以 rename 原子提交（细节见 atomicWriteFile）；不解决并发覆盖。
 * - bash 输出超出头尾摘要时，尽力保存输出日志（目录见 createBashTool 的 logDir）；单个日志最多 64 MiB，
 *   保存失败时不返回路径；保留的日志不会自动删除，由宿主清理。
 * - read 返回的图片会留在会话历史中，之后每轮请求都会重新发送。请求体大小与单次请求图片数量均受 API 限制，
 *   长会话中历史的压缩/裁剪由宿主负责（prepareMessages）。图片处理依赖 sharp，在首次读图时才加载。
 * - 取消：宿主收到 SIGTERM/SIGINT 时应对每次运行的 AbortController 调用 abort() 并等待 runAgent 结束，
 *   以便子进程组完成 SIGTERM→SIGKILL 清理；模块内的 process "exit" 兜底（首次启动 bash 子进程时惰性注册）只做同步 SIGKILL，不能替代完整关闭流程。
 * - 完成与取消同时发生（无需继续工具循环的最终 assistant 回复已完整返回、保存期间触发取消）时，以完成为准，stopReason 报告模型的停止原因。
 * - 平台：bash 工具仅支持 POSIX（依赖 /bin/bash 与进程组语义），已在 macOS 与 Linux 上做过本地执行测试；
 *   Windows 宿主可仅装配文件类工具而不挂载 createBashTool（若调用则直接报错，核心循环与其他工具不受影响）。
 */

// ============================================================================
// 1. 类型契约、运行错误（携带状态快照）与工具定义辅助
// ============================================================================
export type AgentStopReason =
  | Anthropic.Beta.Messages.BetaStopReason
  | "max_turns"
  | "aborted"
  | null;

export type ToolOutput = Anthropic.Beta.Messages.BetaToolResultBlockParam["content"];

// 工具可选的结构化返回：content 进入模型历史（模型后续需要的信息必须写在这里）；
// details 只随 tool_end 事件交给宿主（不进入历史，也不保证可靠持久化）；
// isError 允许不抛错地返回带结构信息的失败结果（记为 is_error 的工具结果）。
export interface AgentToolResult {
  content: ToolOutput;
  details?: unknown;
  isError?: boolean;
}

export interface AgentToolProgress {
  text?: string;
  details?: unknown;
}

export interface AgentToolContext {
  // 工具因取消而失败时，应抛出 signal.reason、name 为 "AbortError" 的错误，或以 signal.reason 为 cause 的错误，
  // 这类错误记为“执行被中断”；其他错误即使发生在取消之后，也按工具自身的确定失败原样记录。
  signal?: AbortSignal;
  toolUseId: string;
  // 执行期间报告进度，转为 tool_progress 观察事件（工具名与 id 由执行器补齐）。
  // 仅在宿主订阅了 onEvent 时提供（工具应以 reportProgress?.() 调用，未提供时可跳过进度相关计算）；
  // 不等待宿主处理、异常被隔离、工具返回后的调用被忽略；高频进度由工具自行节流。details 按引用传递。
  reportProgress?: (progress: AgentToolProgress) => void;
}

export interface AgentTool<TInput = any> {
  name: string;
  description: string;
  input_schema: Anthropic.Beta.Messages.BetaTool["input_schema"];
  // 可选的显式缓存断点；任一工具设置后即视为调用方接管整体缓存配置，runAgent 不再为 system/tools/messages 自动添加默认断点
  cache_control?: Anthropic.Beta.Messages.BetaTool["cache_control"];
  parse: (input: unknown) => TInput;
  run: (
    input: TInput,
    context: AgentToolContext
  ) => Promise<ToolOutput | AgentToolResult> | ToolOutput | AgentToolResult;
}

// 观察事件分为两个阶段：
// 1. 模型流式生成阶段（thinking_delta / text_delta / tool_call_start / tool_input_delta）：随模型输出实时发出。
//    tool_call_start 在模型开始输出某个 tool_use 块时触发，tool_input_delta 随参数原始 JSON 片段（非空 partial_json）到达触发；
//    若流中途取消/报错、因 max_tokens 等原因截断、工具名未知或同批前序工具执行后已取消，已发出流式事件的调用不会进入本地执行阶段。
// 2. 本地工具执行阶段（tool_start / tool_progress / tool_end）：在完整响应到达后按串行执行顺序发出；每次 tool_start 必有配对的 tool_end。
export type AgentEvent =
  | { type: "thinking_delta"; delta: string }
  | { type: "text_delta"; delta: string }
  | { type: "tool_call_start"; id: string; name: string }
  | { type: "tool_input_delta"; id: string; name: string; delta: string }
  | { type: "tool_start"; id: string; name: string; input: unknown }
  | { type: "tool_progress"; id: string; name: string; text?: string; details?: unknown }
  | { type: "tool_end"; id: string; name: string; isError: boolean; outputSummary: string; details?: unknown };

export interface AgentUsage {
  // 仅累计成功取得完整响应（finalMessage）的轮次；流中途出错或取消的轮次不计入，不能直接当作计费账单
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
}

export interface PrepareMessagesContext {
  // 完整历史的浅拷贝（含本次运行已新增的消息）。请勿就地修改其中的消息对象——它们与完整历史共享。
  messages: readonly Anthropic.Beta.Messages.BetaMessageParam[];
  // 本次运行内即将发出的第几次模型请求（从 1 开始）
  turn: number;
  signal?: AbortSignal;
}

export interface AgentOptions {
  client: Anthropic;
  model: string;
  // 调用方持有的完整历史（不会被就地修改）。允许连续同角色消息，按 API 语义视为同一回合。
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  // 本次运行可用的工具（默认无工具，由宿主按需组装）。自定义工具可用 defineTool 定义；内置本地工具见 createReadTool / createWriteTool / createEditTool / createBashTool。
  tools?: AgentTool[];
  // 默认使用不含环境信息的通用提示词；工作目录等上下文由宿主按需写入
  systemPrompt?: string;
  maxTokens?: number;
  maxTurns?: number;
  thinking?: Anthropic.Beta.Messages.BetaThinkingConfigParam;
  // 是否自动在 system、末尾工具与末尾消息上添加默认 ephemeral 缓存断点（默认 true）。每次请求按最终发送的消息与 tools 判断：
  // 任一处已有显式 cache_control 时即整体跳过全部默认断点（避免超出 API 断点上限）；设为 false 仅关闭自动添加，已有配置仍原样发送。
  autoCacheBreakpoints?: boolean;
  signal?: AbortSignal;
  // 本次运行每新增一条完整消息时按序调用（请勿就地修改消息对象）；若抛错将中止后续动作并通过 AgentRunError 带出快照。
  // 该 Promise 会被完整等待，不会因 signal 取消而被放弃——宿主需自行保证保存操作有合理期限。
  // 这是“完整消息”持久化点而非工具事务日志：同一批次中已完成工具的结果，要等整批结束后才作为一条消息落盘。
  onMessage?: (message: Readonly<Anthropic.Beta.Messages.BetaMessageParam>) => void | Promise<void>;
  // 观察侧事件回调：同步异常与异步拒绝均被隔离，不阻塞主流程返回（runAgent 返回时异步回调可能尚未完成，需要可靠投递的宿主应自行排队并等待处理完成）
  onEvent?: (event: AgentEvent) => void | Promise<void>;
  // 每次模型请求（含第一次）之前调用，返回本次实际发送的消息；只影响本次请求，完整历史保持不变。
  // 用于裁剪/摘要较早历史、注入临时检索材料、替换旧图片等。约束（违反时以 AgentRunError 中止）：
  // - 进行中的工具回合（当历史中最后一个 assistant 回合含 tool_use 时，从该回合首条消息到末尾的全部消息）必须原样位于返回结果末尾——
  //   开启 thinking 时该回合的 thinking 签名必须原样回传；临时材料可插在它之前。最后一个 assistant 回合不含 tool_use 时无此限制。
  // - 每个 tool_result 都必须对应紧邻其前的 assistant 回合中的 tool_use（裁剪时成对删除）。
  // 这里只校验本地消息结构与进行中的工具回合；模型相关的 thinking 前缀绑定约束仍由宿主满足：
  // 部分模型的 thinking 还绑定生成它时的 system、tools 与之前的消息，保留 thinking 块原样并不保证可以改写它之前的上下文。
  // 面向这类模型时，宿主应在工具回合进行期间保持已有请求前缀稳定（例如不要逐轮替换插在前面的检索材料），
  // 或采用该模型支持的上下文压缩方式。
  // 返回后仍会经过缺失结果补全与自动缓存断点处理。核心不保存任何状态：调用模型做摘要等昂贵操作应由宿主按历史前缀缓存；
  // 每轮都改写较早前缀会让提示缓存失效，宜“阶梯式”压缩（前缀保持稳定，偶尔整体重写）。
  // 取消时记为 aborted；其他异常中止运行并通过 AgentRunError 带出状态。
  prepareMessages?: (
    context: PrepareMessagesContext
  ) => Promise<Anthropic.Beta.Messages.BetaMessageParam[]> | Anthropic.Beta.Messages.BetaMessageParam[];
  // 每轮整批工具结果落盘后调用的增量上下文钩子（批次之后的刷新点，而非每个工具调用之前的检查点）；
  // 返回非空字符串时作为 user 消息写入完整历史（返回 null 或空字符串则忽略）。非取消异常将中止运行以防遗漏关键指令。
  // 覆盖范围：运行开始时与模型给出最终回复（无工具调用）之后都不会调用；同一批次中，后面的工具在钩子之前就已执行。
  // 因此它不能替代“副作用发生前检查最新状态”——例如发送消息前检查是否有新指令，应放在该工具内部、紧邻实际副作用处。
  getContextInjection?: (context: { signal?: AbortSignal }) => Promise<string | null>;
}

export interface AgentResult {
  // 调用方历史 + 本次运行新增的消息（含末尾恢复消息）。
  // 注意：历史中段缺失工具结果的补全只存在于每次请求的副本中，不会写回这里，也不会经过 onMessage。
  // 只有完整取得的 assistant 响应才会写入：流中途出错或取消时，已通过流式事件（thinking_delta/text_delta/tool_call_start/tool_input_delta）
  // 发出的部分输出不会出现在这里；需要保留“用户已看到的内容”的宿主，应在展示记录中自行保存，而不是把半成品当作模型上下文。
  // 请求级重试由注入的 Anthropic 客户端配置（如 maxRetries）负责，runAgent 不做整轮重试。
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  // 本次运行最后一条完整 assistant 响应中的文本（多段以 "\n" 拼接；该响应无文本块或本次未取得响应时为 ""，不回溯传入的初始历史）
  lastText: string;
  stopReason: AgentStopReason;
  // 本次运行实际发起的模型请求次数（发出请求即计入，含流中途取消的轮次）
  turns: number;
  usage: AgentUsage;
}

export class AgentRunError extends Error {
  readonly messages: Anthropic.Beta.Messages.BetaMessageParam[];
  readonly turns: number;
  readonly usage: AgentUsage;

  constructor(
    message: string,
    state: { messages: Anthropic.Beta.Messages.BetaMessageParam[]; turns: number; usage: AgentUsage },
    options?: { cause?: unknown }
  ) {
    super(message, options);
    this.name = "AgentRunError";
    this.messages = [...state.messages];
    this.turns = state.turns;
    this.usage = { ...state.usage };
  }
}

// 用 Zod 对象定义 AgentTool：JSON Schema 由 Zod 生成，parse 负责运行时校验
export function defineTool<S extends z.ZodObject>(spec: {
  name: string;
  description: string;
  inputSchema: S;
  cache_control?: AgentTool["cache_control"];
  run: (
    input: z.infer<S>,
    context: AgentToolContext
  ) => Promise<ToolOutput | AgentToolResult> | ToolOutput | AgentToolResult;
}): AgentTool<z.infer<S>> {
  return {
    name: spec.name,
    description: spec.description,
    input_schema: z.toJSONSchema(spec.inputSchema, { reused: "ref" }) as AgentTool["input_schema"],
    ...(spec.cache_control ? { cache_control: spec.cache_control } : {}),
    parse: (input) => spec.inputSchema.parse(input),
    run: spec.run,
  };
}

// ============================================================================
// 2. 历史回合恢复、请求副本构造与缓存断点、观察快照等核心辅助函数
// ============================================================================
const DEFAULT_MAX_TURNS = 50;
const DEFAULT_MAX_TOKENS = 8192;
const THINKING_OUTPUT_HEADROOM_TOKENS = 4096;
const TOOL_SUMMARY_MAX_CHARS = 500;
const RECOVERY_RESULT_TEXT = "从上次未完成的运行中恢复：该工具调用的执行状态未知。";

// 避免在 JavaScript 字符串截断处切开 UTF-16 代理对（如 Emoji）
function sliceStringSafe(str: string, maxChars: number): string {
  if (str.length <= maxChars) return str;
  let end = maxChars;
  const code = str.charCodeAt(end - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    end--;
  }
  return str.slice(0, end);
}

// API 语义下的“回合”：相邻的同角色消息会被服务端合并为一个回合。
// system 消息不属于任何回合，也不打断回合。
interface ConversationTurn {
  role: Anthropic.Beta.Messages.BetaMessageParam["role"];
  indices: number[];
}

function groupIntoTurns(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]): ConversationTurn[] {
  const turns: ConversationTurn[] = [];
  for (let i = 0; i < messages.length; i++) {
    const role = messages[i].role;
    if (role === "system") continue;
    const last = turns[turns.length - 1];
    if (last && last.role === role) last.indices.push(i);
    else turns.push({ role, indices: [i] });
  }
  return turns;
}

// 按出现顺序收集一个回合内全部 tool_use id（去重）
function collectToolUseIds(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[], indices: number[]): string[] {
  const ids = new Set<string>();
  for (const idx of indices) {
    const content = messages[idx].content;
    if (!Array.isArray(content)) continue;
    for (const b of content) {
      if (b.type === "tool_use") ids.add(b.id);
    }
  }
  return [...ids];
}

function collectToolResultIds(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[], indices: number[]): Set<string> {
  const ids = new Set<string>();
  for (const idx of indices) {
    const content = messages[idx].content;
    if (!Array.isArray(content)) continue;
    for (const b of content) {
      if (b.type === "tool_result") ids.add(b.tool_use_id);
    }
  }
  return ids;
}

function makeRecoveryResults(ids: string[]): Anthropic.Beta.Messages.BetaToolResultBlockParam[] {
  return ids.map((id) => ({
    type: "tool_result" as const,
    tool_use_id: id,
    is_error: true,
    content: RECOVERY_RESULT_TEXT,
  }));
}

// 末尾恢复：若历史以 assistant 回合结束（可能由多条连续 assistant 消息组成），
// 为该回合内全部 tool_use 补一条“状态未知”的结果消息。该消息会通过 appendMessage 持久化。
function buildTrailingRecoveryMessage(
  messages: Anthropic.Beta.Messages.BetaMessageParam[]
): Anthropic.Beta.Messages.BetaMessageParam | null {
  const turns = groupIntoTurns(messages);
  const lastTurn = turns[turns.length - 1];
  if (!lastTurn || lastTurn.role !== "assistant") return null;

  const toolUseIds = collectToolUseIds(messages, lastTurn.indices);
  if (toolUseIds.length === 0) return null;

  return { role: "user", content: makeRecoveryResults(toolUseIds) };
}

// 检测消息中是否已显式配置了 cache_control
function messagesHaveCacheControl(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]): boolean {
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if ("cache_control" in block && block.cache_control) return true;
    }
  }
  return false;
}

// 结构相等（忽略值为 undefined 的属性与属性顺序），用于校验 prepareMessages 是否原样保留进行中的工具回合
function structurallyEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ra = a as Record<string, unknown>;
  const rb = b as Record<string, unknown>;
  const ka = Object.keys(ra).filter((k) => ra[k] !== undefined);
  const kb = Object.keys(rb).filter((k) => rb[k] !== undefined);
  return ka.length === kb.length && ka.every((k) => Object.hasOwn(rb, k) && structurallyEqual(ra[k], rb[k]));
}

// 进行中的工具回合：最后一个 assistant 回合含 tool_use 时，从该回合首条消息到历史末尾；否则为空
function inProgressToolTurn(
  messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]
): Anthropic.Beta.Messages.BetaMessageParam[] {
  const turns = groupIntoTurns(messages);
  for (let t = turns.length - 1; t >= 0; t--) {
    if (turns[t].role !== "assistant") continue;
    if (collectToolUseIds(messages, turns[t].indices).length === 0) return [];
    return messages.slice(turns[t].indices[0]);
  }
  return [];
}

// 校验 prepareMessages 的返回值（失败时抛错，由主循环包装为 AgentRunError）
function validatePreparedMessages(
  prepared: unknown,
  history: readonly Anthropic.Beta.Messages.BetaMessageParam[]
): asserts prepared is Anthropic.Beta.Messages.BetaMessageParam[] {
  if (!Array.isArray(prepared) || prepared.length === 0) {
    throw new Error("prepareMessages 必须返回非空的消息数组。");
  }
  const tail = inProgressToolTurn(history);
  const offset = prepared.length - tail.length;
  if (offset < 0 || !tail.every((msg, i) => structurallyEqual(prepared[offset + i], msg))) {
    throw new Error(
      "prepareMessages 必须在返回消息的末尾原样保留进行中的工具回合（当历史中最后一个 assistant 回合含 tool_use 时，从该回合首条消息到末尾的全部消息）。"
    );
  }
}

// 每个 tool_result 都必须对应紧邻其前的 assistant 回合中的 tool_use；否则 API 会拒绝请求。
// 这里提前给出指明 id 的错误，而不是悄悄删除（裁剪策略属于宿主）。缺失结果的补全由 buildApiRequestMessages 负责。
function assertToolResultsPaired(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]): void {
  const turns = groupIntoTurns(messages);
  for (let t = 0; t < turns.length; t++) {
    if (turns[t].role !== "user") continue;
    const prev = turns[t - 1];
    const known = new Set(prev?.role === "assistant" ? collectToolUseIds(messages, prev.indices) : []);
    for (const id of collectToolResultIds(messages, turns[t].indices)) {
      if (!known.has(id)) {
        throw new Error(
          `请求消息中包含 tool_use_id 为 '${id}' 的 tool_result，但紧邻其前缺少包含对应 tool_use 的 assistant 回合。裁剪历史时请成对删除 tool_use 与 tool_result。`
        );
      }
    }
  }
}

// 构造发往 API 的请求副本（不修改、不持久化调用方历史）：
// - 按完整回合检查：assistant 回合内所有 tool_use，只要在紧随其后的整个 user 回合中任一条消息里有结果即视为已覆盖；
// - 仅对确实缺失结果的调用补造“状态未知”结果：若存在后续 user 回合，则前插到该回合首条消息的开头
//   （保证 tool_result 位于普通文本之前）；否则在该 assistant 回合之后插入一条新的 user 消息；
// - 不合并、不拆分调用方的消息，不改动任何内容块（thinking 及其签名原样透传），system 消息原样保留。
function buildApiRequestMessages(
  messages: Anthropic.Beta.Messages.BetaMessageParam[],
  addCacheBreakpoint: boolean
): Anthropic.Beta.Messages.BetaMessageParam[] {
  const turns = groupIntoTurns(messages);
  const prependByIndex = new Map<number, Anthropic.Beta.Messages.BetaToolResultBlockParam[]>();
  const insertAfterIndex = new Map<number, Anthropic.Beta.Messages.BetaToolResultBlockParam[]>();

  for (let t = 0; t < turns.length; t++) {
    const turn = turns[t];
    if (turn.role !== "assistant") continue;
    const toolUseIds = collectToolUseIds(messages, turn.indices);
    if (toolUseIds.length === 0) continue;

    const nextTurn = turns[t + 1]; // 角色交替，存在时必为 user 回合
    const covered = nextTurn ? collectToolResultIds(messages, nextTurn.indices) : new Set<string>();
    const missingIds = toolUseIds.filter((id) => !covered.has(id));
    if (missingIds.length === 0) continue;

    const synthetic = makeRecoveryResults(missingIds);
    if (nextTurn) {
      prependByIndex.set(nextTurn.indices[0], synthetic);
    } else {
      insertAfterIndex.set(turn.indices[turn.indices.length - 1], synthetic);
    }
  }

  const normalized: Anthropic.Beta.Messages.BetaMessageParam[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "system") {
      normalized.push({
        ...msg,
        ...(Array.isArray(msg.content) ? { content: msg.content.map((b) => ({ ...b })) } : {}),
      });
      continue;
    }

    const clonedBlocks: Anthropic.Beta.Messages.BetaContentBlockParam[] =
      typeof msg.content === "string"
        ? [{ type: "text", text: msg.content }]
        : msg.content.map((b) => ({ ...b }));

    const prefix = prependByIndex.get(i);
    normalized.push({ role: msg.role, content: prefix ? [...prefix, ...clonedBlocks] : clonedBlocks });

    const after = insertAfterIndex.get(i);
    if (after) {
      normalized.push({ role: "user", content: after });
    }
  }

  if (addCacheBreakpoint && normalized.length > 0) {
    const lastMsg = normalized[normalized.length - 1];
    // 仅作用于当前回合的临时 system 消息（clear_at: "next_user_message"）禁止携带 cache_control，跳过自动断点
    const isTransientSystem =
      lastMsg.role === "system" && "clear_at" in lastMsg && lastMsg.clear_at === "next_user_message";
    if (!isTransientSystem && Array.isArray(lastMsg.content) && lastMsg.content.length > 0) {
      const lastIdx = lastMsg.content.length - 1;
      const lastBlock = lastMsg.content[lastIdx];
      const cacheable = lastBlock.type !== "thinking" && lastBlock.type !== "redacted_thinking";
      const alreadyMarked = "cache_control" in lastBlock && Boolean(lastBlock.cache_control);
      if (cacheable && !alreadyMarked) {
        lastMsg.content[lastIdx] = {
          ...lastBlock,
          cache_control: { type: "ephemeral" },
        } as Anthropic.Beta.Messages.BetaContentBlockParam;
      }
    }
  }

  return normalized;
}

function raceWithSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  promise.catch(() => {});
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(signal.reason);

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (val) => {
        signal.removeEventListener("abort", onAbort);
        resolve(val);
      },
      (err) => {
        signal.removeEventListener("abort", onAbort);
        reject(err);
      }
    );
  });
}

// 按抛出的错误本身判断是否由取消导致，而不是按 catch 时 signal 是否已取消：
// 工具已得到确定结果（如命令以非零码退出）后才到达的取消，不能把真实失败改报为中断。
function isAbortCausedError(err: unknown, signal?: AbortSignal): boolean {
  if (!signal?.aborted) return false;
  // reason 可以是任意值（含 null、NaN），用 Object.is 比较
  if (Object.is(err, signal.reason)) return true;
  const e = err as { name?: unknown; cause?: unknown } | null;
  return e?.name === "AbortError" || (e?.cause !== undefined && Object.is(e.cause, signal.reason));
}

// ToolOutput 只可能是字符串、数组或 undefined，因此“带 content 属性的普通对象”可无歧义地识别为结构化结果
function isAgentToolResult(value: unknown): value is AgentToolResult {
  return typeof value === "object" && value !== null && !Array.isArray(value) && "content" in value;
}

function summarizeToolOutput(output: ToolOutput): string {
  if (output === undefined || output === null) return "";
  if (typeof output === "string") return sliceStringSafe(output, TOOL_SUMMARY_MAX_CHARS);
  try {
    // 图片块只保留类型说明，避免把 Base64 数据塞进观察事件
    const summary = output.map((block) =>
      block.type === "image"
        ? { type: "text", text: `[图片: ${block.source.type === "base64" ? block.source.media_type : block.source.type}]` }
        : block
    );
    return sliceStringSafe(JSON.stringify(summary) ?? "", TOOL_SUMMARY_MAX_CHARS);
  } catch {
    return "[复杂工具输出]";
  }
}

// ============================================================================
// 3. 核心 Agent 执行函数（流式请求、串行工具执行与主循环）
// ============================================================================
type EmitEvent = (build: () => AgentEvent) => void;
type BetaStreamParams = Parameters<Anthropic["beta"]["messages"]["stream"]>[0];

interface ToolBatchExecutionContext {
  toolMap: ReadonlyMap<string, AgentTool>;
  toolNamesList: string;
  signal?: AbortSignal;
  hasEventSubscriber: boolean;
  emit: EmitEvent;
}

function buildSkippedToolResults(
  blocks: readonly Anthropic.Beta.Messages.BetaToolUseBlock[],
  reason: string
): Anthropic.Beta.Messages.BetaToolResultBlockParam[] {
  return blocks.map((b) => ({
    type: "tool_result" as const,
    tool_use_id: b.id,
    is_error: true,
    content: reason,
  }));
}

// 发起单轮流式请求，并在有订阅者时桥接增量观察事件，返回完整最终响应
function streamModelTurn(
  client: Anthropic,
  params: BetaStreamParams,
  signal: AbortSignal | undefined,
  subscribeEvents: boolean,
  emit: EmitEvent
): Promise<Anthropic.Beta.Messages.BetaMessage> {
  const stream = client.beta.messages.stream(params, { signal });
  if (subscribeEvents) {
    // 直接从底层 streamEvent 读取 tool_use 块起点与原始 partial_json 片段：
    // SDK 的 contentBlock 在块结束时才触发，而 inputJson 会在每次增量上全量重解析累积 JSON（大参数时退化为 O(n²)）且不带 id/name。
    const streamingToolCalls = new Map<number, { id: string; name: string }>();
    stream
      .on("thinking", (delta) => emit(() => ({ type: "thinking_delta", delta })))
      .on("text", (delta) => emit(() => ({ type: "text_delta", delta })))
      .on("streamEvent", (event) => {
        if (event.type === "message_start") {
          streamingToolCalls.clear();
        } else if (event.type === "content_block_start" && event.content_block.type === "tool_use") {
          const { id, name } = event.content_block;
          streamingToolCalls.set(event.index, { id, name });
          emit(() => ({ type: "tool_call_start", id, name }));
        } else if (
          event.type === "content_block_delta" &&
          event.delta.type === "input_json_delta" &&
          event.delta.partial_json
        ) {
          const call = streamingToolCalls.get(event.index);
          if (call) {
            const delta = event.delta.partial_json;
            emit(() => ({ type: "tool_input_delta", id: call.id, name: call.name, delta }));
          }
        } else if (event.type === "content_block_stop") {
          streamingToolCalls.delete(event.index);
        }
      });
  }
  return stream.finalMessage();
}

// 执行单个 tool_use 块：保证每次发出 tool_start 后必发出配对的 tool_end
async function executeSingleToolBlock(
  block: Anthropic.Beta.Messages.BetaToolUseBlock,
  ctx: ToolBatchExecutionContext
): Promise<Anthropic.Beta.Messages.BetaToolResultBlockParam> {
  const { toolMap, toolNamesList, signal, hasEventSubscriber, emit } = ctx;

  if (signal?.aborted) {
    return {
      type: "tool_result",
      tool_use_id: block.id,
      content: "因运行已取消，该工具调用已跳过（未启动）。",
      is_error: true,
    };
  }

  const tool = toolMap.get(block.name);
  if (!tool) {
    return {
      type: "tool_result",
      tool_use_id: block.id,
      content: `错误：未知工具 '${block.name}'。可用工具：${toolNamesList || "（无）"}`,
      is_error: true,
    };
  }

  // 观察者拿独立的输入快照（与下方执行用副本相互隔离），防止日志器就地修改参数影响实际工具执行
  emit(() => ({
    type: "tool_start",
    id: block.id,
    name: block.name,
    input: structuredClone(block.input),
  }));

  let resultBlock: Anthropic.Beta.Messages.BetaToolResultBlockParam;
  let details: unknown;
  if (signal?.aborted) {
    // onEvent(tool_start) 同步回调期间可能触发取消：工具尚未执行，跳过运行但仍发出下方配对的 tool_end
    resultBlock = {
      type: "tool_result",
      tool_use_id: block.id,
      content: "因运行在启动前取消，该工具调用已跳过（未执行）。",
      is_error: true,
    };
  } else {
    // 进度只在工具运行期间有效；工具返回（或抛错）后的调用被忽略
    let running = true;
    const reportProgress = (progress: AgentToolProgress) => {
      if (!running) return;
      emit(() => ({
        type: "tool_progress",
        id: block.id,
        name: block.name,
        ...(progress?.text !== undefined ? { text: progress.text } : {}),
        ...(progress?.details !== undefined ? { details: progress.details } : {}),
      }));
    };
    try {
      // 工具可就地修改自己的执行参数，但不能改写已记录在历史中的模型消息
      const parsed = tool.parse(structuredClone(block.input));
      const output = await tool.run(parsed, {
        signal,
        toolUseId: block.id,
        ...(hasEventSubscriber ? { reportProgress } : {}),
      });
      if (isAgentToolResult(output)) {
        resultBlock = {
          type: "tool_result",
          tool_use_id: block.id,
          content: output.content,
          ...(output.isError ? { is_error: true } : {}),
        };
        details = output.details;
      } else {
        resultBlock = { type: "tool_result", tool_use_id: block.id, content: output };
      }
    } catch (err: unknown) {
      const errMsg = isAbortCausedError(err, signal)
        ? "因运行已取消，该工具调用在执行中途被中断，可能已产生部分副作用；后续操作前请先核实当前状态。"
        : err instanceof Error
          ? err.message
          : String(err);
      resultBlock = { type: "tool_result", tool_use_id: block.id, content: errMsg, is_error: true };
    } finally {
      running = false;
    }
  }

  const finished = resultBlock;
  emit(() => ({
    type: "tool_end",
    id: block.id,
    name: block.name,
    isError: finished.is_error === true,
    outputSummary: summarizeToolOutput(finished.content),
    ...(details !== undefined ? { details } : {}),
  }));

  return finished;
}

async function executeToolBatch(
  blocks: readonly Anthropic.Beta.Messages.BetaToolUseBlock[],
  ctx: ToolBatchExecutionContext
): Promise<Anthropic.Beta.Messages.BetaToolResultBlockParam[]> {
  const results: Anthropic.Beta.Messages.BetaToolResultBlockParam[] = [];
  for (const block of blocks) {
    results.push(await executeSingleToolBlock(block, ctx));
  }
  return results;
}

export async function runAgent(options: AgentOptions): Promise<AgentResult> {
  const {
    client,
    model,
    signal,
    maxTurns = DEFAULT_MAX_TURNS,
    thinking,
    prepareMessages,
    getContextInjection,
    onMessage,
    onEvent,
  } = options;

  if (!Number.isInteger(maxTurns) || maxTurns < 1) {
    throw new Error(`maxTurns 无效（${maxTurns}）：必须为正整数。`);
  }

  const budgetTokens = thinking && "budget_tokens" in thinking ? thinking.budget_tokens : 0;
  const maxTokens = options.maxTokens ?? (budgetTokens ? budgetTokens + THINKING_OUTPUT_HEADROOM_TOKENS : DEFAULT_MAX_TOKENS);
  if (budgetTokens && maxTokens <= budgetTokens) {
    throw new Error(`maxTokens（${maxTokens}）必须大于 thinking.budget_tokens（${budgetTokens}）。`);
  }

  const messages: Anthropic.Beta.Messages.BetaMessageParam[] = [...options.messages];
  const tools = options.tools ?? [];
  const toolMap = new Map<string, AgentTool>(tools.map((t) => [t.name, t]));
  const toolNamesList = Array.from(toolMap.keys()).join(", ");

  // 自动缓存断点按每次请求的最终消息判断：调用方在 tools 或本次发送的消息中显式设置了 cache_control 时沿用其配置
  const autoCacheBreakpoints = options.autoCacheBreakpoints ?? true;
  const toolsHaveCacheControl = tools.some((t) => t.cache_control);

  const buildApiTools = (addDefaultBreakpoint: boolean): Anthropic.Beta.Messages.BetaTool[] =>
    tools.map((t, idx) => ({
      name: t.name,
      description: t.description,
      input_schema: t.input_schema,
      ...(t.cache_control
        ? { cache_control: t.cache_control }
        : addDefaultBreakpoint && idx === tools.length - 1
          ? { cache_control: { type: "ephemeral" as const } }
          : {}),
    }));

  const systemPrompt =
    options.systemPrompt ??
    (tools.length > 0
      ? `你是一个乐于助人的助手，可以调用工具完成任务。可用工具：${toolNamesList}。`
      : "你是一个乐于助人的助手。");

  let lastText = "";
  let stopReason: AgentStopReason = null;
  let turns = 0;
  const usage: AgentUsage = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 };

  // 观察事件：仅在有订阅者时才构造事件（含复制输入、生成摘要）；构造或回调中的同步异常与异步拒绝均被隔离
  const emit: EmitEvent = (build) => {
    if (!onEvent) return;
    try {
      Promise.resolve(onEvent(build())).catch(() => {});
    } catch {}
  };

  const appendMessage = async (msg: Anthropic.Beta.Messages.BetaMessageParam) => {
    messages.push(msg);
    if (onMessage) {
      await onMessage(msg);
    }
  };

  try {
    // 入口预取消检查：若调用前 signal 已取消，直接返回 aborted，不写入恢复消息
    if (signal?.aborted) {
      return { messages, lastText, stopReason: "aborted", turns, usage };
    }

    const trailingRecovery = buildTrailingRecoveryMessage(messages);
    if (trailingRecovery) {
      await appendMessage(trailingRecovery);
    }

    for (let turn = 0; turn < maxTurns; turn++) {
      if (signal?.aborted) {
        stopReason = "aborted";
        break;
      }

      // 准备本次请求的消息：默认即完整历史；prepareMessages 的结果只用于本次请求
      let requestMessages: Anthropic.Beta.Messages.BetaMessageParam[] = messages;
      if (prepareMessages) {
        let prepared: unknown;
        try {
          const pending = Promise.resolve().then(() =>
            prepareMessages({ messages: [...messages], turn: turns + 1, signal })
          );
          prepared = await raceWithSignal(pending, signal);
        } catch (err) {
          if (signal?.aborted) {
            stopReason = "aborted";
            break;
          }
          throw err;
        }
        if (signal?.aborted) {
          stopReason = "aborted";
          break;
        }
        validatePreparedMessages(prepared, messages);
        requestMessages = prepared;
      }
      assertToolResultsPaired(requestMessages);

      const addDefaultCacheBreakpoints =
        autoCacheBreakpoints && !toolsHaveCacheControl && !messagesHaveCacheControl(requestMessages);
      const apiTools = buildApiTools(addDefaultCacheBreakpoints);
      const apiMessages = buildApiRequestMessages(requestMessages, addDefaultCacheBreakpoints);
      turns++;

      let response: Anthropic.Beta.Messages.BetaMessage;
      try {
        response = await streamModelTurn(
          client,
          {
            model,
            max_tokens: maxTokens,
            system: addDefaultCacheBreakpoints
              ? [{ type: "text", text: systemPrompt, cache_control: { type: "ephemeral" } }]
              : systemPrompt,
            ...(apiTools.length > 0 ? { tools: apiTools } : {}),
            messages: apiMessages,
            ...(thinking ? { thinking } : {}),
          },
          signal,
          Boolean(onEvent),
          emit
        );
      } catch (err) {
        if (signal?.aborted) {
          stopReason = "aborted";
          break;
        }
        throw err;
      }

      usage.inputTokens += response.usage.input_tokens;
      usage.outputTokens += response.usage.output_tokens;
      usage.cacheReadTokens += response.usage.cache_read_input_tokens ?? 0;
      usage.cacheCreationTokens += response.usage.cache_creation_input_tokens ?? 0;

      await appendMessage({ role: "assistant", content: response.content });

      lastText = response.content
        .filter((b): b is Anthropic.Beta.Messages.BetaTextBlock => b.type === "text")
        .map((b) => b.text)
        .join("\n");

      stopReason = response.stop_reason;

      const toolUseBlocks = response.content.filter(
        (b): b is Anthropic.Beta.Messages.BetaToolUseBlock => b.type === "tool_use"
      );

      if (toolUseBlocks.length === 0) {
        break;
      }

      // 非 tool_use 停止但响应已包含 tool_use 块时，必须补齐错误结果以维持历史配对：
      // - max_tokens 截断：写入提示后继续进入下一轮，让模型有机会分批或缩小参数重试；
      // - 其他停止原因（如 stop_sequence / refusal）：写入未执行说明后立即结束运行。
      if (response.stop_reason !== "tool_use" && response.stop_reason !== "max_tokens") {
        await appendMessage({
          role: "user",
          content: buildSkippedToolResults(
            toolUseBlocks,
            `模型生成因 '${response.stop_reason}' 停止，该工具调用未执行。`
          ),
        });
        break;
      }

      const userContent: Anthropic.Beta.Messages.BetaContentBlockParam[] =
        response.stop_reason === "max_tokens"
          ? buildSkippedToolResults(
              toolUseBlocks,
              "工具调用因达到 max_tokens 输出上限而被截断，未执行。请减小单次工具调用的参数内容（如分批处理）后重试。"
            )
          : await executeToolBatch(toolUseBlocks, {
              toolMap,
              toolNamesList,
              signal,
              hasEventSubscriber: Boolean(onEvent),
              emit,
            });

      // 1. 先记录并持久化本轮工具结果（或 max_tokens 截断未执行结果）
      await appendMessage({ role: "user", content: userContent });

      if (signal?.aborted) {
        stopReason = "aborted";
        break;
      }

      // 2. 获取增量上下文（若钩子抛错且非取消，向上抛出以避免遗漏关键控制指令）
      if (getContextInjection) {
        let injected: string | null = null;
        try {
          injected = await raceWithSignal(getContextInjection({ signal }), signal);
        } catch (err) {
          if (signal?.aborted) {
            stopReason = "aborted";
            break;
          }
          throw err;
        }
        if (injected && !signal?.aborted) {
          await appendMessage({
            role: "user",
            content: [{ type: "text", text: injected }],
          });
        }
      }

      // 若在增量上下文返回或保存注入消息期间触发了取消，优先报告 aborted 而非 max_turns
      if (signal?.aborted) {
        stopReason = "aborted";
        break;
      }

      if (turn === maxTurns - 1) {
        stopReason = "max_turns";
      }
    }

    return { messages, lastText, stopReason, turns, usage };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    throw new AgentRunError(message, { messages, turns, usage }, { cause: err });
  }
}

// ============================================================================
// 4. UTF-8 字符边界安全的头尾输出缓冲区与受控进程组执行器
// ============================================================================
const BASH_HEAD_BYTES = 16 * 1024;
const BASH_TAIL_BYTES = 32 * 1024;
// 输出超出头尾摘要时尽力保存输出日志；单个日志的磁盘上限，防止持续刷屏的命令写满磁盘
const MAX_BASH_LOG_BYTES = 64 * 1024 * 1024;
const KILL_GRACE_MS = 2_000;
const IO_DRAIN_GRACE_MS = 200;
const GROUP_POLL_MS = 40;
// bash 执行中进度报告的节流间隔与单次增量的字符上限（超出时只保留末尾并注明省略量）
const BASH_PROGRESS_INTERVAL_MS = 250;
const BASH_PROGRESS_MAX_CHARS = 8_192;

function errnoCode(err: unknown): string | undefined {
  return (err as NodeJS.ErrnoException | null)?.code;
}

// 将常量字节数格式化为文案（仅用于整 KiB/MiB 的限额常量）
function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${bytes / (1024 * 1024)} MiB` : `${bytes / 1024} KiB`;
}

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

// 节流的增量文本进度：push 累积新文本，每个间隔最多报告一次；累积超过上限时只保留末尾并注明省略量。
// flush 立即报告剩余内容并停止计时（工具返回前调用）。
function createThrottledTextReporter(report: (progress: AgentToolProgress) => void): {
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
    const text = skipped > 0 ? `[... 已省略 ${skipped} 个字符 ...]\n${pending}` : pending;
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
    // 省略量按解码后的字节计算（与 head/tail 同一口径）；总输出（totalBytes）为原始字节数
    const omitted = Math.max(1, this.decodedBytes - this.head.length - this.tail.length);
    const logNote = savedLog
      ? `；${savedLog.complete ? "完整输出" : `开头 ${formatBytes(MAX_BASH_LOG_BYTES)} 输出（超出上限部分未记录）`}已保存至 ${savedLog.path}` +
        `（可用 bash 检索或分段查看，如 rg/grep -n、sed -n 'X,Yp'、tail）`
      : "";
    return stripTrailingNewline(
      [
        this.head.toString("utf-8"),
        `\n... [中间已省略 ${omitted} 字节，总输出 ${this.totalBytes} 字节${logNote}] ...\n`,
        this.tail.toString("utf-8"),
      ].join("")
    );
  }
}

// 输出日志（单文件上限 MAX_BASH_LOG_BYTES）：输出接近头尾摘要容量前只在内存暂存（不超过约 48 KiB），超过后才创建日志文件，
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
let exitHookRegistered = false;

// 仅在首次实际启动受控进程组时注册退出兜底，避免只使用核心循环时产生模块加载期全局监听副作用
function trackActiveProcessGroup(pid: number): void {
  if (!exitHookRegistered) {
    exitHookRegistered = true;
    process.once("exit", () => {
      for (const activePid of activeProcessGroups) {
        sendProcessGroupSignal(activePid, "SIGKILL");
      }
    });
  }
  activeProcessGroups.add(pid);
}

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
  // 在继承 process.env 的基础上覆盖这些变量（CI、GIT_PAGER、GIT_TERMINAL_PROMPT 固定强制设为非交互值）；传入 {} 不会得到干净环境，不能当作白名单
  env?: NodeJS.ProcessEnv;
  signal?: AbortSignal;
  // 输出超出头尾摘要时保存输出日志的目录（按需调用，仅在确实需要创建日志时才求值）；
  // 省略则不保存日志。单个日志上限 64 MiB，保留到宿主清理为止（超时截断时也会保留并在错误消息中注明路径；取消、启动失败或未截断时不保留）。
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
      return reject(new Error("bash 工具仅支持 POSIX 平台（依赖 /bin/bash 与进程组）。"));
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
    if (pid) trackActiveProcessGroup(pid);

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
          return reject(new Error(`命令执行超时（超过 ${timeoutMs}ms 已终止）${output ? `：\n${output}` : "。"}`));
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

      // 注意：killEscalationTimer 保持 ref 状态（不调用 unref），防止宿主进程在等待强杀完成前提前退出事件循环
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

// ============================================================================
// 5. 本地文件/图片读写基础与独立导出的内置工具工厂（由宿主按需组装）
// ============================================================================
const MAX_READ_CHARS = 30_000;
const MAX_READ_BYTES = 5 * 1024 * 1024;
// 图片单独的磁盘大小上限：read 会先缩放再发送，内存由 MAX_IMAGE_PIXELS 约束
const MAX_IMAGE_FILE_BYTES = 50 * 1024 * 1024;
const MAX_IMAGE_EDGE = 2000;
const MAX_IMAGE_PIXELS = 40_000_000;
// 本工具对单张图片的保守传输预算（Base64 字符数），不是所有模型统一的 API 上限
const MAX_IMAGE_BASE64_CHARS = 4_500_000;

// ignoreBOM: true 表示“不要吞掉 BOM”，解码结果会保留开头的 U+FEFF，
// 从而 edit 写回时 BOM 字节原样保留（默认 false 会静默剥离 BOM）。
const utf8FatalDecoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

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
          throw new Error(`拒绝访问悬空符号链接（目标不存在）：${targetPath}`);
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
    throw new Error(`检测到二进制文件：${filePath}，无法作为 UTF-8 文本处理。`);
  }
  try {
    return utf8FatalDecoder.decode(buf);
  } catch {
    throw new Error(`文件不是合法的 UTF-8 文本：${filePath}。`);
  }
}

// 按文件头魔数识别 read 支持的图片格式（需至少 12 字节才能识别 WebP）
function isSupportedImage(head: Buffer): boolean {
  return (
    head.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) ||
    (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) ||
    ["GIF87a", "GIF89a"].includes(head.toString("ascii", 0, 6)) ||
    (head.toString("ascii", 0, 4) === "RIFF" && head.toString("ascii", 8, 12) === "WEBP")
  );
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

// sharp 是原生依赖，按需加载：只用文本工具的宿主不会因平台二进制缺失而无法加载本模块
async function loadSharp(): Promise<typeof import("sharp").default> {
  try {
    return (await import("sharp")).default;
  } catch (err) {
    throw new Error("读取图片依赖 'sharp' 包，但加载失败。", { cause: err });
  }
}

function base64Length(byteLength: number): number {
  return 4 * Math.ceil(byteLength / 3);
}

// PNG 是否确定为静态图：按 chunk 边界扫描到 IDAT 为止，出现 acTL（APNG 动画声明，规范要求位于 IDAT 之前）
// 或结构无法可靠解析时返回 false。仅用于决定能否原样返回字节，误判为“非静态”只会退回处理分支。
function isStaticPng(buf: Buffer): boolean {
  let offset = 8;
  while (offset + 8 <= buf.length) {
    const length = buf.readUInt32BE(offset);
    const type = buf.toString("latin1", offset + 4, offset + 8);
    if (type === "acTL") return false;
    if (type === "IDAT") return true;
    offset += 12 + length;
  }
  return false;
}

// 自动转正、等比缩放到 MAX_IMAGE_EDGE 以内（不裁剪、不放大），只取第一帧。
// - 已合规的普通静态 JPEG/PNG/WebP（无需转正/缩放、8 位 sRGB 或灰度、无 ICC、体积在预算内）经完整解码校验后
//   原样返回压缩字节，保留原始编码质量；注意这也会保留 EXIF/XMP 等元数据（处理分支会按 sharp 默认移除）。
// - 其余情况输出尽量沿用 API 支持的源格式（JPEG/WebP 原格式，PNG/GIF 输出 PNG）；
//   超出预算时从原图管线重新铺白底转 JPEG，避免在第一次有损结果上再次有损编码。
async function readImage(buf: Buffer, filePath: string, signal?: AbortSignal): Promise<ToolOutput> {
  const sharp = await loadSharp();
  signal?.throwIfAborted();
  const image = sharp(buf, { pages: 1, page: 0, limitInputPixels: MAX_IMAGE_PIXELS, failOn: "warning" });
  const metadata = await image.metadata();
  signal?.throwIfAborted();

  const format = metadata.format;
  const canPassThrough =
    (format === "jpeg" || format === "png" || format === "webp") &&
    metadata.width > 0 &&
    metadata.height > 0 &&
    metadata.width <= MAX_IMAGE_EDGE &&
    metadata.height <= MAX_IMAGE_EDGE &&
    base64Length(buf.length) <= MAX_IMAGE_BASE64_CHARS &&
    (metadata.orientation ?? 1) === 1 &&
    (metadata.pages ?? 1) === 1 &&
    metadata.depth === "uchar" &&
    (metadata.space === "srgb" || metadata.space === "b-w") &&
    !metadata.hasProfile &&
    (format !== "png" || isStaticPng(buf));
  if (canPassThrough) {
    // metadata() 只读头部；完整解码一次以拒绝截断/损坏的数据，解码出的像素直接丢弃
    await image.clone().raw().toBuffer();
    signal?.throwIfAborted();
    const mediaType = `image/${format}` as const;
    return [
      {
        type: "text",
        text: `已读取图片 ${filePath}：${metadata.width}x${metadata.height}，未重编码，直接以 ${mediaType} 发送。`,
      },
      { type: "image", source: { type: "base64", media_type: mediaType, data: buf.toString("base64") } },
    ];
  }

  // 基础管线只描述转正与缩放；编码设置只加在 clone() 出的副本上，回退可从同一原图管线重新分支
  const base = image.autoOrient().resize({
    width: MAX_IMAGE_EDGE,
    height: MAX_IMAGE_EDGE,
    fit: "inside",
    withoutEnlargement: true,
  });
  const preferred = base.clone();
  const encoded =
    format === "jpeg"
      ? preferred.jpeg({ quality: 90, chromaSubsampling: "4:4:4" })
      : format === "webp"
        ? preferred.webp({ quality: 90 })
        : preferred.png();
  let result = await encoded.toBuffer({ resolveWithObject: true });
  signal?.throwIfAborted();

  let compressed = false;
  let flattened = false;
  if (base64Length(result.data.length) > MAX_IMAGE_BASE64_CHARS) {
    result = await base
      .clone()
      .flatten({ background: "#ffffff" })
      .jpeg({ quality: 80, chromaSubsampling: "4:2:0" })
      .toBuffer({ resolveWithObject: true });
    signal?.throwIfAborted();
    compressed = true;
    // channels === 4 也可能是 CMYK，透明度以 hasAlpha 为准
    flattened = metadata.hasAlpha === true;
  }
  if (base64Length(result.data.length) > MAX_IMAGE_BASE64_CHARS) {
    throw new Error(
      `图片经缩放压缩后仍超过 ${MAX_IMAGE_BASE64_CHARS} 个 Base64 字符上限：${filePath}。请进一步缩小尺寸或压缩后再读取。`
    );
  }
  const data = result.data.toString("base64");

  const mediaType =
    result.info.format === "jpeg" ? "image/jpeg" : result.info.format === "webp" ? "image/webp" : "image/png";
  const notes = [
    `已读取图片 ${filePath}：原图转正后尺寸 ${metadata.autoOrient.width}x${metadata.autoOrient.height}，` +
      `实际发送尺寸 ${result.info.width}x${result.info.height}（${mediaType}）。`,
  ];
  if ((metadata.pages ?? 1) > 1) notes.push("仅取第一帧。");
  if (compressed) notes.push("为控制在传输体积上限内，已转为 JPEG 压缩。");
  if (flattened) notes.push("透明区域已铺白底合成。");
  return [
    { type: "text", text: notes.join("") },
    { type: "image", source: { type: "base64", media_type: mediaType, data } },
  ];
}

// 原子提交写入：同目录临时文件 → 完整写入并 fsync → 关闭 → rename 覆盖目标。
// - 提交边界是 rename：提交前观察到的失败或取消会清理临时文件，原目标不变；
//   rename 一旦开始即可能完成，此后按成功返回（不会在提交后再报告取消，避免“报失败、实际已改”）。
// - 只提供运行期间的原子替换与提交前失败保护；不承诺系统崩溃或掉电后的结果（未对目录 fsync）。
// - 只保留普通 rwx 权限位；setuid/setgid/sticky、属主/属组、ACL、xattr 不保留。
// - 目标已存在但当前进程不可写（如 0444）时拒绝，保持与直接写入相同的语义——rename 只看目录权限，
//   否则会悄悄覆盖用户设为只读的文件。该检查与 rename 之间仍有时间窗口，仅防误操作。
// - targetPath 应为已解析的真实路径：工作区内的符号链接因此被保留；硬链接会被解绑（目标路径指向新 inode）。
async function atomicWriteFile(targetPath: string, content: string, signal?: AbortSignal): Promise<void> {
  const existing = await fs.stat(targetPath).catch((err: unknown) => {
    if (errnoCode(err) === "ENOENT") return null;
    throw err;
  });
  if (existing && !existing.isFile()) {
    throw new Error(`目标不是普通文件（如目录或特殊文件），无法覆盖：${targetPath}`);
  }
  if (existing) {
    await fs.access(targetPath, fs.constants.W_OK).catch(() => {
      throw new Error(`权限被拒绝，文件不可写：${targetPath}`);
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

// 展示时去掉 UTF-8 BOM（不可见字符，避免模型将其复制进 old_text）；edit 仍会原样保留文件中的 BOM
function formatReadText(decodedText: string, offset?: number, limit?: number): string {
  const text = stripLeadingBom(decodedText);
  if (text.length === 0) {
    return "（空文件）";
  }

  const lines = text.split(/\r?\n/);
  // 文件末尾的换行符不额外产生一行；没有换行符结尾的最后一行仍计入
  if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
  const start = (offset ?? 1) - 1;
  if (start >= lines.length) {
    return `（offset ${offset} 超出文件末尾；总行数：${lines.length}）`;
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
    const shownRange =
      completeLines === 1 ? `第 ${start + 1} 行` : `第 ${start + 1}-${start + completeLines} 行`;
    const hint =
      completeLines > 0
        ? `${shownRange}已完整显示；可指定 offset=${start + completeLines + 1} 继续读取。`
        : `第 ${start + 1} 行单行即超过字符上限，无法用此工具继续读取；请改用 bash（如 \`sed -n '${start + 1}p' <文件路径> | cut -c 1-20000\`）分段读取。`;
    return numbered + `\n... [已在 ${MAX_READ_CHARS} 字符处截断。文件共 ${lines.length} 行。${hint}]`;
  }
  if (end < lines.length) {
    const remaining = lines.length - end;
    const shownRange = start + 1 === end ? `第 ${end} 行` : `第 ${start + 1}-${end} 行`;
    return (
      numbered +
      `\n... [当前显示${shownRange}（共 ${lines.length} 行），剩余 ${remaining} 行。` +
      `可指定 offset=${end + 1} 继续读取。]`
    );
  }
  return numbered;
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

// 换行策略（逐项判断）：
// 1. 纯 CRLF 文件：无论走精确匹配还是 LF→CRLF 回退匹配，new_text 中的换行一律转为 CRLF，保持文件风格；
// 2. LF 文件或混合换行文件：精确匹配时按原样替换（尊重调用方给出的换行）；
//    精确匹配未命中而 CRLF 归一化后命中时，被替换区域本身是 CRLF，new_text 同步转为 CRLF。
// 任何情况下都不会把已有的 "\r\n" 从中间劈开。
// 批量语义：所有 edits 都在同一份原文上定位，每项必须唯一、区域互不重叠（相邻可以）；
// 任一项失败则整体不写入，全部校验通过后一次原子提交。不做引号/空白/Unicode 归一化等模糊匹配。
function applyTextEdits(
  rawContent: string,
  edits: ReadonlyArray<{ old_text: string; new_text: string }>,
  filePath: string
): { updated: string; firstChangedLine: number } {
  const hasCrLf = rawContent.includes("\r\n");
  const hasBareLf = rawContent.replace(/\r\n/g, "").includes("\n");
  const isPureCrLfFile = hasCrLf && !hasBareLf;
  const label = (i: number) => (edits.length === 1 ? "old_text" : `edits[${i}].old_text`);

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
      throw new Error(`在 ${filePath} 中未匹配到指定的 ${label(i)}。请先读取文件确认精确原文。文件未做任何修改。`);
    }
    if (matches.length > 1) {
      throw new Error(
        `指定的 ${label(i)} 在 ${filePath} 中命中多处。请补充更多前后上下文行以唯一定位。文件未做任何修改。`
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
        `edits[${prev.index}] 与 edits[${cur.index}] 在 ${filePath} 中的匹配范围重叠。` +
          `请合并为同一项编辑，或改用互不重叠的 old_text。文件未做任何修改。`
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
    throw new Error(`编辑未对 ${filePath} 产生任何改动：替换后的内容与原文件完全一致。`);
  }

  // 首个实际变化所在的行号：取原文与新内容公共前缀的结束处（匹配文本可能含未变化的上下文，首项也可能是无变化编辑）。
  // 上面已排除 updated === rawContent，因此必有差异位置。
  let firstChangedLine = 1;
  for (let i = 0; i < rawContent.length && i < updated.length && rawContent[i] === updated[i]; i++) {
    if (rawContent.charCodeAt(i) === 10) firstChangedLine++;
  }

  return { updated, firstChangedLine };
}

function formatBashToolResult(result: BashRunResult): AgentToolResult {
  const { exitCode, exitSignal, output, log } = result;
  const details = { exitCode, exitSignal, ...(log ? { log } : {}) };
  if (exitCode !== 0) {
    const statusDesc =
      exitCode !== null
        ? `退出码 ${exitCode}`
        : exitSignal
          ? `信号 ${exitSignal}`
          : "未知信号";
    const emptyNote =
      exitCode !== null
        ? `（命令以退出码 ${exitCode} 退出，无输出）`
        : exitSignal
          ? `（命令因信号 ${exitSignal} 终止，无输出）`
          : "（命令因未知信号终止，无输出）";
    return { content: `命令执行失败（${statusDesc}）：\n${output || emptyNote}`, isError: true, details };
  }
  return { content: output || "（命令正常退出，退出码 0，无输出）", details };
}

function isInsideRoot(rootDir: string, targetPath: string): boolean {
  const rel = path.relative(rootDir, targetPath);
  return rel !== ".." && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel);
}

// 路径检查用于防止误操作（目录穿越、指向工作区外的符号链接），不是安全沙箱：
// 检查与后续按路径打开之间存在 TOCTOU 窗口，硬链接也不受 realpath 约束，且 bash 工具本身可执行任意命令。
// 返回已解析符号链接的真实路径，后续读写均基于该路径，避免二次跟随链接，并让原子写入保留工作区内的符号链接。
async function resolveWorkspacePath(workspaceRoot: string, filePath: string): Promise<string> {
  const resolved = path.resolve(workspaceRoot, filePath);
  if (!isInsideRoot(workspaceRoot, resolved)) {
    throw new Error(`路径超出工作区范围（${workspaceRoot}）：${filePath}`);
  }

  const realRoot = await resolveExistingRealPath(workspaceRoot);
  const realTarget = await resolveExistingRealPath(resolved);
  if (!isInsideRoot(realRoot, realTarget)) {
    throw new Error(`拒绝访问越界符号链接：${filePath} 指向工作区之外（${realTarget}）`);
  }
  return realTarget;
}

// 创建绑定到指定工作区根目录的 read 工具（支持 UTF-8 文本分页读取与图片自动转正缩放，工具本身不做语法分析）
export function createReadTool(workspace: string) {
  const root = path.resolve(workspace);
  return defineTool({
    name: "read",
    description:
      `读取 UTF-8 文本文件（上限 ${formatBytes(MAX_READ_BYTES)}）或 JPEG、PNG、GIF、WebP 图片（上限 ${formatBytes(MAX_IMAGE_FILE_BYTES)}）。` +
      `文本支持起始行号 offset（从 1 开始）与最大读取行数 limit。图片忽略 offset/limit，会自动转正并等比缩放至 ` +
      `${MAX_IMAGE_EDGE}x${MAX_IMAGE_EDGE} 以内（不裁剪、不放大），仅发送第一帧；读取图片需要支持视觉能力的模型。`,
    inputSchema: z.object({
      file_path: z.string().min(1),
      offset: z.number().int().min(1).optional(),
      limit: z.number().int().min(1).max(5000).optional(),
    }),
    run: async ({ file_path, offset, limit }, { signal: sig }) => {
      sig?.throwIfAborted();
      const fullPath = await resolveWorkspacePath(root, file_path);
      const stat = await fs.stat(fullPath);
      if (!stat.isFile()) {
        throw new Error(`目标不是普通文件（如目录或特殊文件），无法读取：${file_path}`);
      }
      // 文本与图片的大小上限不同：读入整个文件前先按文件头判断类型；读入后再按实际内容复核一次
      const assertWithinLimit = (isImage: boolean, size: number) => {
        if (isImage && size > MAX_IMAGE_FILE_BYTES) {
          throw new Error(
            `图片文件大小（${size} 字节）超过 ${formatBytes(MAX_IMAGE_FILE_BYTES)} 上限：${file_path}。请先压缩或缩小图片后再读取。`
          );
        }
        if (!isImage && size > MAX_READ_BYTES) {
          throw new Error(
            `文件大小（${size} 字节）超过 ${formatBytes(MAX_READ_BYTES)} 上限，请改用 bash 工具（如 head/tail/rg）读取。`
          );
        }
      };
      assertWithinLimit(isSupportedImage(await readFileHead(fullPath, 12)), stat.size);

      const buf = await fs.readFile(fullPath, { signal: sig });
      const isImage = isSupportedImage(buf);
      assertWithinLimit(isImage, buf.length);
      if (isImage) return readImage(buf, file_path, sig);

      return formatReadText(decodeStrictUtf8(buf, file_path), offset, limit);
    },
  });
}

// 创建绑定到指定工作区根目录的 write 工具（面向 UTF-8 文本，经临时文件 + rename 原子提交）
export function createWriteTool(workspace: string) {
  const root = path.resolve(workspace);
  return defineTool({
    name: "write",
    description: "创建新文件（自动创建不存在的父目录），或完整覆盖已有的普通文件。",
    inputSchema: z.object({
      file_path: z.string().min(1),
      content: z.string(),
    }),
    run: async ({ file_path, content }, { signal: sig }) => {
      sig?.throwIfAborted();
      const fullPath = await resolveWorkspacePath(root, file_path);
      await fs.mkdir(path.dirname(fullPath), { recursive: true });
      await atomicWriteFile(fullPath, content, sig);
      return `已成功向 ${file_path} 写入 ${Buffer.byteLength(content, "utf8")} 字节`;
    },
  });
}

// 创建绑定到指定工作区根目录的 edit 工具（面向 UTF-8 文本，支持多处互不重叠精确替换与 CRLF 兼容，一次原子提交）
export function createEditTool(workspace: string) {
  const root = path.resolve(workspace);
  return defineTool({
    name: "edit",
    description:
      "精确替换已有 UTF-8 文件中的一处或多处文本，并一次原子写入。" +
      "每个 old_text 均在修改前的原文件上匹配（而非前序编辑的结果），必须在原文件中唯一命中，且各处匹配范围互不重叠；" +
      "任一项失败则整体不写入。若原文件全部采用 CRLF 换行，new_text 中的换行会自动转为 CRLF。",
    inputSchema: z.object({
      file_path: z.string().min(1),
      edits: z
        .array(
          z.object({
            old_text: z.string().min(1, "old_text 不能为空"),
            new_text: z.string(),
          })
        )
        .min(1, "edits 至少需要包含一项编辑"),
    }),
    run: async ({ file_path, edits }, { signal: sig }) => {
      sig?.throwIfAborted();
      const fullPath = await resolveWorkspacePath(root, file_path);
      const stat = await fs.stat(fullPath);
      if (!stat.isFile()) {
        throw new Error(`目标不是普通文件（如目录或特殊文件），无法编辑：${file_path}`);
      }
      if (stat.size > MAX_READ_BYTES) {
        throw new Error(`文件大小（${stat.size} 字节）超过 ${formatBytes(MAX_READ_BYTES)} 上限。`);
      }

      const buf = await fs.readFile(fullPath, { signal: sig });
      const rawContent = decodeStrictUtf8(buf, file_path);
      const { updated, firstChangedLine } = applyTextEdits(rawContent, edits, file_path);

      // rawContent 保留了 BOM（见 utf8FatalDecoder），写回时原样保留
      await atomicWriteFile(fullPath, updated, sig);
      return {
        content:
          edits.length === 1
            ? `已成功编辑 ${file_path}`
            : `已成功对 ${file_path} 应用 ${edits.length} 处编辑`,
        details: { path: file_path, edits: edits.length, firstChangedLine },
      };
    },
  });
}

export interface BashToolOptions {
  env?: NodeJS.ProcessEnv;
  // bash 输出被截断时保存输出日志的目录（相对路径基于 workspace，单个日志上限 64 MiB）：
  // - 省略：首次需要时在系统临时目录下创建本工具实例专用目录（agent-bash-logs-XXXXXX）；该目录默认在工作区外，
  //   仅限工作区的 read 工具会拒绝读取，模型需用 bash 检索；若希望 read 可读，宿主可将其设为工作区内的目录。
  // - null：不保存日志，仅返回头尾摘要。
  // 日志不会自动删除，清理由宿主负责。
  logDir?: string | null;
}

// 创建以指定工作区为执行目录（cwd）的受控进程组 bash 工具（仅支持 POSIX）
export function createBashTool(workspace: string, options?: BashToolOptions) {
  const root = path.resolve(workspace);
  const logDirOption = options?.logDir;
  let defaultLogDir: string | undefined;
  const resolveBashLogDir: (() => string) | undefined =
    logDirOption === null
      ? undefined
      : logDirOption !== undefined
        ? () => path.resolve(root, logDirOption)
        : () => (defaultLogDir ??= mkdtempSync(path.join(os.tmpdir(), "agent-bash-logs-")));

  return defineTool({
    name: "bash",
    description:
      "在工作区中执行前台 bash 命令；命令退出时，同一进程组内的后台子进程会被一并终止。" +
      `输出过长时仅保留开头 ${formatBytes(BASH_HEAD_BYTES)} 与末尾 ${formatBytes(BASH_TAIL_BYTES)}` +
      (resolveBashLogDir
        ? `，并尽力将输出另存为日志文件（上限 ${formatBytes(MAX_BASH_LOG_BYTES)}）；若保存成功，结果中会注明日志路径及是否完整。`
        : "。"),
    inputSchema: z.object({
      command: z.string().min(1),
      timeout_ms: z.number().int().min(100).max(300_000).optional(),
    }),
    // 非零退出不抛错：返回带 isError 的结构化结果，content 供模型阅读，宿主另从 details 取得退出码、信号与日志信息。
    // 执行期间按 BASH_PROGRESS_INTERVAL_MS 节流，把新增输出作为 tool_progress 的 text 报告（增量，过多时只保留末尾）。
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
      return formatBashToolResult(result);
    },
  });
}
