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

/**
 * Agent 单次运行结束的原因：
 * - `BetaStopReason`：模型原生的停止原因（如 `"end_turn"`、`"tool_use"`、`"max_tokens"`、`"stop_sequence"`、`"refusal"` 等）；
 * - `"max_turns"`：达到 `maxTurns` 最大轮次上限仍未自然结束；
 * - `"aborted"`：运行在开始前或执行期间被 `AbortSignal` 取消；
 * - `null`：尚未取得停止原因时的初始/兜底值。
 */
export type AgentStopReason =
  | Anthropic.Beta.Messages.BetaStopReason
  | "max_turns"
  | "aborted"
  | null;

/**
 * 工具返回给模型的原始内容类型（对应 Anthropic `tool_result` 内容块的 `content` 字段，支持纯文本或文本/图片内容块数组）。
 */
export type ToolOutput = Anthropic.Beta.Messages.BetaToolResultBlockParam["content"];

/**
 * 工具可选的结构化返回对象：
 * - `content` 进入模型历史（模型后续推理需要的信息必须写在这里）；
 * - `details` 只随 `tool_end` 事件交给宿主（不进入历史，也不保证可靠持久化）；
 * - `isError` 允许不抛错地返回带结构信息的失败结果（记为 `is_error` 的工具结果）。
 */
export interface AgentToolResult {
  /** 写入会话历史并回传给模型的工具输出内容。 */
  content: ToolOutput;
  /** 仅随 `tool_end` 观察事件透传给宿主的附加结构化信息，不发送给模型。 */
  details?: unknown;
  /** 是否将该结果标记为工具执行失败（对应 API 的 `is_error: true`）。 */
  isError?: boolean;
}

/**
 * 工具执行期间向宿主上报的增量进度信息。
 */
export interface AgentToolProgress {
  /** 可选的增量进度文本（如命令实时输出片段或阶段说明）。 */
  text?: string;
  /** 可选的结构化进度详情（按引用随 `tool_progress` 事件透传给宿主）。 */
  details?: unknown;
}

/**
 * 单次工具调用在本地执行时收到的运行上下文。
 */
export interface AgentToolContext {
  /**
   * 当前运行的取消信号。
   * 工具因取消而失败时，应抛出 `signal.reason`、`name` 为 `"AbortError"` 的错误，或以 `signal.reason` 为 `cause` 的错误，
   * 这类错误记为“执行被中断”；其他错误即使发生在取消之后，也按工具自身的确定失败原样记录。
   */
  signal?: AbortSignal;
  /** 本次工具调用对应的模型 `tool_use` 块唯一 ID。 */
  toolUseId: string;
  /**
   * 执行期间报告进度，转为 `tool_progress` 观察事件（工具名与 `id` 由执行器补齐）。
   * 仅在宿主订阅了 `onEvent` 时提供（工具应以 `reportProgress?.()` 调用，未提供时可跳过进度相关计算）；
   * 不等待宿主处理、异常被隔离、工具返回后的调用被忽略；高频进度由工具自行节流。`details` 按引用传递。
   */
  reportProgress?: (progress: AgentToolProgress) => void;
}

/**
 * 可挂载到 Agent 中的工具定义契约。
 *
 * @template TInput 经 `parse` 校验转换后的强类型工具入参。
 */
export interface AgentTool<TInput = any> {
  /** 工具唯一名称（模型通过该名称发起 `tool_use` 调用）。 */
  name: string;
  /** 提供给模型的工具功能说明、使用时机与边界约束描述。 */
  description: string;
  /** 提供给 Anthropic API 的工具参数 JSON Schema。 */
  input_schema: Anthropic.Beta.Messages.BetaTool["input_schema"];
  /**
   * 可选的显式缓存断点；任一工具设置后即视为调用方接管整体缓存配置，
   * `runAgent` 不再为 `system`/`tools`/`messages` 自动添加默认断点。
   */
  cache_control?: Anthropic.Beta.Messages.BetaTool["cache_control"];
  /** 对模型生成的原始未知参数做运行时校验与解析，不合法时应抛出错误。 */
  parse: (input: unknown) => TInput;
  /** 执行工具逻辑，支持同步或异步返回原始输出（`ToolOutput`）或结构化结果（`AgentToolResult`）。 */
  run: (
    input: TInput,
    context: AgentToolContext
  ) => Promise<ToolOutput | AgentToolResult> | ToolOutput | AgentToolResult;
}

/**
 * Agent 运行期间发出的只读观察事件，分为两个阶段：
 * 1. 模型流式生成阶段（`thinking_delta` / `text_delta` / `tool_call_start` / `tool_input_delta`）：随模型输出实时发出。
 *    `tool_call_start` 在模型开始输出某个 `tool_use` 块时触发，`tool_input_delta` 随参数原始 JSON 片段（非空 `partial_json`）到达触发；
 *    若流中途取消/报错、因 `max_tokens` 等原因截断、工具名未知或同批前序工具执行后已取消，已发出流式事件的调用不会进入本地执行阶段。
 * 2. 本地工具执行阶段（`tool_start` / `tool_progress` / `tool_end`）：在完整响应到达后按串行执行顺序发出；每次 `tool_start` 必有配对的 `tool_end`。
 */
export type AgentEvent =
  /** 模型扩展思考（thinking）的流式增量文本片段。 */
  | { type: "thinking_delta"; delta: string }
  /** 模型回复正文的流式增量文本片段。 */
  | { type: "text_delta"; delta: string }
  /** 模型开始流式生成某个 `tool_use` 工具调用块。 */
  | { type: "tool_call_start"; id: string; name: string }
  /** 模型流式生成工具参数 JSON 的原始增量片段（`partial_json`）。 */
  | { type: "tool_input_delta"; id: string; name: string; delta: string }
  /** 本地开始执行某个已注册工具（`input` 为深拷贝快照）。 */
  | { type: "tool_start"; id: string; name: string; input: unknown }
  /** 工具在本地执行期间通过 `reportProgress` 上报的增量进度。 */
  | { type: "tool_progress"; id: string; name: string; text?: string; details?: unknown }
  /** 本地工具执行结束（无论成功、报错或启动前跳过，必与 `tool_start` 成对出现）。 */
  | { type: "tool_end"; id: string; name: string; isError: boolean; outputSummary: string; details?: unknown };

/**
 * 本次运行累计的模型 Token 用量统计。
 * 注意：仅累计成功取得完整响应（`finalMessage`）的轮次；流中途出错或取消的轮次不计入，不能直接当作计费账单。
 */
export interface AgentUsage {
  /** 累计常规输入 Token 数（不含缓存读写部分）。 */
  inputTokens: number;
  /** 累计模型输出 Token 数（含思考与正文/工具调用）。 */
  outputTokens: number;
  /** 累计从提示缓存命中的输入 Token 数。 */
  cacheReadTokens: number;
  /** 累计新建写入提示缓存的输入 Token 数。 */
  cacheCreationTokens: number;
}

/**
 * `prepareMessages` 钩子在每轮发起模型请求前收到的上下文信息。
 */
export interface PrepareMessagesContext {
  /** 完整历史的浅拷贝（含本次运行已新增的消息）。请勿就地修改其中的消息对象——它们与完整历史共享。 */
  messages: readonly Anthropic.Beta.Messages.BetaMessageParam[];
  /** 本次运行内即将发出的第几次模型请求（从 1 开始计数）。 */
  turn: number;
  /** 当前运行的取消信号。 */
  signal?: AbortSignal;
}

/**
 * `runAgent` 无状态运行的入参选项。
 */
export interface AgentOptions {
  /** 由宿主创建并注入的 Anthropic SDK 客户端实例（含认证与重试策略配置）。 */
  client: Anthropic;
  /** 目标模型标识（如 `"claude-sonnet-4-5"` 等）。 */
  model: string;
  /** 调用方持有的完整历史（不会被就地修改）。允许连续同角色消息，按 API 语义视为同一回合。 */
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  /**
   * 本次运行可用的工具列表（默认无工具，由宿主按需组装）。
   * 自定义工具可用 `defineTool` 定义；内置本地工具见 `createReadTool` / `createWriteTool` / `createEditTool` / `createBashTool`。
   */
  tools?: AgentTool[];
  /** 系统提示词；默认使用不含环境信息的通用提示词，工作目录等上下文由宿主按需写入。 */
  systemPrompt?: string;
  /** 单次模型请求允许生成的最大 Token 数（默认 `DEFAULT_MAX_TOKENS` = 98304）。 */
  maxTokens?: number;
  /** 本次运行允许发起的最大模型请求轮次（正整数，默认 `DEFAULT_MAX_TURNS` = 200）。 */
  maxTurns?: number;
  /** 可选的模型扩展思考（Extended Thinking）配置。 */
  thinking?: Anthropic.Beta.Messages.BetaThinkingConfigParam;
  /**
   * 是否自动在 `system`、末尾工具与末尾消息上添加默认 `ephemeral` 缓存断点（默认 `true`）。
   * 每次请求按最终发送的消息与 `tools` 判断：任一处已有显式 `cache_control` 时即整体跳过全部默认断点（避免超出 API 断点上限）；
   * 设为 `false` 仅关闭自动添加，已有配置仍原样发送。
   */
  autoCacheBreakpoints?: boolean;
  /** 用于取消本次运行的 `AbortSignal`（终止流式请求、跳过未启动工具并终止受控子进程组）。 */
  signal?: AbortSignal;
  /**
   * 本次运行每新增一条完整消息时按序调用（请勿就地修改消息对象）；若抛错将中止后续动作并通过 `AgentRunError` 带出快照。
   * 该 Promise 会被完整等待，不会因 `signal` 取消而被放弃——宿主需自行保证保存操作有合理期限。
   * 这是“完整消息”持久化点而非工具事务日志：同一批次中已完成工具的结果，要等整批结束后才作为一条消息落盘。
   */
  onMessage?: (message: Readonly<Anthropic.Beta.Messages.BetaMessageParam>) => void | Promise<void>;
  /**
   * 观察侧事件回调：同步异常与异步拒绝均被隔离，不阻塞主流程返回
   * （`runAgent` 返回时异步回调可能尚未完成，需要可靠投递的宿主应自行排队并等待处理完成）。
   */
  onEvent?: (event: AgentEvent) => void | Promise<void>;
  /**
   * 每次模型请求（含第一次）之前调用，返回本次实际发送的消息；只影响本次请求，完整历史保持不变。
   * 用于裁剪/摘要较早历史、注入临时检索材料、替换旧图片等。约束（违反时以 `AgentRunError` 中止）：
   * - 进行中的工具回合（当历史中最后一个 `assistant` 回合含 `tool_use` 时，从该回合首条消息到末尾的全部消息）必须原样位于返回结果末尾——
   *   开启 `thinking` 时该回合的 `thinking` 签名必须原样回传；临时材料可插在它之前。最后一个 `assistant` 回合不含 `tool_use` 时无此限制。
   * - 每个 `tool_result` 都必须对应紧邻其前的 `assistant` 回合中的 `tool_use`（裁剪时成对删除）。
   * 这里只校验本地消息结构与进行中的工具回合；模型相关的 `thinking` 前缀绑定约束仍由宿主满足：
   * 部分模型的 `thinking` 还绑定生成它时的 `system`、`tools` 与之前的消息，保留 `thinking` 块原样并不保证可以改写它之前的上下文。
   * 面向这类模型时，宿主应在工具回合进行期间保持已有请求前缀稳定（例如不要逐轮替换插在前面的检索材料），
   * 或采用该模型支持的上下文压缩方式。
   * 返回后仍会经过缺失结果补全与自动缓存断点处理。核心不保存任何状态：调用模型做摘要等昂贵操作应由宿主按历史前缀缓存；
   * 每轮都改写较早前缀会让提示缓存失效，宜“阶梯式”压缩（前缀保持稳定，偶尔整体重写）。
   * 取消时记为 `"aborted"`；其他异常中止运行并通过 `AgentRunError` 带出状态。
   */
  prepareMessages?: (
    context: PrepareMessagesContext
  ) => Promise<Anthropic.Beta.Messages.BetaMessageParam[]> | Anthropic.Beta.Messages.BetaMessageParam[];
  /**
   * 每轮整批工具结果落盘后调用的增量上下文钩子（批次之后的刷新点，而非每个工具调用之前的检查点）；
   * 返回非空字符串时作为 `user` 消息写入完整历史（返回 `null` 或空字符串则忽略）。非取消异常将中止运行以防遗漏关键指令。
   * 覆盖范围：运行开始时与模型给出最终回复（无工具调用）之后都不会调用；同一批次中，后面的工具在钩子之前就已执行。
   * 因此它不能替代“副作用发生前检查最新状态”——例如发送消息前检查是否有新指令，应放在该工具内部、紧邻实际副作用处。
   */
  getContextInjection?: (context: { signal?: AbortSignal }) => Promise<string | null>;
}

/**
 * `runAgent` 运行完成（含自然结束、达到轮次上限或被取消）时返回的结果快照。
 */
export interface AgentResult {
  /**
   * 调用方历史 + 本次运行新增的消息（含末尾恢复消息）。
   * 注意：历史中段缺失工具结果的补全只存在于每次请求的副本中，不会写回这里，也不会经过 `onMessage`。
   * 只有完整取得的 `assistant` 响应才会写入：流中途出错或取消时，已通过流式事件（`thinking_delta`/`text_delta`/`tool_call_start`/`tool_input_delta`）
   * 发出的部分输出不会出现在这里；需要保留“用户已看到的内容”的宿主，应在展示记录中自行保存，而不是把半成品当作模型上下文。
   * 请求级重试由注入的 Anthropic 客户端配置（如 `maxRetries`）负责，`runAgent` 不做整轮重试。
   */
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  /** 本次运行最后一条完整 `assistant` 响应中的文本（多段以 `"\n"` 拼接；该响应无文本块或本次未取得响应时为 `""`，不回溯传入的初始历史）。 */
  lastText: string;
  /** 本次运行的停止原因。 */
  stopReason: AgentStopReason;
  /** 本次运行实际发起的模型请求次数（发出请求即计入，含流中途取消的轮次）。 */
  turns: number;
  /** 本次运行已完成响应轮次的累计 Token 用量。 */
  usage: AgentUsage;
}

/**
 * Agent 运行期间发生未预期错误（如 API 调用失败、宿主钩子抛错、消息配对校验失败等）时抛出的异常类。
 * 随附错误发生时刻已确认的会话历史快照、已发起的模型请求轮次与累计 Token 用量，供宿主恢复状态或记录审计日志。
 */
export class AgentRunError extends Error {
  /** 异常发生前已确认写入的完整会话消息快照（浅拷贝）。 */
  readonly messages: Anthropic.Beta.Messages.BetaMessageParam[];
  /** 异常发生前已实际发出的模型请求次数。 */
  readonly turns: number;
  /** 异常发生前已完成轮次的累计 Token 用量快照。 */
  readonly usage: AgentUsage;

  /**
   * 构造携带运行状态快照的 `AgentRunError`。
   *
   * @param message 错误描述文本。
   * @param state 异常发生时刻的会话消息、请求轮次与 Token 用量状态。
   * @param options 标准 `Error` 选项（如原始异常 `cause`）。
   */
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

/**
 * 使用 Zod 对象 Schema 定义强类型 `AgentTool` 的辅助工厂函数：
 * 自动将 `inputSchema` 转换为 Anthropic API 所需的 JSON Schema，并用 `inputSchema.parse` 完成运行时入参校验。
 *
 * @template S 工具入参对应的 `z.ZodObject` Schema 类型。
 * @param spec 工具名称、说明、Zod 入参 Schema、可选缓存控制与执行函数。
 * @returns 可直接传入 `AgentOptions.tools` 的 `AgentTool` 实例。
 */
export function defineTool<S extends z.ZodObject>(spec: {
  /** 工具唯一名称。 */
  name: string;
  /** 提供给模型的工具功能与参数说明。 */
  description: string;
  /** 用于生成 JSON Schema 及运行时参数校验的 Zod 对象 Schema。 */
  inputSchema: S;
  /** 可选的显式提示缓存断点配置。 */
  cache_control?: AgentTool["cache_control"];
  /** 接收经 Zod 校验后的强类型参数并执行工具逻辑。 */
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

/** 默认的最大模型请求轮次上限，防止工具调用陷入无限循环。 */
const DEFAULT_MAX_TURNS = 200;

/** 单次模型响应的默认最大输出 Token 数（96 Ki）。 */
const DEFAULT_MAX_TOKENS = 98_304;

/** `tool_end` 观察事件中 `outputSummary` 字段的最大字符数上限。 */
const TOOL_SUMMARY_MAX_CHARS = 500;

/** 历史中存在未配对 `tool_use`（如上次运行中断）时，自动补造的错误态 `tool_result` 说明文案。 */
const RECOVERY_RESULT_TEXT = "从上次未完成的运行中恢复：该工具调用的执行状态未知。";

/**
 * 按最大 UTF-16 码元长度安全截断字符串，避免在截断处切开代理对（如 Emoji 或生僻汉字）留下孤立高位代理项。
 *
 * @param str 待截断的原始字符串。
 * @param maxChars 允许保留的最大 UTF-16 码元数。
 * @returns 截断后的合法 UTF-16 字符串。
 */
function sliceStringSafe(str: string, maxChars: number): string {
  if (str.length <= maxChars) return str;
  let end = maxChars;
  const code = str.charCodeAt(end - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    end--;
  }
  return str.slice(0, end);
}

/**
 * API 语义下的单个对话“回合”：相邻的同角色消息会被服务端合并为一个回合。
 * `system` 消息不属于任何回合，也不打断前后相邻同角色消息的回合合并。
 */
interface ConversationTurn {
  /** 该回合的消息角色（如 `"user"` 或 `"assistant"`）。 */
  role: Anthropic.Beta.Messages.BetaMessageParam["role"];
  /** 该回合包含的消息在原始消息数组中的下标列表（按出现顺序排列）。 */
  indices: number[];
}

/**
 * 将消息数组按 Anthropic API 语义归并为交替的对话回合列表（忽略 `system` 消息且不因其打断回合）。
 *
 * @param messages 待分组的消息列表。
 * @returns 按时间顺序排列的对话回合数组。
 */
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

/**
 * 按出现顺序收集一个回合内全部 `tool_use` 内容块的唯一 ID（自动去重）。
 *
 * @param messages 完整消息列表。
 * @param indices 当前 `assistant` 回合所包含的消息下标数组。
 * @returns 去重后保持首次出现顺序的 `tool_use` ID 列表。
 */
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

/**
 * 收集一个回合内全部 `tool_result` 内容块所关联的 `tool_use_id` 集合。
 *
 * @param messages 完整消息列表。
 * @param indices 当前 `user` 回合所包含的消息下标数组。
 * @returns 已覆盖的 `tool_use_id` 集合。
 */
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

/**
 * 为一组缺失结果的 `tool_use` ID 批量生成“执行状态未知”的错误态恢复结果块。
 *
 * @param ids 缺失 `tool_result` 的 `tool_use` ID 列表。
 * @returns 标记为 `is_error: true` 的 `tool_result` 内容块数组。
 */
function makeRecoveryResults(ids: string[]): Anthropic.Beta.Messages.BetaToolResultBlockParam[] {
  return ids.map((id) => ({
    type: "tool_result" as const,
    tool_use_id: id,
    is_error: true,
    content: RECOVERY_RESULT_TEXT,
  }));
}

/**
 * 末尾恢复：若历史以 `assistant` 回合结束（可能由多条连续 `assistant` 消息组成）且其中含有 `tool_use`，
 * 为该回合内全部 `tool_use` 补造一条“状态未知”的 `user` 结果消息；否则返回 `null`。
 * 该消息在 `runAgent` 启动时通过 `appendMessage` 写入历史并持久化。
 *
 * @param messages 当前完整历史消息数组。
 * @returns 需要追加到末尾的恢复消息，无需恢复时返回 `null`。
 */
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

/**
 * 检测消息列表中是否有任意内容块已显式配置了 `cache_control` 缓存断点。
 *
 * @param messages 待检查的消息列表。
 * @returns 若任一内容块带有非空 `cache_control` 则返回 `true`，否则返回 `false`。
 */
function messagesHaveCacheControl(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]): boolean {
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if ("cache_control" in block && block.cache_control) return true;
    }
  }
  return false;
}

/**
 * 深度比较两个值在结构语义上是否相等（忽略值为 `undefined` 的属性与对象属性顺序），
 * 用于校验 `prepareMessages` 是否原样保留了进行中的工具回合。
 *
 * @param a 待比较的第一个值。
 * @param b 待比较的第二个值。
 * @returns 结构语义一致时返回 `true`，否则返回 `false`。
 */
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

/**
 * 提取历史中“进行中的工具回合”消息切片：当最后一个 `assistant` 回合含 `tool_use` 时，
 * 返回从该回合首条消息到历史末尾的全部消息；否则返回空数组。
 *
 * @param messages 完整会话历史。
 * @returns 必须在 `prepareMessages` 返回结果末尾原样保留的消息切片。
 */
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

/**
 * 校验 `prepareMessages` 的返回值是否合法（必须为非空数组，且末尾原样保留进行中的工具回合）；
 * 校验失败时抛错，由主循环包装为 `AgentRunError`。
 *
 * @param prepared `prepareMessages` 返回的候选消息数组。
 * @param history 当前完整会话历史。
 */
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

/**
 * 断言每个 `tool_result` 都对应紧邻其前的 `assistant` 回合中的 `tool_use`；否则 API 会拒绝请求。
 * 这里提前抛出指明具体 `tool_use_id` 的错误，而不是悄悄删除（裁剪策略属于宿主）。缺失结果的补全由 `buildApiRequestMessages` 负责。
 *
 * @param messages 本轮准备发往 API 的消息列表。
 */
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

/**
 * 构造发往 API 的归一化请求消息副本（不修改、不持久化调用方历史）：
 * - 按完整回合检查：`assistant` 回合内所有 `tool_use`，只要在紧随其后的整个 `user` 回合中任一条消息里有结果即视为已覆盖；
 * - 仅对确实缺失结果的调用补造“状态未知”结果：若存在后续 `user` 回合，则前插到该回合首条消息的开头
 *   （保证 `tool_result` 位于普通文本之前）；否则在该 `assistant` 回合之后插入一条新的 `user` 消息；
 * - 不合并、不拆分调用方的消息，不改动任何内容块（`thinking` 及其签名原样透传），`system` 消息原样保留；
 * - 当 `addCacheBreakpoint` 为 `true` 时，在最后一条非临时消息的可缓存末尾块上附加 `ephemeral` 缓存断点。
 *
 * @param messages 经校验的本轮请求消息列表。
 * @param addCacheBreakpoint 是否在末尾消息的最后一个可缓存内容块上自动添加默认缓存断点。
 * @returns 可直接发送给 Anthropic API 的深/浅隔离消息副本。
 */
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

/**
 * 让异步 Promise 与 `AbortSignal` 竞速：若信号已取消或在等待期间触发取消，立即以 `signal.reason` 拒绝，
 * 同时屏蔽原 Promise 后续可能出现的未捕获拒绝，并在结算后清理事件监听。
 *
 * @template T 原 Promise 的解析值类型。
 * @param promise 待竞速的目标异步任务。
 * @param signal 可选的取消信号。
 * @returns 随目标任务完成或随取消信号立即拒绝的 Promise。
 */
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

/**
 * 按抛出的错误本身判断是否由取消信号导致，而不是仅看 `catch` 时 `signal` 是否已取消：
 * 工具已得到确定结果（如命令以非零码退出或参数校验失败）后才到达的取消，不能把真实失败改报为中断。
 *
 * @param err 捕获到的未知异常值。
 * @param signal 当前运行的取消信号。
 * @returns 确认异常由该取消信号引起时返回 `true`，否则返回 `false`。
 */
function isAbortCausedError(err: unknown, signal?: AbortSignal): boolean {
  if (!signal?.aborted) return false;
  // reason 可以是任意值（含 null、NaN），用 Object.is 比较
  if (Object.is(err, signal.reason)) return true;
  const e = err as { name?: unknown; cause?: unknown } | null;
  return e?.name === "AbortError" || (e?.cause !== undefined && Object.is(e.cause, signal.reason));
}

/**
 * 类型守卫：判断工具返回值是否为结构化的 `AgentToolResult`。
 * `ToolOutput` 只可能是字符串、数组或 `undefined`，因此“带 `content` 属性的普通非空对象”可无歧义地识别为结构化结果。
 *
 * @param value 工具 `run` 函数返回的值。
 * @returns 为 `AgentToolResult` 结构时返回 `true`。
 */
function isAgentToolResult(value: unknown): value is AgentToolResult {
  return typeof value === "object" && value !== null && !Array.isArray(value) && "content" in value;
}

/**
 * 将工具输出生成适合放入 `tool_end.outputSummary` 观察事件的简短文本摘要（最长 `TOOL_SUMMARY_MAX_CHARS` 字符）。
 * 数组型输出中的图片块会被替换为媒体类型说明，避免将庞大的 Base64 图像数据塞入事件流。
 *
 * @param output 工具返回给模型的 `ToolOutput` 内容。
 * @returns 安全截断后的摘要字符串。
 */
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

/**
 * 惰性观察事件发射器类型：接收一个按需构造 `AgentEvent` 的工厂函数，
 * 仅在存在 `onEvent` 订阅者时才求值并派发，避免无订阅者时产生深拷贝或摘要序列化开销。
 */
type EmitEvent = (build: () => AgentEvent) => void;

/** Anthropic Beta Messages 流式请求入参类型别名。 */
type BetaStreamParams = Parameters<Anthropic["beta"]["messages"]["stream"]>[0];

/**
 * 单轮工具调用批次串行执行所需的共享运行上下文。
 */
interface ToolBatchExecutionContext {
  /** 工具名称到 `AgentTool` 实例的只读查找映射。 */
  toolMap: ReadonlyMap<string, AgentTool>;
  /** 逗号分隔的可用工具名称列表文本（用于未知工具报错提示）。 */
  toolNamesList: string;
  /** 当前运行的取消信号。 */
  signal?: AbortSignal;
  /** 宿主是否传入了 `onEvent` 订阅回调（决定是否向工具注入 `reportProgress`）。 */
  hasEventSubscriber: boolean;
  /** 惰性观察事件发射器。 */
  emit: EmitEvent;
}

/**
 * 为一批未进入实际执行的 `tool_use` 块批量构造指定原因的错误态 `tool_result` 块，
 * 以确保会话历史始终满足 Anthropic API 的 `tool_use` / `tool_result` 配对要求。
 *
 * @param blocks 被跳过的模型 `tool_use` 内容块列表。
 * @param reason 向模型说明该调用未执行的具体原因文案。
 * @returns 标记为 `is_error: true` 的 `tool_result` 内容块数组。
 */
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

/**
 * 发起单轮模型流式请求，并在存在订阅者时桥接流式增量观察事件（思考、正文、工具调用起点与参数 JSON 增量），
 * 等待并返回聚合完成的最终模型响应消息。
 *
 * @param client Anthropic SDK 客户端实例。
 * @param params 流式请求参数（含模型、系统提示词、工具定义与归一化消息）。
 * @param signal 可选的取消信号。
 * @param subscribeEvents 是否监听底层流事件并转换为 `AgentEvent`。
 * @param emit 惰性观察事件发射器。
 * @returns 完整接收后的 `BetaMessage` 响应对象。
 */
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

/**
 * 在本地执行单个 `tool_use` 块：
 * 依次完成取消预检、工具查找、参数深拷贝隔离、`parse` 校验、`run` 执行与异常兜底，
 * 并严格保证一旦发出 `tool_start` 观察事件，必有配对的 `tool_end` 事件发出。
 *
 * @param block 模型输出的单个 `tool_use` 调用块。
 * @param ctx 工具批次共享执行上下文。
 * @returns 可写入 `user` 消息回传给模型的 `tool_result` 内容块。
 */
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
    /** 向宿主派发与当前工具调用绑定的 `tool_progress` 增量事件。 */
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

/**
 * 按模型响应中的出现顺序严格串行执行整批 `tool_use` 工具调用块，并返回对应的结果块列表。
 *
 * @param blocks 当前 `assistant` 响应中待执行的全部 `tool_use` 块。
 * @param ctx 工具批次共享执行上下文。
 * @returns 与 `blocks` 一一对应的 `tool_result` 内容块数组。
 */
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

/**
 * 执行无状态的“模型推理 ↔ 工具调用”核心主循环：
 * 1. 校验运行参数、检查入口取消状态，并为上次中断遗留的末尾未配对 `tool_use` 补齐恢复结果；
 * 2. 每轮依次执行：`prepareMessages` 上下文整理 → 自动缓存断点装配 → 流式请求模型 → 累计用量并持久化 `assistant` 消息；
 * 3. 若响应不含 `tool_use` 则结束；若含 `tool_use` 则按停止原因处理截断或串行执行工具批次，持久化 `tool_result`，并按需调用 `getContextInjection` 注入增量指令；
 * 4. 正常结束、取消或达到 `maxTurns` 时返回 `AgentResult`；其他未预期异常统一包装为携带最新状态快照的 `AgentRunError` 抛出。
 *
 * @param options 本次 Agent 运行的完整配置项（客户端、模型、消息历史、工具集与生命周期钩子）。
 * @returns 包含最新完整历史、末条文本、停止原因、请求轮次与累计 Token 用量的 `AgentResult`。
 * @throws {Error} 当 `maxTurns` 参数配置本身不合法时直接抛出。
 * @throws {AgentRunError} 当运行期间模型请求、消息校验或宿主钩子发生非取消异常时抛出（附带状态快照）。
 */
export async function runAgent(options: AgentOptions): Promise<AgentResult> {
  const {
    client,
    model,
    signal,
    maxTokens = DEFAULT_MAX_TOKENS,
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

  const messages: Anthropic.Beta.Messages.BetaMessageParam[] = [...options.messages];
  const tools = options.tools ?? [];
  const toolMap = new Map<string, AgentTool>(tools.map((t) => [t.name, t]));
  const toolNamesList = Array.from(toolMap.keys()).join(", ");

  // 自动缓存断点按每次请求的最终消息判断：调用方在 tools 或本次发送的消息中显式设置了 cache_control 时沿用其配置
  const autoCacheBreakpoints = options.autoCacheBreakpoints ?? true;
  const toolsHaveCacheControl = tools.some((t) => t.cache_control);

  /** 根据本轮是否启用默认缓存断点，构造发往 API 的工具声明数组（默认断点标记在末尾工具上）。 */
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

  /** 观察事件发射器：仅在有订阅者时才构造事件（含复制输入、生成摘要）；构造或回调中的同步异常与异步拒绝均被隔离。 */
  const emit: EmitEvent = (build) => {
    if (!onEvent) return;
    try {
      Promise.resolve(onEvent(build())).catch(() => {});
    } catch {}
  };

  /** 将一条新增的完整消息写入本次运行历史，并完整等待宿主的 `onMessage` 持久化回调。 */
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

/** Bash 命令输出摘要保留的开头最大字节数（16 KiB）。 */
const BASH_HEAD_BYTES = 16 * 1024;

/** Bash 命令输出摘要保留的末尾最大字节数（32 KiB）。 */
const BASH_TAIL_BYTES = 32 * 1024;

/** Bash 输出超出头尾摘要时尽力保存输出日志；单个日志文件的磁盘上限（64 MiB），防止持续刷屏的命令写满磁盘。 */
const MAX_BASH_LOG_BYTES = 64 * 1024 * 1024;

/** 终止受控进程组时，发送 `SIGTERM` 后等待优雅退出的宽限时间（毫秒）；超时后升级为 `SIGKILL` 强杀。 */
const KILL_GRACE_MS = 2_000;

/** Shell 主进程退出后，等待管道残余输出排空（I/O drain）的最大宽限时间（毫秒）。 */
const IO_DRAIN_GRACE_MS = 200;

/** 终止流程中轮询进程组存活状态的间隔，以及发送 `SIGKILL` 后等待收尾的延迟时间（毫秒）。 */
const GROUP_POLL_MS = 40;

/** Bash 执行期间向宿主发送 `tool_progress` 增量输出事件的节流时间间隔（毫秒）。 */
const BASH_PROGRESS_INTERVAL_MS = 250;

/** 单次节流周期内累积的进度增量文本最大字符上限（超出时只保留末尾并注明省略字符数）。 */
const BASH_PROGRESS_MAX_CHARS = 8_192;

/**
 * 安全提取 Node.js 系统异常对象上的错误码（如 `"ENOENT"`、`"EPERM"`）。
 *
 * @param err 捕获到的未知异常值。
 * @returns 系统错误码字符串；若不存在则返回 `undefined`。
 */
function errnoCode(err: unknown): string | undefined {
  return (err as NodeJS.ErrnoException | null)?.code;
}

/**
 * 将整 KiB 或整 MiB 的字节限额常量格式化为人类可读的单位文案（如 `"16 KiB"`、`"64 MiB"`）。
 *
 * @param bytes 字节数（仅用于整 KiB / MiB 的限额常量）。
 * @returns 带 `KiB` 或 `MiB` 单位的字符串。
 */
function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${bytes / (1024 * 1024)} MiB` : `${bytes / 1024} KiB`;
}

/**
 * 判断单个字节是否为 UTF-8 多字节序列的后续字节（即二进制高两位为 `10`，范围 `0x80`–`0xBF`）。
 *
 * @param byte 待检查的无符号字节值（0–255）。
 * @returns 为 UTF-8 后续字节时返回 `true`。
 */
function isUtf8ContinuationByte(byte: number): boolean {
  return (byte & 0xc0) === 0x80;
}

/**
 * 在不超过 `maxLen` 的前提下，向下回退对齐到合法的 UTF-8 字符边界（避免在前缀截断处切开多字节字符）。
 *
 * @param buf UTF-8 编码的字节缓冲区。
 * @param maxLen 允许保留的最大前缀字节长度。
 * @returns 对齐后的安全前缀结束字节下标。
 */
function alignUtf8PrefixEnd(buf: Buffer, maxLen: number): number {
  let end = Math.min(buf.length, maxLen);
  while (end > 0 && isUtf8ContinuationByte(buf[end])) {
    end--;
  }
  return end;
}

/**
 * 从 `startIdx` 起向上推进对齐到合法的 UTF-8 字符起始边界（避免在后缀截断起点处残留半个多字节字符）。
 *
 * @param buf UTF-8 编码的字节缓冲区。
 * @param startIdx 期望的后缀起始字节下标。
 * @returns 对齐后的安全后缀起始字节下标。
 */
function alignUtf8SuffixStart(buf: Buffer, startIdx: number): number {
  let start = Math.max(0, startIdx);
  while (start < buf.length && isUtf8ContinuationByte(buf[start])) {
    start++;
  }
  return start;
}

/**
 * 创建带时间节流与字符容量保护的增量文本进度上报器：
 * - `push` 累积新到达的文本，每个 `BASH_PROGRESS_INTERVAL_MS` 间隔最多触发一次 `report`；
 *   单窗口内累积超过 `BASH_PROGRESS_MAX_CHARS` 时只保留末尾完整字符（不切开代理对）并注明省略量；
 * - `flush` 立即上报尚未发出的剩余内容并清理定时器（供工具返回前在 `finally` 中调用）。
 *
 * @param report 底层进度上报回调。
 * @returns 包含 `push` 与 `flush` 方法的控制器。
 */
function createThrottledTextReporter(report: (progress: AgentToolProgress) => void): {
  /** 追加一段增量输出文本并按节流周期安排上报。 */
  push: (text: string) => void;
  /** 立即冲刷所有待上报文本并清除节流定时器。 */
  flush: () => void;
} {
  let pending = "";
  let skipped = 0;
  let timer: NodeJS.Timeout | undefined;
  /** 冲刷当前窗口暂存的文本并重置计数。 */
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (!pending && !skipped) return;
    const text = skipped > 0 ? `[... 已省略 ${skipped} 个字符 ...]\n${pending}` : pending;
    pending = "";
    skipped = 0;
    report({ text });
  };
  /** 累积新文本并在需要时裁剪头部、启动节流定时器。 */
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

/**
 * UTF-8 字符边界安全的头尾输出缓冲区。
 *
 * 不变式：`head` 始终是原输出（解码后）的连续前缀；`tail` 始终是原输出的连续后缀。
 * 一旦有任何内容进入 `tail`，`head` 即永久封口——字符边界对齐留下的少量空位宁可闲置，
 * 也不允许后到的内容补进 `head` 而排到先到内容之前。
 */
export class HeadTailBuffer {
  /** 已保存的输出开头前缀字节块。 */
  private head = Buffer.alloc(0);
  /** 已保存的输出末尾滚动后缀字节块。 */
  private tail = Buffer.alloc(0);
  /** `head` 是否已封口（一旦溢出部分进入 `tail` 即置为 `true`，不再向 `head` 追加）。 */
  private headSealed = false;
  /** 累计接收到的流原始字节总数（含可能被解码替换的非法字节）。 */
  private totalBytes = 0;
  /** 累计经 UTF-8 解码后再编码的有效字节总数（与 `head`/`tail` 同一统计口径）。 */
  private decodedBytes = 0;

  /**
   * 构造头尾输出缓冲区。
   *
   * @param headLimit 开头部分最多保留的字节数（默认 `BASH_HEAD_BYTES`）。
   * @param tailLimit 末尾部分最多保留的字节数（默认 `BASH_TAIL_BYTES`）。
   */
  constructor(
    private readonly headLimit: number = BASH_HEAD_BYTES,
    private readonly tailLimit: number = BASH_TAIL_BYTES
  ) {}

  /**
   * 向缓冲区追加一段已解码文本。
   *
   * @param text 已解码的完整字符文本（多个流应各自用独立的 `StringDecoder` 解码后再传入）。
   * @param rawByteLength 该段输出对应的底层原始字节长度。
   */
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

  /**
   * 将超出 `head` 容量的字节块滚动追加到 `tail`，并在超出 `tailLimit` 时从头部按 UTF-8 字符边界裁剪。
   *
   * @param chunk 待写入尾部缓冲区的 UTF-8 字节块。
   */
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

  /** 是否有中间内容被省略（即已解码总字节数大于 `head` 与 `tail` 实际保留字节数之和）。 */
  get isTruncated(): boolean {
    return this.decodedBytes > this.head.length + this.tail.length;
  }

  /**
   * 将缓冲区内容格式化为最终输出字符串（自动剥离末尾单个换行符）；
   * 若发生了截断，在 `head` 与 `tail` 之间插入省略字节统计及可选的日志文件检索提示。
   *
   * @param savedLog 已成功落盘的日志文件信息（含路径与是否完整记录）。
   * @returns 供模型阅读的完整或头尾摘要输出文本。
   */
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

/**
 * Bash 溢出输出日志管理器（单文件上限 `MAX_BASH_LOG_BYTES`）：
 * 输出接近头尾摘要容量前只在内存暂存（不超过约 48 KiB），超过 `openThreshold` 后才惰性创建日志文件，
 * 先写入暂存内容，再流式追加后续输出。写入的是与 `HeadTailBuffer` 相同的已解码文本，
 * 达到 `MAX_BASH_LOG_BYTES` 后按 UTF-8 边界截断并停止写入。任何日志 I/O 失败只会让日志不可用，不影响命令结果。
 */
class BashOutputLog {
  /** 尚未达到落盘阈值前在内存中暂存的文本片段。 */
  private pending: string[] = [];
  /** 内存暂存区累计的 UTF-8 字节数。 */
  private pendingBytes = 0;
  /** 已写入日志文件的累计字节数。 */
  private writtenBytes = 0;
  /** 底层文件写入流（超过阈值打开后才存在）。 */
  private stream: WriteStream | undefined;
  /** 日志文件的绝对路径。 */
  private filePath: string | undefined;
  /** 日志创建或写入过程中是否发生过 I/O 错误。 */
  private failed = false;
  /** 是否因达到 `MAX_BASH_LOG_BYTES` 上限而停止继续写入。 */
  private capped = false;
  /** 日志文件是否已由当前实例以 `"wx"` 模式成功创建（防止撞名时误删他人文件）。 */
  private created = false;

  /**
   * 构造按需落盘的 Bash 输出日志器。
   *
   * @param resolveDir 日志目录惰性解析函数；为 `undefined` 时禁用日志保存。
   * @param openThreshold 触发创建磁盘日志文件的输出字节数阈值。
   */
  constructor(
    private readonly resolveDir: (() => string) | undefined,
    private readonly openThreshold: number
  ) {}

  /**
   * 追加一段已解码输出文本：未达阈值时缓存在内存，超过阈值时打开日志文件并流式写入。
   *
   * @param text 已解码的 UTF-8 文本片段。
   */
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

  /**
   * 确保日志目录存在并以独占模式（`wx`，权限 `0600`）创建新的日志文件写入流。
   *
   * @returns 成功创建流时返回 `true`，失败时标记 `failed` 并返回 `false`。
   */
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

  /**
   * 在不超过 `MAX_BASH_LOG_BYTES` 预算的前提下，按 UTF-8 字符边界向日志流写入文本。
   *
   * @param text 待写入的 UTF-8 字符串。
   */
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

  /**
   * 结束写入并等待文件真正关闭（`"close"` 事件），再按流自身的错误状态判断日志是否可用：
   * `destroyed` 只表示开始销毁，`error` 事件可能尚未派发；`end()` 回调也不代表文件描述符已关闭。
   * `keep` 为 `false` 或写入失败时删除文件——但只删除本次成功创建的文件（`"wx"` 撞名失败时路径属于他人）。不会抛错。
   *
   * @param keep 是否需要保留该日志文件（仅命令未启动报错、未取消且输出确实发生截断时保留）。
   * @returns 日志可用时返回 `{ path, complete }`，否则返回 `undefined`。
   */
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

/** 当前宿主进程内所有尚在运行的受控 Bash 子进程组领头 PID（PGID）集合。 */
const activeProcessGroups = new Set<number>();

/** 是否已在 `process` 上惰性注册过 `"exit"` 同步强杀兜底监听器。 */
let exitHookRegistered = false;

/**
 * 登记一个新启动的受控进程组 PID，并在首次实际启动进程组时惰性注册宿主 `process.once("exit")` 同步 `SIGKILL` 兜底钩子，
 * 避免仅使用 Agent 核心循环时在模块加载期产生全局事件监听副作用。
 *
 * @param pid 以 `detached: true` 启动的子进程领头 PID（即进程组 ID）。
 */
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

/**
 * 探测指定进程组（`-pid`）内是否仍有存活进程。
 * 发送信号 `0` 不实际杀死进程；若抛错且错误码为 `EPERM` 表示进程存在但无权发信号，仍判定为存活。
 *
 * @param pid 进程组领头 PID。
 * @returns 进程组内仍有进程存活时返回 `true`，已全部退出时返回 `false`。
 */
function isProcessGroupAlive(pid: number): boolean {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (err) {
    return errnoCode(err) === "EPERM";
  }
}

/**
 * 向整个进程组（`-pid`）发送信号，忽略进程组已退出（`ESRCH`）等异常。
 *
 * @param pid 进程组领头 PID（为空时直接跳过）。
 * @param sig 要发送的 POSIX 信号（如 `"SIGTERM"` 或 `"SIGKILL"`）。
 */
function sendProcessGroupSignal(pid: number | undefined, sig: NodeJS.Signals): void {
  if (!pid) return;
  try {
    process.kill(-pid, sig);
  } catch {}
}

/**
 * `runBashProcessGroup` 执行受控 Bash 进程组时的可选配置。
 */
export interface BashRunOptions {
  /**
   * 在继承 `process.env` 的基础上覆盖这些环境变量
   * （`CI`、`GIT_PAGER`、`GIT_TERMINAL_PROMPT` 固定强制设为非交互值）；传入 `{}` 不会得到干净环境，不能当作白名单。
   */
  env?: NodeJS.ProcessEnv;
  /** 用于中途取消并终止整个进程组的取消信号。 */
  signal?: AbortSignal;
  /**
   * 输出超出头尾摘要时保存输出日志的目录（按需调用，仅在确实需要创建日志时才求值）；
   * 省略则不保存日志。单个日志上限 64 MiB，保留到宿主清理为止
   * （超时截断时也会保留并在错误消息中注明路径；取消、启动失败或未截断时不保留）。
   */
  logDir?: () => string;
  /** 每段已解码输出（`stdout`/`stderr` 按到达顺序）的同步观察回调，异常被隔离；用于进度展示，不影响结果。 */
  onOutput?: (text: string) => void;
}

/**
 * `runBashProcessGroup` 命令执行完成后的返回结果。
 */
export interface BashRunResult {
  /** Shell 主进程退出码；因信号终止时为 `null`。 */
  exitCode: number | null;
  /** 导致 Shell 主进程终止的信号名称；正常退出时为 `null`。 */
  exitSignal: NodeJS.Signals | null;
  /** 经 `HeadTailBuffer` 合并与截断格式化后的 `stdout`/`stderr` 输出文本。 */
  output: string;
  /** 已保存的输出日志元数据（仅在输出被截断且日志文件落盘成功时存在）。 */
  log?: { path: string; complete: boolean };
}

/**
 * 在独立受控进程组中执行 `/bin/bash` 命令（仅支持 POSIX 平台）：
 * - 以 `detached: true` 建立新进程组，命令结束、超时或取消时统一清理同一进程组内的全部后台子进程；
 * - 对 `stdout` 与 `stderr` 分别使用独立 `StringDecoder` 解码，汇总至 `HeadTailBuffer` 并在超长时按需写入 `BashOutputLog`；
 * - 超时或取消时执行优雅终止升级序列：先发 `SIGTERM` 并轮询进程组存活状态，超过 `KILL_GRACE_MS` 未退尽则升级为 `SIGKILL`。
 *
 * @param command 要执行的 Bash 命令字符串。
 * @param cwd 命令执行所在的工作目录路径。
 * @param timeoutMs 命令允许运行的最大毫秒数（超时后终止进程组并以包含已有输出的 `Error` 拒绝）。
 * @param options 环境变量覆盖、取消信号、日志目录解析器与实时输出回调。
 * @returns 包含退出码、终止信号、格式化输出与可选日志路径的 `BashRunResult`。
 */
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

    /** 汇总单段已解码输出到头尾缓冲区、溢出日志与实时观察回调。 */
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

    /** 幂等收尾：清理全部定时器与监听、冲刷解码器、销毁流、强杀残留进程组并等待日志关闭后结算 Promise。 */
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

    /** 安排管道排空宽限期定时器，防止孙进程继承管道句柄不关导致无限阻塞，到期后强制收尾。 */
    const scheduleDrainAndFinalize = () => {
      if (drainTimer || settled) return;
      drainTimer = setTimeout(() => finalize(), IO_DRAIN_GRACE_MS);
    };

    /** 终止流程中：当 Shell 已退出且整个进程组已无存活进程时，停止轮询与强杀升级计时，转入输出排空。 */
    const checkGroupDead = () => {
      if (shellExited && (!pid || !isProcessGroupAlive(pid))) {
        clearInterval(groupPollTimer);
        clearTimeout(killEscalationTimer);
        scheduleDrainAndFinalize();
      }
    };

    /** 启动进程组终止序列：先发送 `SIGTERM` 并高频轮询，宽限期 `KILL_GRACE_MS` 届满仍未退尽则发送 `SIGKILL`。 */
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

    /** 响应外部 `AbortSignal` 取消：若命令尚未自然退出则标记为已取消并启动进程组终止序列。 */
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

/** `read` 工具单次返回给模型的带行号格式化文本最大字符数预算。 */
const MAX_READ_CHARS = 30_000;

/** `read` 与 `edit` 工具允许读取的 UTF-8 文本文件最大磁盘字节上限（5 MiB）。 */
const MAX_READ_BYTES = 5 * 1024 * 1024;

/** 图片单独的磁盘大小上限（50 MiB）：`read` 会先缩放再发送，解码内存由 `MAX_IMAGE_PIXELS` 约束。 */
const MAX_IMAGE_FILE_BYTES = 50 * 1024 * 1024;

/** 图片发送给模型前等比缩放的最大长边像素上限（2000 px）。 */
const MAX_IMAGE_EDGE = 2000;

/** `sharp` 解码单张图片允许的最大像素总数上限（4000 万像素），防止超大分辨率图片耗尽内存。 */
const MAX_IMAGE_PIXELS = 40_000_000;

/** 本工具对单张图片的保守传输预算（Base64 字符数上限），不是所有模型统一的 API 上限。 */
const MAX_IMAGE_BASE64_CHARS = 4_500_000;

/**
 * 严格 UTF-8 文本解码器：`fatal: true` 遇到非法字节序列立即抛错；
 * `ignoreBOM: true` 表示“不要吞掉 BOM”，解码结果会保留开头的 `U+FEFF`，
 * 从而 `edit` 写回时 BOM 字节原样保留（默认 `false` 会静默剥离 BOM）。
 */
const utf8FatalDecoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/**
 * 解析目标路径的真实物理路径（消除沿途所有符号链接）：
 * 若目标文件或部分上级目录尚不存在（如准备新建文件），则向上逐级回溯到最近一个已存在的祖先目录解析 `realpath`，
 * 再将未创建的尾部路径片段拼回；若途中遇到目标不存在的悬空符号链接，则抛出拒绝访问错误。
 *
 * @param targetPath 待解析的目标路径。
 * @returns 解析符号链接后的规范真实路径。
 */
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

/**
 * 将文件字节缓冲区严格解码为 UTF-8 文本：
 * 先检查开头 1024 字节是否含 NUL（`0x00`）以快速拦截二进制文件，再通过 `utf8FatalDecoder` 校验完整 UTF-8 编码合法性。
 *
 * @param buf 文件原始字节缓冲区。
 * @param filePath 用于拼装错误提示的文件路径。
 * @returns 解码后的 UTF-8 字符串（若原文件带 BOM 则保留开头的 `U+FEFF`）。
 */
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

/**
 * 按文件头魔数（Magic Bytes）识别 `read` 支持的图片格式（PNG、JPEG、GIF87a/GIF89a、WebP；需至少 12 字节才能识别 WebP）。
 *
 * @param head 文件开头的字节片段（建议至少 12 字节）。
 * @returns 属于受支持图片格式时返回 `true`。
 */
function isSupportedImage(head: Buffer): boolean {
  return (
    head.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) ||
    (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) ||
    ["GIF87a", "GIF89a"].includes(head.toString("ascii", 0, 6)) ||
    (head.toString("ascii", 0, 4) === "RIFF" && head.toString("ascii", 8, 12) === "WEBP")
  );
}

/**
 * 只读打开文件并读取开头最多 `length` 个字节，用于在整文件读入内存前低开销探测文件类型。
 *
 * @param filePath 目标文件绝对路径。
 * @param length 要读取的最大头部字节数。
 * @returns 实际读取到的文件头字节切片。
 */
async function readFileHead(filePath: string, length: number): Promise<Buffer> {
  const handle = await fs.open(filePath, "r");
  try {
    const { bytesRead, buffer } = await handle.read(Buffer.alloc(length), 0, length, 0);
    return buffer.subarray(0, bytesRead);
  } finally {
    await handle.close();
  }
}

/**
 * 按需动态加载 `sharp` 原生图片处理模块：
 * 只使用文本工具的宿主不会因运行环境缺少 `sharp` 平台二进制而在加载本文件时报错。
 *
 * @returns `sharp` 默认导出函数。
 */
async function loadSharp(): Promise<typeof import("sharp").default> {
  try {
    return (await import("sharp")).default;
  } catch (err) {
    throw new Error("读取图片依赖 'sharp' 包，但加载失败。", { cause: err });
  }
}

/**
 * 根据原始字节长度精确计算标准 Base64 编码（含 `=` 填充位）后的字符长度。
 *
 * @param byteLength 二进制数据的字节长度。
 * @returns 对应的 Base64 字符串长度。
 */
function base64Length(byteLength: number): number {
  return 4 * Math.ceil(byteLength / 3);
}

/**
 * 判断 PNG 缓冲区是否确定为静态图：按 chunk 边界扫描到首个 `IDAT` 为止，
 * 若出现 `acTL`（APNG 动画控制块，规范要求位于 `IDAT` 之前）或 chunk 结构无法可靠解析则返回 `false`。
 * 仅用于决定能否原样直通返回原始字节，误判为“非静态”只会安全退回重编码分支。
 *
 * @param buf 完整的 PNG 文件字节缓冲区。
 * @returns 确认为普通静态 PNG 时返回 `true`。
 */
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

/**
 * 读取并规范化图片为可发送给多模态模型的文本说明 + Base64 图像内容块：
 * 自动按 EXIF 转正、等比缩放到 `MAX_IMAGE_EDGE` 以内（不裁剪、不放大），动图只取第一帧。
 * - 已合规的普通静态 JPEG/PNG/WebP（无需转正/缩放、8 位 sRGB 或灰度、无 ICC、体积在预算内）经完整解码校验后
 *   原样返回压缩字节，保留原始编码质量；注意这也会保留 EXIF/XMP 等元数据（处理分支会按 `sharp` 默认移除）。
 * - 其余情况输出尽量沿用 API 支持的源格式（JPEG/WebP 原格式，PNG/GIF 输出 PNG）；
 *   超出预算时从原图管线重新铺白底转 JPEG，避免在第一次有损结果上再次有损编码。
 *
 * @param buf 图片原始文件字节。
 * @param filePath 用于生成说明文案的文件路径。
 * @param signal 可选的取消信号。
 * @returns 包含图片元信息文本块与 Base64 `image` 块的数组。
 */
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

/**
 * 原子提交写入 UTF-8 文本文件：同目录创建独占临时文件 → 完整写入并 `fsync` → 关闭 → `rename` 原子覆盖目标。
 * - 提交边界是 `rename`：提交前观察到的失败或取消会清理临时文件，原目标不变；
 *   `rename` 一旦开始即可能完成，此后按成功返回（不会在提交后再报告取消，避免“报失败、实际已改”）。
 * - 只提供运行期间的原子替换与提交前失败保护；不承诺系统崩溃或掉电后的结果（未对目录 `fsync`）。
 * - 只保留普通 `rwx` 权限位；`setuid`/`setgid`/`sticky`、属主/属组、ACL、xattr 不保留。
 * - 目标已存在但当前进程不可写（如 `0444`）时拒绝，保持与直接写入相同的语义——`rename` 只看目录权限，
 *   否则会悄悄覆盖用户设为只读的文件。该检查与 `rename` 之间仍有时间窗口，仅防误操作。
 * - `targetPath` 应为已解析的真实路径：工作区内的符号链接因此被保留；硬链接会被解绑（目标路径指向新 inode）。
 *
 * @param targetPath 已解析符号链接的目标文件真实路径。
 * @param content 要完整写入的 UTF-8 文本内容。
 * @param signal 可选的取消信号。
 */
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

/**
 * 剥离字符串开头可能存在的单个 UTF-8 BOM（`U+FEFF`）字符。
 *
 * @param text 原始解码文本。
 * @returns 不含前导 BOM 的文本。
 */
function stripLeadingBom(text: string): string {
  return text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
}

/**
 * 将解码后的文件内容格式化为带 1 基行号（如 `   1 | ...`）的分页展示文本：
 * - 展示时去掉不可见的 UTF-8 BOM（避免模型将其误复制进 `edit` 的 `old_text`；`edit` 写回时仍会原样保留文件中的 BOM）；
 * - 按 `offset` / `limit` 截取行范围，并受 `MAX_READ_CHARS` 总字符预算保护，截断时附带后续分页读取提示。
 *
 * @param decodedText 经 UTF-8 解码的文件全文。
 * @param offset 起始读取行号（从 1 开始，默认 1）。
 * @param limit 最大读取行数（省略则读至文件末尾或达到字符预算上限）。
 * @returns 格式化后的带行号文本或边界提示信息。
 */
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

/**
 * 在原文本中查找所有不把 `"\r\n"` 从中间劈开的有效精确匹配位置（最多收集 2 处即可判断唯一性）。
 *
 * @param content 待检索的原文件全文。
 * @param needle 待定位的目标子串（非空）。
 * @returns 有效匹配的起始字符下标数组（长度 0、1 或 2）。
 */
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

/**
 * 在同一份原文本上校验并批量应用多处互不重叠的文本替换：
 *
 * 换行策略（逐项判断）：
 * 1. 纯 CRLF 文件：无论走精确匹配还是 LF→CRLF 回退匹配，`new_text` 中的换行一律转为 CRLF，保持文件风格；
 * 2. LF 文件或混合换行文件：精确匹配时按原样替换（尊重调用方给出的换行）；
 *    精确匹配未命中而 CRLF 归一化后命中时，被替换区域本身是 CRLF，`new_text` 同步转为 CRLF。
 * 任何情况下都不会把已有的 `"\r\n"` 从中间劈开。
 *
 * 批量语义：所有 `edits` 都在同一份原文上定位，每项必须唯一命中、区域互不重叠（相邻可以）；
 * 任一项失败或替换后全文无实际变化则直接抛错（整体不写入）。不做引号/空白/Unicode 归一化等模糊匹配。
 *
 * @param rawContent 修改前的原文件完整字符串（含可能存在的开头 BOM）。
 * @param edits 待应用的旧文本与新文本替换项列表。
 * @param filePath 用于拼装错误提示的文件路径。
 * @returns 替换后的完整新文本 `updated` 以及首个发生实际变化的 1 基行号 `firstChangedLine`。
 */
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

/**
 * 将 `BashRunResult` 转换为符合 `AgentToolResult` 契约的工具返回结构：
 * 非零退出（含信号终止）时返回 `isError: true` 并附带退出状态说明，同时将退出码、信号与日志信息放入 `details` 交给宿主。
 *
 * @param result 底层进程组执行完成后的原始结果。
 * @returns 包含模型可读文本 `content` 与宿主侧元数据 `details` 的 `AgentToolResult`。
 */
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

/**
 * 判断规范化后的绝对路径 `targetPath` 在路径层级上是否位于根目录 `rootDir` 内部（含等于 `rootDir` 本身）。
 *
 * @param rootDir 基准根目录绝对路径。
 * @param targetPath 待校验的目标绝对路径。
 * @returns 位于 `rootDir` 之内时返回 `true`，越界（如 `..` 或跨盘符）时返回 `false`。
 */
function isInsideRoot(rootDir: string, targetPath: string): boolean {
  const rel = path.relative(rootDir, targetPath);
  return rel !== ".." && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel);
}

/**
 * 将相对或绝对路径解析并校验在工作区根目录之内，返回解析符号链接后的真实路径。
 *
 * 注意：路径检查仅用于防止误操作（如 `..` 目录穿越、指向工作区外的符号链接），不是安全沙箱：
 * 检查与后续按路径打开之间存在 TOCTOU 窗口，硬链接也不受 `realpath` 约束，且 `bash` 工具本身可执行任意命令。
 * 返回已解析符号链接的真实路径，后续读写均基于该路径，避免二次跟随链接，并让原子写入保留工作区内的符号链接。
 *
 * @param workspaceRoot 工作区根目录路径。
 * @param filePath 工具入参给定的相对或绝对文件路径。
 * @returns 确认位于工作区真实目录内的目标真实路径。
 */
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

/**
 * 创建绑定到指定工作区根目录的内置 `read` 工具实例：
 * 支持 UTF-8 文本按行分页读取（带 1 基行号）以及 JPEG/PNG/GIF/WebP 图片自动转正与等比缩放读取（工具本身不做语法分析）。
 *
 * @param workspace 允许读取文件的工作区根目录路径。
 * @returns 名为 `"read"` 的 `AgentTool` 实例。
 */
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
      /** 按文件是否为受支持图片分别校验对应的磁盘字节大小上限。 */
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
      // 文本与图片的大小上限不同：读入整个文件前先按文件头判断类型；读入后再按实际内容复核一次
      assertWithinLimit(isSupportedImage(await readFileHead(fullPath, 12)), stat.size);

      const buf = await fs.readFile(fullPath, { signal: sig });
      const isImage = isSupportedImage(buf);
      assertWithinLimit(isImage, buf.length);
      if (isImage) return readImage(buf, file_path, sig);

      return formatReadText(decodeStrictUtf8(buf, file_path), offset, limit);
    },
  });
}

/**
 * 创建绑定到指定工作区根目录的内置 `write` 工具实例：
 * 面向 UTF-8 文本文件，自动创建缺失的父目录，并通过同目录临时文件 + `fsync` + `rename` 原子提交写入。
 *
 * @param workspace 允许写入文件的工作区根目录路径。
 * @returns 名为 `"write"` 的 `AgentTool` 实例。
 */
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

/**
 * 创建绑定到指定工作区根目录的内置 `edit` 工具实例：
 * 面向已有 UTF-8 文本文件，支持在同一份原文上定位一处或多处互不重叠的精确替换（兼容 CRLF 换行并保留 BOM），校验全部通过后一次原子提交。
 *
 * @param workspace 允许编辑文件的工作区根目录路径。
 * @returns 名为 `"edit"` 的 `AgentTool` 实例。
 */
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

/**
 * `createBashTool` 创建内置 Bash 工具时的可选配置项。
 */
export interface BashToolOptions {
  /** 在继承宿主 `process.env` 的基础上附加或覆盖的子进程环境变量。 */
  env?: NodeJS.ProcessEnv;
  /**
   * `bash` 输出被截断时保存输出日志的目录（相对路径基于 `workspace`，单个日志上限 64 MiB）：
   * - 省略：首次需要时在系统临时目录下创建本工具实例专用目录（`agent-bash-logs-XXXXXX`）；该目录默认在工作区外，
   *   仅限工作区的 `read` 工具会拒绝读取，模型需用 `bash` 检索；若希望 `read` 可读，宿主可将其设为工作区内的目录。
   * - `null`：不保存日志，仅返回头尾摘要。
   * 日志不会自动删除，清理由宿主负责。
   */
  logDir?: string | null;
}

/**
 * 创建以指定工作区为执行目录（`cwd`）的受控进程组内置 `bash` 工具实例（仅支持 POSIX 平台）：
 * 命令退出、超时或取消时一并清理同进程组后台子进程；支持超长输出头尾摘要与可选溢出日志落盘，并按节流周期上报实时增量输出进度。
 *
 * @param workspace 命令执行的默认工作区目录路径。
 * @param options 环境变量覆盖与溢出日志目录策略配置。
 * @returns 名为 `"bash"` 的 `AgentTool` 实例。
 */
export function createBashTool(workspace: string, options?: BashToolOptions) {
  const root = path.resolve(workspace);
  const logDirOption = options?.logDir;
  let defaultLogDir: string | undefined;
  /** 按需返回溢出日志保存目录路径；配置为 `null` 时为 `undefined`（不保存日志）。 */
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
