import type Anthropic from "@anthropic-ai/sdk";

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
  // 工具因取消而失败时，应抛出 signal.reason、AbortError，或以 signal.reason 为 cause 的错误，
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
  // 可选的显式缓存断点；设置后视为调用方接管缓存配置，runAgent 不再自动添加默认断点
  cache_control?: Anthropic.Beta.Messages.BetaTool["cache_control"];
  parse: (input: unknown) => TInput;
  run: (
    input: TInput,
    context: AgentToolContext
  ) => Promise<ToolOutput | AgentToolResult> | ToolOutput | AgentToolResult;
}

export type AgentEvent =
  | { type: "thinking_delta"; delta: string }
  | { type: "text_delta"; delta: string }
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
  // 本次运行可用的工具（默认无工具）。文件/命令能力用 createWorkspaceTools(workspace, { env, bashLogDir }) 创建。
  tools?: AgentTool[];
  // 默认使用不含环境信息的通用提示词；工作目录等上下文由宿主按需写入
  systemPrompt?: string;
  maxTokens?: number;
  maxTurns?: number;
  thinking?: Anthropic.Beta.Messages.BetaThinkingConfigParam;
  // 是否自动添加默认 ephemeral 缓存断点（默认 true）。每次请求按最终发送的消息与 tools 判断：
  // 两者均无显式 cache_control 时才自动添加；设为 false 只是不再自动添加，调用方自带的 cache_control 仍原样发送。
  autoCacheBreakpoints?: boolean;
  signal?: AbortSignal;
  // 本次运行每新增一条完整消息时按序调用（请勿就地修改消息对象）；若抛错将中止后续动作并通过 AgentRunError 带出快照。
  // 该 Promise 会被完整等待，不会因 signal 取消而被放弃——宿主需自行保证保存操作有合理期限。
  // 这是“完整消息”持久化点而非工具事务日志：同一批次中已完成工具的结果，要等整批结束后才作为一条消息落盘。
  onMessage?: (message: Readonly<Anthropic.Beta.Messages.BetaMessageParam>) => void | Promise<void>;
  // 观察侧事件回调：同步异常与异步拒绝均被隔离，不阻塞主流程返回（runAgent 返回时异步回调可能尚未完成，需要可靠投递的宿主自行排队/flush）
  onEvent?: (event: AgentEvent) => void | Promise<void>;
  // 每次模型请求（含第一次）之前调用，返回本次实际发送的消息；只影响本次请求，完整历史保持不变。
  // 用于裁剪/摘要较早历史、注入临时检索材料、替换旧图片等。约束（违反时以 AgentRunError 中止）：
  // - 进行中的工具回合（最后一条含 tool_use 的 assistant 回合及其后全部消息）必须原样位于返回结果末尾——
  //   开启 thinking 时该回合的 thinking 签名必须原样回传；临时材料可插在它之前。最后的 assistant 回合不含 tool_use 时不受此限。
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
  // 返回的文本作为 user 消息写入完整历史。非取消异常将中止运行以防遗漏关键指令。
  // 覆盖范围：运行开始时与模型给出最终回复（无工具调用）之后都不会调用；同一批次中，后面的工具在钩子之前就已执行。
  // 因此它不能替代“副作用发生前检查最新状态”——例如发送消息前检查是否有新指令，应放在该工具内部、紧邻实际副作用处。
  getContextInjection?: (context: { signal?: AbortSignal }) => Promise<string | null>;
}

export interface AgentResult {
  // 调用方历史 + 本次运行新增的消息（含末尾恢复消息）。
  // 注意：历史中段缺失工具结果的补全只存在于每次请求的副本中，不会写回这里，也不会经过 onMessage。
  // 只有完整取得的 assistant 响应才会写入：流中途出错或取消时，已通过 text_delta/thinking_delta 展示的部分输出
  // 不会出现在这里；需要保留“用户已看到的内容”的宿主，应在展示记录中自行保存，而不是把半成品当作模型上下文。
  // 请求级重试由注入的 Anthropic 客户端配置（如 maxRetries）负责，runAgent 不做整轮重试。
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  lastText: string;
  stopReason: AgentStopReason;
  turns: number;
  usage: AgentUsage;
}
