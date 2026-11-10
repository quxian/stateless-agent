import type Anthropic from "@anthropic-ai/sdk";
import type {
  AgentEvent,
  AgentOptions,
  AgentResult,
  AgentStopReason,
  AgentTool,
  AgentToolProgress,
  AgentToolResult,
  AgentUsage,
  ToolOutput,
} from "./types.ts";
import {
  assertToolResultsPaired,
  buildApiRequestMessages,
  buildTrailingRecoveryMessage,
  messagesHaveCacheControl,
  validatePreparedMessages,
} from "./messages.ts";
import { sliceStringSafe } from "./utils.ts";

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
  if (typeof output === "string") return sliceStringSafe(output, 500);
  try {
    // 图片块只保留类型说明，避免把 Base64 数据塞进观察事件
    const summary = output.map((block) =>
      block.type === "image"
        ? { type: "text", text: `[Image: ${block.source.type === "base64" ? block.source.media_type : block.source.type}]` }
        : block
    );
    return sliceStringSafe(JSON.stringify(summary) ?? "", 500);
  } catch {
    return "[Complex tool output]";
  }
}

export async function runAgent(options: AgentOptions): Promise<AgentResult> {
  const {
    client,
    model,
    signal,
    maxTurns = 50,
    thinking,
    prepareMessages,
    getContextInjection,
    onMessage,
    onEvent,
  } = options;

  if (!Number.isInteger(maxTurns) || maxTurns < 1) {
    throw new Error(`Invalid maxTurns: ${maxTurns}. Must be a positive integer.`);
  }

  const budgetTokens = thinking && "budget_tokens" in thinking ? thinking.budget_tokens : 0;
  const maxTokens = options.maxTokens ?? (budgetTokens ? budgetTokens + 4096 : 8192);
  if (budgetTokens && maxTokens <= budgetTokens) {
    throw new Error(`maxTokens (${maxTokens}) must be greater than thinking.budget_tokens (${budgetTokens}).`);
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
      ? `You are a helpful assistant that can use tools to complete tasks. Available tools: ${toolNamesList}.`
      : "You are a helpful assistant.");

  let lastText = "";
  let stopReason: AgentStopReason = null;
  let turns = 0;
  const usage: AgentUsage = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 };

  // 观察事件：仅在有订阅者时才构造事件（含复制输入、生成摘要）；构造或回调中的同步异常与异步拒绝均被隔离
  const emit = (build: () => AgentEvent) => {
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
          const pending = Promise.resolve().then(() => prepareMessages({ messages: [...messages], turn: turns + 1, signal }));
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

      const stream = client.beta.messages
        .stream(
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
          { signal }
        )
        .on("thinking", (delta) => emit(() => ({ type: "thinking_delta", delta })))
        .on("text", (delta) => emit(() => ({ type: "text_delta", delta })));

      let response: Anthropic.Beta.Messages.BetaMessage;
      try {
        response = await stream.finalMessage();
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

      if (response.stop_reason !== "tool_use" && toolUseBlocks.length === 0) {
        break;
      }

      const userContent: Anthropic.Beta.Messages.BetaContentBlockParam[] = [];

      if (response.stop_reason !== "tool_use") {
        for (const b of toolUseBlocks) {
          userContent.push({
            type: "tool_result",
            tool_use_id: b.id,
            is_error: true,
            content:
              response.stop_reason === "max_tokens"
                ? "Tool call was truncated due to hitting max_tokens and was NOT executed. Retry with smaller input."
                : `Tool call was not executed because generation stopped with reason '${response.stop_reason}'.`,
          });
        }
        if (response.stop_reason !== "max_tokens") {
          await appendMessage({ role: "user", content: userContent });
          break;
        }
      } else {
        if (toolUseBlocks.length === 0) break;

        for (const block of toolUseBlocks) {
          if (signal?.aborted) {
            userContent.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: "Tool execution was SKIPPED (never started) because the agent run was aborted.",
              is_error: true,
            });
            continue;
          }

          const tool = toolMap.get(block.name);
          if (!tool) {
            userContent.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: `Error: Unknown tool '${block.name}'. Available tools: ${toolNamesList || "(none)"}`,
              is_error: true,
            });
            continue;
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
            resultBlock = {
              type: "tool_result",
              tool_use_id: block.id,
              content: "Tool execution was SKIPPED (aborted during tool_start) before running.",
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
                ...(onEvent ? { reportProgress } : {}),
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
                ? "Tool execution was INTERRUPTED after starting because the run was aborted; partial side effects may have occurred. Verify workspace state."
                : err instanceof Error
                  ? err.message
                  : String(err);
              resultBlock = { type: "tool_result", tool_use_id: block.id, content: errMsg, is_error: true };
            } finally {
              running = false;
            }
          }

          userContent.push(resultBlock);
          const finished = resultBlock;
          emit(() => ({
            type: "tool_end",
            id: block.id,
            name: block.name,
            isError: finished.is_error === true,
            outputSummary: summarizeToolOutput(finished.content),
            ...(details !== undefined ? { details } : {}),
          }));
        }
      }

      // 1. 先记录并持久化本轮工具结果（或截断未执行结果）
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

      // 若在保存注入消息期间触发了取消，优先报告 aborted 而非 max_turns
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
