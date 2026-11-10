import type Anthropic from "@anthropic-ai/sdk";

const RECOVERY_RESULT_TEXT = "Recovered from incomplete prior run: tool execution state is unknown.";

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
export function buildTrailingRecoveryMessage(
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
export function messagesHaveCacheControl(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]): boolean {
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
export function validatePreparedMessages(
  prepared: unknown,
  history: readonly Anthropic.Beta.Messages.BetaMessageParam[]
): asserts prepared is Anthropic.Beta.Messages.BetaMessageParam[] {
  if (!Array.isArray(prepared) || prepared.length === 0) {
    throw new Error("prepareMessages must return a non-empty array of messages.");
  }
  const tail = inProgressToolTurn(history);
  const offset = prepared.length - tail.length;
  if (offset < 0 || !tail.every((msg, i) => structurallyEqual(prepared[offset + i], msg))) {
    throw new Error(
      "prepareMessages must keep the in-progress tool turn (the last assistant turn containing tool_use and every " +
        "message after it) unchanged at the end of the returned messages."
    );
  }
}

// 每个 tool_result 都必须对应紧邻其前的 assistant 回合中的 tool_use；否则 API 会拒绝请求。
// 这里提前给出指明 id 的错误，而不是悄悄删除（裁剪策略属于宿主）。缺失结果的补全由 buildApiRequestMessages 负责。
export function assertToolResultsPaired(messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]): void {
  const turns = groupIntoTurns(messages);
  for (let t = 0; t < turns.length; t++) {
    if (turns[t].role !== "user") continue;
    const prev = turns[t - 1];
    const known = new Set(prev?.role === "assistant" ? collectToolUseIds(messages, prev.indices) : []);
    for (const id of collectToolResultIds(messages, turns[t].indices)) {
      if (!known.has(id)) {
        throw new Error(
          `Request messages contain a tool_result for '${id}' without a matching tool_use in the immediately ` +
            "preceding assistant turn. When trimming history, remove tool_use and tool_result blocks together."
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
export function buildApiRequestMessages(
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
