import type Anthropic from "@anthropic-ai/sdk";

/** 相邻同角色消息在 API 中属于一个回合；system 消息不打断合并。 */
interface ConversationTurn {
  role: Exclude<Anthropic.Beta.Messages.BetaMessageParam["role"], "system">;
  firstIndex: number;
  lastIndex: number;
  toolUseIds: Set<string>;
  toolResultIds: Set<string>;
}

/** 请求校验、工具恢复与压缩共用同一套回合边界。 */
export function groupIntoTurns(
  messages: readonly Anthropic.Beta.Messages.BetaMessageParam[]
): ConversationTurn[] {
  const turns: ConversationTurn[] = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role === "system") continue;
    let turn = turns[turns.length - 1];
    if (!turn || turn.role !== msg.role) {
      turn = {
        role: msg.role,
        firstIndex: i,
        lastIndex: i,
        toolUseIds: new Set(),
        toolResultIds: new Set(),
      };
      turns.push(turn);
    } else {
      turn.lastIndex = i;
    }
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_use") turn.toolUseIds.add(block.id);
        else if (block.type === "tool_result") turn.toolResultIds.add(block.tool_use_id);
      }
    }
  }
  return turns;
}
