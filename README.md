# stateless-agent

无状态的模型与工具调用循环。宿主管理会话、工具和生命周期；Agent 在每次主模型请求前检查上下文预算，必要时生成摘要，再继续执行任务。

## 使用

需要 Node.js 24 或以上版本，以及兼容 Anthropic Messages API 的模型服务。

```sh
npm ci
npm run typecheck
```

配置 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL`，使用兼容服务时配置 `ANTHROPIC_BASE_URL`。`contextWindow` 必填，应填写所选模型实际支持的上下文窗口；下面的 `200000` 仅为示例。

```ts
import Anthropic from "@anthropic-ai/sdk";
import { appendFile } from "node:fs/promises";
import { AgentRunError, createReadTool, runAgent } from "./stateless-agent.ts";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL,
});

let messages: Anthropic.Beta.Messages.BetaMessageParam[] = [
  { role: "user", content: "读取 README.md，说明这个项目的用途。" },
];

try {
  const result = await runAgent({
    client,
    model: process.env.ANTHROPIC_MODEL!,
    contextWindow: Number(process.env.ANTHROPIC_CONTEXT_WINDOW ?? 200_000),
    tools: [createReadTool(process.cwd())],
    messages,
    onMessage: (message) =>
      appendFile("conversation.jsonl", JSON.stringify(message) + "\n"),
    onEvent: (event) => {
      if (event.type === "text_delta") process.stdout.write(event.delta);
      if (event.type === "compaction") {
        console.log("\n上下文已压缩，估算 Token：",
          event.estimatedTokensBefore, "→", event.estimatedTokensAfter);
      }
    },
  });
  messages = result.messages;
} catch (error) {
  if (error instanceof AgentRunError) messages = error.messages;
  throw error;
}
```

下一次运行使用保存的 `messages`，需要新任务时再追加用户消息。输入的用户消息由调用方自行保存；`onMessage` 只提供运行中新追加的完整消息。

## 上下文与记录

| 接口 | 用途 |
| --- | --- |
| `messages` | 当前有效上下文；压缩时以摘要替换较早内容，不就地修改调用方输入 |
| `result.messages` / `AgentRunError.messages` | 正常结束、取消或运行失败后的有效上下文，供保存和续跑 |
| `onMessage` | 接收实际追加的完整消息，可保存原始记录；会等待回调完成，摘要替换不触发 |
| `onEvent` | 观察流式输出、工具执行和压缩完成；异常被隔离，异步回调可能晚于运行结束 |

有效上下文用于继续推理，完整聊天记录由宿主管理。压缩状态已包含在返回的 `messages` 中，没有额外摘要检查点。`compaction` 事件是尽力通知，不承担可靠保存；需要保存上下文时使用正常或异常返回的消息快照。

`getContextInjection` 在每批工具结果保存后读取增量文本，将其作为 `user` 消息追加并交给 `onMessage`。它不在首次请求前或最终回复后调用，也不能代替工具执行副作用前的检查。

## 自动压缩

- 每轮正常发起主模型请求前由内置 `compactMessages` 检查上下文预算（预算足够时不请求摘要模型）；明确超窗恢复时强制压缩，思考前缀错误恢复时剥离思考块后直接重发。
- 预算计入系统提示、工具定义、消息及输出预留。本次运行内，最近一次完整主响应的用量（含缓存和输出）用于校准已有前缀；新增工具结果、注入消息和摘要单独估算。上下文改写后清除旧校准，独立运行从本地估算开始。
- 摘要使用同一个 `client` 和 `model` 流式生成，每次输入最多约 `48000` 个估算 Token，并受模型窗口限制。摘要正文预留最多 `4096` Token；摘要请求的推理与正文合计输出上限为 `min(16384, maxTokens)`，不强制关闭模型思考。
- 完成摘要替换或移除服务端明确拒绝回放的思考块后，发出 `compaction` 事件，其中 Token 数是估算值。截断、空白或预算不合格的摘要不会被采用；取消或失败时保留已采用的有效上下文。
- 服务端明确报告输入超窗时，进一步压缩并重试一次；明确的思考前缀绑定错误也最多恢复一次。两类恢复可先后发生，均占用 `maxTurns`，只重试主请求，不重复执行工具。普通 `413` 请求体过大和输出参数错误不触发压缩。
- `usage` 包含主模型与摘要请求中已取得完整响应的用量；`turns`、`maxTurns` 只计算主模型请求。

`maxTokens` 默认取 `min(98304, floor(contextWindow / 4))`；显式设置时必须为正整数且小于 `contextWindow`，也应符合模型的输出限制。`maxTurns` 默认 `200`。

压缩时保留当前工具回合的 `thinking` 原块，由服务端判断能否继续回放。只有收到明确的“绑定到不同对话”错误，且不是手动 `enabled` 模式时，才移除思考块并重试；损坏签名、要求原样保留思考的错误不自动修复。模型支持时，宿主也可显式配置 `thinking.block_binding.prefix_mismatch_behavior: "drop_block"`，Agent 会附带对应 beta 标记。

临时纯文本 `system` 可使用 `clear_at: "next_user_message"`，Agent 自动附带相应 beta 标记；后续任意 `user` 消息（包括工具结果）会使其失效。普通请求仍原样回传，压缩时移除失效条目且不将其写进摘要。持久指令应放在 `systemPrompt`；中间的持久 `system` 保持原位，可能限制可压缩范围。最新消息或工具结果本身过大时，需要缩小输入、工具输出或输出预留。

## 工具与验证

内置工具可按需装配：`createReadTool`、`createWriteTool`、`createEditTool`、`createBashTool`；自定义工具使用 `defineTool`。工具在单次运行中串行执行，Bash 工具需要 POSIX 环境。

Agent 主循环、上下文压缩与内置工具始终保留在 `stateless-agent.ts` 单文件中。仓库不提交测试文件。

`npm run example` 运行 `example.ts` 中的真实模型示例，会访问配置的模型服务并执行工具；`npm run typecheck` 仅做类型检查。
