# stateless-agent

无状态的模型与工具调用循环。宿主管理会话、工具和生命周期；Agent 在每轮正常请求前检查上下文预算，必要时生成摘要，再继续执行任务。

## 使用

需要 Node.js 24 或以上版本，以及兼容 Anthropic Messages API 的模型服务。

```sh
npm ci
npm run typecheck
```

配置 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL`，使用兼容服务时配置 `ANTHROPIC_BASE_URL`。`contextWindow` 必填，应填写所选模型实际支持的上下文窗口。

`example.ts` 展示两条明确用户输入的单进程内存会话，包括正常与异常快照接续、结束原因判断。它不包含文件会话存储或自动续跑。

下面仅示意宿主的 JSONL 记录接点；`options`、`messages` 和本轮 `userMessage` 由宿主准备：

```ts
import type Anthropic from "@anthropic-ai/sdk";
import { appendFile } from "node:fs/promises";
import { AgentRunError, runAgent } from "./stateless-agent.ts";

const recordMessage = (message: Readonly<Anthropic.Beta.Messages.BetaMessageParam>) =>
  appendFile("conversation.jsonl", JSON.stringify(message) + "\n");

await recordMessage(userMessage);
messages = [...messages, userMessage];

try {
  const result = await runAgent({ ...options, messages, onMessage: recordMessage });
  messages = result.messages;
} catch (error) {
  if (error instanceof AgentRunError) messages = error.messages;
  throw error;
}
```

初始用户输入由宿主记录；`onMessage` 只提供本次运行中新追加的完整消息。上面的片段只追加原始记录并更新内存上下文；上下文快照的持久化由宿主另行完成，同一会话的运行和保存应串行处理。

## 上下文与记录

| 数据 | 入口与出口 | 用途 |
| --- | --- | --- |
| 原始追加记录 | 宿主记录的初始输入、`onMessage` 收到的新消息 | 保留完整记录；摘要替换不触发 `onMessage` |
| 有效上下文 | 传入的 `messages`、`result.messages` / `AgentRunError.messages` | 供 `runAgent` 续跑；旧历史可能已被摘要替换，不就地修改调用方输入数组 |
| 模型请求副本 | 内部构造 | 按需补齐缺失工具结果、添加缓存标记；与上下文快照和原始记录不保证逐项相同 |

`onMessage` 按追加顺序调用并完整等待：消息先进入内存，再交给回调。回调失败会终止运行，`AgentRunError.messages` 仍保留该消息，取消也不会掩盖该错误。快照不表示存储已确认成功；保存可能未完成，也可能已写入后报错，需要宿主处理该次保存。使用快照续跑时，已有消息不会重新触发 `onMessage`，不能依赖续跑自动补写日志，也不能用压缩后的快照覆盖原始记录。

恢复末尾悬空的 `tool_use` 时，会追加表示“执行状态未知”的 `user` 结果并交给 `onMessage`，保留后续 assistant 前的回合边界。历史中段缺失的结果只在请求副本中补齐，不改写既有记录或有效上下文，也不给压缩增加新的配对约束。恢复步骤不自动重执行旧工具；“未知”不代表工具未产生副作用。

`onEvent` 用于观察流式输出、工具执行和压缩；异常被隔离，异步回调可能晚于运行结束。压缩状态包含在返回的 `messages` 中，运行中没有可靠的上下文检查点；`compaction` 只有估算信息，不承担保存。进程被强制终止时，不保证能够取得最终快照。

### 运行中注入

`getContextInjection` 在每批工具结果追加并完成 `onMessage`（若提供）后读取增量输入，接受非空字符串，或由 `text`、`image`、`document` 组成的只读 `AgentContextBlock` 数组；`null`、`""`、`[]` 表示无新增内容。入口检查基本结构，复制数组与内容块顶层后作为 `user` 消息追加并交给 `onMessage`。

回调返回的是待追加内容：取消或校验失败可能阻止它进入上下文。队列读取不能直接把 `shift()` 等移除操作当作交付成功；宿主应在对应 `onMessage` 保存完成后确认消费，失败时结合错误快照处理。已经追加的内容也可能因随后取消或达到 `maxTurns` 而尚未发给模型。这个钩子不在首次请求前或最终回复后调用，也不能代替工具执行副作用前的检查。

宿主需自行准备目标服务与模型支持的内容块；直接注入不经过内置 `read` 的图片校验与缩放。交付后不得再修改内容块及其嵌套数据；块上显式设置 `cache_control` 时，默认缓存断点策略整体让位。

## 结束与接续

`lastText` 是最近一次完整接收的主模型响应中的文本，可能为空、是工具调用前的说明，或因输出上限而不完整。正常返回表示本次调用结束，宿主需要结合 `stopReason` 决定后续操作：

| `stopReason` | 含义与接续 |
| --- | --- |
| `end_turn` | 模型结束本轮；是否满足用户任务仍需判断，可以接收下一条用户输入 |
| `max_tokens` | 达到单次输出上限，内容可能不完整；根据上下文决定是否调整输出预算或继续 |
| `max_turns` | 达到主请求轮次上限；宿主明确决定是否继续，不应靠自动循环绕过限制 |
| `aborted` | 本次运行已取消；保留上下文，不自动重启 |
| 其他模型停止原因 | 按对应原因处理，不统一视为任务成功 |

下一次运行接回 `result.messages`；运行异常时先保留 `AgentRunError.messages` 再处理错误。普通参数校验错误不携带运行快照。新任务追加新的用户消息；继续未完成运行时，不重复追加原输入。接回快照不会自动解决保存失败或非法输入等错误。

## 自动压缩

- 每轮正常发起主模型请求前由内置 `compactMessages` 检查上下文预算（预算足够时不请求摘要模型）；明确超窗恢复时强制压缩，思考前缀错误恢复时剥离思考块后直接重发。
- 预算计入系统提示、工具定义、消息及输出预留。图片以及无法直接估算内容的文档源（`base64`、`url`、`file`）使用固定启发式近似值，避免把传输编码长度误计为模型阅读文本；实际消耗取决于附件与模型。本次运行内，最近一次完整主响应的用量（含缓存和输出）用于校准已有前缀，新增工具结果、注入消息和摘要单独估算；前缀校准与单次超窗恢复仅能缓解部分估算误差。上下文改写后清除旧校准，独立运行从本地估算开始。
- 摘要使用同一个 `client` 和 `model` 流式生成，每次输入最多约 `48000` 个估算 Token，并受模型窗口限制；压缩旧历史时会省略图片与二进制文档数据（保留文本源文档、文档标题及已有文字观察），后续仍需查看细节的附件建议附带可由工具重新读取的资源说明。摘要正文预留最多 `4096` Token；摘要请求的推理与正文合计输出上限为 `min(16384, maxTokens)`，不强制关闭模型思考。
- 完成摘要替换或移除服务端明确拒绝回放的思考块后，发出 `compaction` 事件，其中 Token 数是估算值。截断、空白或预算不合格的摘要不会被采用；取消或失败时保留已采用的有效上下文。
- 服务端明确报告输入超窗时，进一步压缩并重试一次；明确的思考前缀绑定错误也最多恢复一次。两类恢复可先后发生，均占用 `maxTurns`，只重试主请求，不重复执行工具。普通 `413` 请求体过大和输出参数错误不触发压缩。
- `usage` 包含主模型与摘要请求中已取得完整响应的用量；`turns`、`maxTurns` 只计算主模型请求。

`maxTokens` 默认取 `min(98304, floor(contextWindow / 4))`；显式设置时必须为正整数且小于 `contextWindow`，也应符合模型的输出限制。`maxTurns` 默认 `200`。

压缩时保留当前工具回合的 `thinking` 原块，由服务端判断能否继续回放。只有收到明确的“绑定到不同对话”错误，且不是手动 `enabled` 模式时，才移除思考块并重试；损坏签名、要求原样保留思考的错误不自动修复。模型支持时，宿主也可显式配置 `thinking.block_binding.prefix_mismatch_behavior: "drop_block"`，Agent 会附带对应 beta 标记。

临时纯文本 `system` 可使用 `clear_at: "next_user_message"`，Agent 自动附带相应 beta 标记；后续任意 `user` 消息（包括工具结果）会使其失效。普通请求仍原样回传，压缩时移除失效条目且不将其写进摘要。持久指令应放在 `systemPrompt`；中间的持久 `system` 保持原位，可能限制可压缩范围。最新消息或工具结果本身过大时，需要缩小输入、工具输出或输出预留。

## 工具与验证

内置工具可按需装配：`createReadTool`、`createWriteTool`、`createEditTool`、`createBashTool`；自定义工具使用 `defineTool`。工具在单次运行中串行执行，Bash 工具需要 POSIX 环境。

`AgentToolResult.details` 仅随 `onEvent` 的 `tool_end` 透传，不进入 `messages` 或 `onMessage`；该观察通道不保证可靠保存。需要后续恢复的数据由工具或宿主另行保存。

Agent 主循环、上下文压缩与内置工具始终保留在 `stateless-agent.ts` 单文件中。仓库不提交测试文件。

`npm run example` 运行 `example.ts` 中的真实模型示例，会访问配置的模型服务并执行工具；`npm run typecheck` 仅做类型检查。
