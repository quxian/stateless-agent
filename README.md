# stateless-agent

纯粹、无状态的模型—工具调用循环。

## 核心设计（一句话心智模型）

`runAgent` 是一个**纯函数式的上下文状态推进器**：
- **状态只有一份 `messages`**：传入 `messages`，返回推进（并在超预算时自动摘要压缩）后的 `result.messages`（异常时为 `AgentRunError.messages`）。内存续跑或落盘持久化，永远只存取这一份数组。
- **观察只有一个 `onEvent`**：流式文本/思考增量、工具进度、新增完整消息（`message`）与压缩摘要（`compaction`）统一从只读非阻塞的 `onEvent` 旁路发出，绝不干扰主控制流。

## 快速上手

需要 Node.js 24+ 及兼容 Anthropic Messages API 的模型服务。

```sh
npm ci
npm run typecheck
npm run example
```

最小接入与持久化示意：

```ts
import Anthropic from "@anthropic-ai/sdk";
import { readFile, writeFile } from "node:fs/promises";
import { type AgentMessage, AgentRunError, createBashTool, createReadTool, runAgent } from "./stateless-agent.ts";

const client = new Anthropic();
const tools = [createReadTool(process.cwd()), createBashTool(process.cwd())];

let messages: AgentMessage[] = JSON.parse(await readFile("session.json", "utf8").catch(() => "[]"));
messages = [...messages, { role: "user", content: "检查当前目录结构并说明主要文件作用。" }];

try {
  const result = await runAgent({
    client,
    model: "claude-sonnet-4-5",
    tools,
    messages,
    onEvent: (e) => {
      if (e.type === "text_delta") process.stdout.write(e.delta);
    },
  });
  messages = result.messages;
} catch (err) {
  if (err instanceof AgentRunError) messages = err.messages;
  throw err;
} finally {
  await writeFile("session.json", JSON.stringify(messages, null, 2));
}
```

> 如果你还需要额外保留一份从不裁剪的旁路审计日志（如 `audit.jsonl`），只需在 `onEvent` 中监听 `event.type === "message" || event.type === "compaction"` 追加写入即可，与 `runAgent` 状态互不耦合。

## 每轮模型调用前携带新数据（`getTurnContext`）

当 Agent 正在执行多轮工具循环时，若用户发来中途补充/纠偏指令（Steering）、后台任务完成产生异步通知，或宿主需要注入环境增量信息，可传入 `getTurnContext`：

```ts
const steeringQueue: string[] = [];

const result = await runAgent({
  client,
  model,
  tools,
  messages,
  getTurnContext: ({ turn }) => steeringQueue.shift(),
});
```

- **调用时机**：每次发起主模型请求前（`turn = 0, 1, 2...`）且在上下文压缩检查之前调用。
- **零丢消息保证**：返回非空字符串或 `AgentInputBlock[]`（`text` / `image` / `document`）时，引擎会**立即同步追加进 `messages`**（并发出 `onEvent` 的 `message` 事件），随后即使触发 `AbortSignal` 取消，该消息也已安全保存在返回的 `result.messages` 中，因此队列可直接放心使用 `shift()` / `splice()` 消费。
- **缓存与协议友好**：新增内容以 `user` 消息追加在末尾（不改写 `systemPrompt`，不破坏前缀 KV 缓存与 `thinking` 签名绑定），发往模型时自动将相邻同角色消息合并为严格交替角色。

## 结束原因与续跑

| `stopReason` | 含义与接续方式 |
| --- | --- |
| `end_turn` | 模型正常结束本轮回复；下次提问时追加新的 `{ role: "user", content }` 后传入 |
| `max_turns` | 达到主请求轮次上限（默认 `100`）；如需继续当前任务，直接原样传回 `messages`（不要重复追加用户输入） |
| `aborted` | 被 `AbortSignal` 取消；未配对的工具调用已自动补齐，可直接原样传回 `messages` 续跑或追加新指令 |
| `max_tokens` / 其他 | 达到单次输出上限或特定停止条件；按业务需要调整参数或继续对话 |

若上次进程意外崩溃导致历史中残留未返回结果的 `tool_use`，下次 `runAgent` 启动时会在入口自动补齐“执行状态未知”的错误结果块，保证任意历史快照均可直接恢复运行。

## 自动上下文压缩

1. **触发时机**：每轮主模型请求前估算当前上下文开销（首轮按字节启发式估算，同次运行后续轮次自动锚定上一轮 API 真实 `usage` 用量）；超出窗口安全预算（`contextWindow - maxTokens - safety`）时自动触发压缩。若服务端仍返回明确的输入超窗错误，会强制压缩并原地重试一次。
2. **安全切分与单次摘要**：严格在合法回合边界切分，绝不拆散 `tool_use` 与对应的 `tool_result`，保留最新回合与约 40% 窗口余量，将较早历史通过单次流式请求浓缩为一条摘要消息，并通过 `onEvent` 发出 `{ type: "compaction", summary, estimatedTokensBefore, estimatedTokensAfter }`。
3. **思考块兼容**：压缩时自动剥离旧回合的 `thinking` 块；若压缩改写前缀后服务端拒绝回放当前回合的 `thinking` 签名（非强制手动 `enabled` 模式），会自动剥离思考块并重试一次。
4. **自动缓存断点**：当工具与消息均未手动设置 `cache_control` 时，自动在 `system`、末尾工具与末尾消息块上添加 `ephemeral` 缓存断点。

## 内置工具与自定义扩展

单文件 [stateless-agent.ts](file:///Users/zys/Desktop/me/agent/stateless-agent.ts) 内置四个工作区工具（含真实路径越界及悬空符号链接防护，单次运行内串行执行）：
- `createReadTool(workspace)`：UTF-8 文本按行分页读取（上限 5 MiB），以及 JPEG/PNG/GIF/WebP 图片读取（依赖 `sharp` 自动转正并缩放至 2000px 内）。
- `createWriteTool(workspace)`：自动创建父目录，同目录临时文件 + `rename` 原子写入。
- `createEditTool(workspace)`：单处精确唯一匹配替换（`{ file_path, old_text, new_text }`），兼容 CRLF 换行并保留 BOM，原子写入。
- `createBashTool(workspace, options?)`：仅支持 POSIX，独立进程组执行，含 `SIGTERM → SIGKILL` 升级终止、管道排空保护、前 16 KiB + 后 32 KiB UTF-8 安全截断及节流进度上报。

自定义工具可通过 `defineTool({ name, description, inputSchema, run })` 使用 Zod 对象 Schema 强类型定义。仓库保持单文件实现，不提交测试文件。
