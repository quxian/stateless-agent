import Anthropic from "@anthropic-ai/sdk";
import {
  AgentRunError,
  type AgentOptions,
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  runAgent,
} from "./stateless-agent.ts";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || 'zys-test',
  baseURL: process.env.ANTHROPIC_BASE_URL || 'http://127.0.0.1:8317',
});

const workspace = process.cwd();
const tools = [
  createReadTool(workspace),
  createWriteTool(workspace),
  createEditTool(workspace),
  createBashTool(workspace),
];

type Message = AgentOptions["messages"][number];

// 原始记录和有效上下文分别由宿主持有；本示例只保存在内存中，没有落盘。
const history: Readonly<Message>[] = [];
let messages: AgentOptions["messages"] = [];

for (const content of [
  "读取当前目录的 README.md，用三个要点概括这个项目的用途和职责边界。",
  "基于你上一轮的三个要点，用一句话向首次接入者说明调用方与 Agent 各自负责什么。",
  "分析 stateless-agent.ts 这个文件",
]) {
  const input: Message = { role: "user", content };
  history.push(input);
  messages = [...messages, input];
  console.log("\n--- User ---\n" + content);

  try {
    const result = await runAgent({
      client,
      model: process.env.ANTHROPIC_MODEL ?? "gemini-4-hydrogen-xhigh",
      contextWindow: Number(process.env.ANTHROPIC_CONTEXT_WINDOW ?? 200_000),
      tools,
      systemPrompt: `You are an autonomous agent working in ${workspace}. Available tools: ${tools.map((t) => t.name).join(", ")}.`,
      messages,
      onMessage: (message) => {
        history.push(message);
      },
      onEvent: (event) => {
        if (event.type === "text_delta") process.stdout.write(event.delta);
        if (event.type === "compaction") {
          console.log(`\n[compaction] estimated tokens: ${event.estimatedTokensBefore} → ${event.estimatedTokensAfter}`);
        }
        if (event.type === "tool_start") console.log(`\n[tool_start] ${event.name}`, event.input);
        if (event.type === "tool_progress" && event.text) process.stdout.write(event.text);
        if (event.type === "tool_end") console.log(`[tool_end] ${event.name} (error=${event.isError})`);
      },
    });
    messages = result.messages;

    console.log("\n\n--- Result ---");
    console.log({
      stopReason: result.stopReason,
      turns: result.turns,
      usage: result.usage,
      recordedMessages: history.length,
      contextMessages: messages.length,
    });

    if (result.stopReason !== "end_turn") {
      // 若宿主决定继续本次未完成运行，应传回 messages，不重复追加本轮用户输入。
      console.log("本次运行未以 end_turn 结束，停止示例，由宿主决定后续是否续跑。");
      break;
    }
  } catch (error) {
    if (error instanceof AgentRunError) messages = error.messages;
    console.error("\n运行失败，停止示例。", {
      recordedMessages: history.length,
      contextMessages: messages.length,
    });
    throw error;
  }
}
