import Anthropic from "@anthropic-ai/sdk";
import {
  type AgentMessage,
  AgentRunError,
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  runAgent,
} from "./stateless-agent.ts";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || "zys-test",
  baseURL: process.env.ANTHROPIC_BASE_URL || "http://127.0.0.1:8317",
});

const workspace = process.cwd();
const tools = [
  createReadTool(workspace),
  createWriteTool(workspace),
  createEditTool(workspace),
  createBashTool(workspace),
];

// 单一状态真相：无论是否发生过压缩，续跑或落盘永远只读写这一份 messages。
let messages: AgentMessage[] = [];

for (const content of [
  "读取当前目录的 README.md，用三个要点概括这个项目的用途和职责边界。",
  "基于你上一轮的三个要点，用一句话向首次接入者说明调用方与 Agent 各自负责什么。",
  "分析 stateless-agent.ts 这个文件",
]) {
  messages = [...messages, { role: "user", content }];
  console.log("\n--- User ---\n" + content);

  try {
    const result = await runAgent({
      client,
      model: process.env.ANTHROPIC_MODEL ?? "gemini-4-hydrogen-xhigh",
      contextWindow: process.env.ANTHROPIC_CONTEXT_WINDOW
        ? Number(process.env.ANTHROPIC_CONTEXT_WINDOW)
        : undefined,
      tools,
      systemPrompt: `You are an autonomous agent working in ${workspace}. Available tools: ${tools.map((t) => t.name).join(", ")}.`,
      messages,
      onEvent: (event) => {
        if (event.type === "text_delta") process.stdout.write(event.delta);
        if (event.type === "compaction") {
          console.log(`\n[compaction] ${event.estimatedTokensBefore} → ${event.estimatedTokensAfter} tokens`);
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
      contextMessages: messages.length,
    });

    if (result.stopReason !== "end_turn") {
      // 若宿主决定继续未完成的任务，直接原样传回 messages 即可，不要重复追加本轮用户输入。
      console.log("本次运行未以 end_turn 结束，停止示例。");
      break;
    }
  } catch (error) {
    if (error instanceof AgentRunError) messages = error.messages;
    console.error("\n运行失败，已保留最新上下文快照。", { contextMessages: messages.length });
    throw error;
  }
}
