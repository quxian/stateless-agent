import Anthropic from "@anthropic-ai/sdk";
import {
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

const result = await runAgent({
  client,
  model: process.env.ANTHROPIC_MODEL ?? "gpt-6-astra",
  tools,
  systemPrompt: `You are an autonomous agent working in ${workspace}. Available tools: ${tools.map((t) => t.name).join(", ")}.`,
  messages: [
    { role: "user", content: "读取当前目录下图片，并分析内容" },
  ],
  onEvent: (event) => {
    if (event.type === "text_delta") process.stdout.write(event.delta);
    if (event.type === "tool_start") console.log(`\n[tool_start] ${event.name}`, event.input);
    if (event.type === "tool_progress" && event.text) process.stdout.write(event.text);
    if (event.type === "tool_end") console.log(`[tool_end] ${event.name} (error=${event.isError})`);
  },
});

console.log("\n\n--- Result ---");
console.log({
  stopReason: result.stopReason,
  turns: result.turns,
  usage: result.usage,
});

