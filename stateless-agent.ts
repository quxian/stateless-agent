/**
 * 无状态的模型—工具调用循环：会话历史、工具装配与生命周期均由宿主管理。
 * 核心流程：准备本轮请求（prepareMessages）→ 请求模型 → 串行执行工具 → 保存结果 → 继续或结束。
 *
 * 运行契约（宿主职责边界）：
 * - 会话历史由调用方传入、由返回值/onMessage 带出；runAgent 不持有跨调用状态。
 * - 工具由宿主显式传入（默认无工具）；文件类能力通过 createWorkspaceTools(workspace) 装配，工作区只属于这组工具。
 *   历史中含 tool_use/tool_result 时，请求必须定义相应工具（API 要求），宿主应保持工具集与历史一致。
 * - 上下文整理（裁剪、摘要、检索注入、图片替换）由宿主通过 prepareMessages 完成，只影响本次请求，不改写完整历史；
 *   约束见 AgentOptions.prepareMessages。
 * - 单次运行内工具严格串行执行。但工作区按“单写入者”设计：多个 agent 实例、并发直接调用工具或外部编辑器
 *   同时修改同一文件时，可能发生“最后提交者覆盖”的丢更新；多写入者场景需要宿主提供文件级协调或版本冲突检测。
 * - 路径检查只用于防误操作，不是安全沙箱（存在 TOCTOU 窗口、硬链接不受约束、bash 可执行任意命令）。
 *   用于不可信代码或多租户环境时，隔离必须由操作系统权限、容器等执行环境承担。
 * - write/edit 以 rename 原子提交（细节见 src/workspace-tools.ts 的 atomicWriteFile）；不解决并发覆盖。
 * - bash 输出超出头尾摘要时，尽力保存输出日志（目录见 createWorkspaceTools 的 bashLogDir）；单个日志最多 64 MiB，
 *   保存失败时不返回路径；保留的日志不会自动删除，由宿主清理。
 * - read 返回的图片会留在会话历史中，之后每轮请求都会重新发送。请求体大小与单次请求图片数量均受 API 限制，
 *   长会话中历史的压缩/裁剪由宿主负责（prepareMessages）。图片处理依赖 sharp，在首次读图时才加载。
 * - 取消：宿主收到 SIGTERM/SIGINT 时应对每次运行的 AbortController 调用 abort() 并等待 runAgent 结束，
 *   以便子进程组完成 SIGTERM→SIGKILL 清理；Bash 模块内的 process "exit" 兜底只做同步 SIGKILL，不能替代完整关闭流程。
 * - 完成与取消同时发生（最终 assistant 已完整返回、保存期间触发取消）时，以完成为准，stopReason 报告模型的停止原因。
 * - 平台：bash 工具仅支持 POSIX（依赖 /bin/bash 与进程组语义），已在 macOS 与 Linux 上做过本地执行测试；
 *   Windows 上 bash 工具直接报错，核心循环与其他工具不受影响。
 */

export { AgentRunError, runAgent } from "./src/agent.ts";
export { HeadTailBuffer, runBashProcessGroup } from "./src/bash.ts";
export { createWorkspaceTools } from "./src/workspace-tools.ts";
export type {
  AgentStopReason,
  ToolOutput,
  AgentToolResult,
  AgentToolProgress,
  AgentToolContext,
  AgentTool,
  AgentEvent,
  AgentUsage,
  PrepareMessagesContext,
  AgentOptions,
  AgentResult,
} from "./src/types.ts";
export type { BashRunOptions, BashRunResult } from "./src/bash.ts";
