# stateless-agent

无状态的模型与工具调用循环。会话历史、工具装配与生命周期由宿主管理。

## 入口与结构

继续从根入口导入，公开接口保持不变：

```ts
import { runAgent, createWorkspaceTools, AgentRunError } from "./stateless-agent.ts";
import type { AgentOptions, AgentTool } from "./stateless-agent.ts";
```

| 文件 | 职责 |
| --- | --- |
| `stateless-agent.ts` | 运行契约与公开导出 |
| `src/agent.ts` | Agent 主循环、取消处理、观察事件与错误快照 |
| `src/types.ts` | Agent、工具、事件和宿主钩子的类型契约 |
| `src/messages.ts` | 历史回合恢复、请求消息校验与缓存断点 |
| `src/workspace-tools.ts` | 工作区工具装配、路径检查、文本读写与原子编辑 |
| `src/bash.ts` | Bash 进程组、头尾输出缓冲、日志与进度节流 |
| `src/image.ts` | 图片识别、按需加载 sharp、缩放与编码 |
| `src/utils.ts` | 跨模块共用的错误码读取与字符串安全截断 |

内部模块直接引用所属实现；调用方统一使用根入口。

## 验证

需要 Node.js 24 或以上版本。

```sh
npm ci
npm run typecheck
```

`npm test` 运行 `test.ts` 中的真实模型示例，需要先配置可用的模型服务；它不是离线回归测试。
