# 可执行风险知识

这是算法读取的唯一知识目录，不放文献 PDF 或实验结果。

- `risks/`：风险定义与跨 Agent 约束。
- `templates/`：局部风险子图、工作流与工具能力模板。
- `schemas/`：知识条目的格式约束。

来源和设计依据放在 [docs/references/](../docs/references/README.md) 与 [docs/design/](../docs/design/)。
某次实验的训练、划分或阈值配置放在 `experiments/<主题>/configs/`，不要混成通用知识。
源码直接读取本目录；构建 wheel 时打包同一份 JSON，不维护第二份手写副本。
