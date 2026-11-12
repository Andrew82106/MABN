# MAS 监测近邻补核：2026-09-20

本页记录原文/作者代码入口，不是完整文献综述或复现成绩。实验和适配边界见 `paperAlpha/research/MAS原生对照接入_当前动作与未来风险分离_20260920.md`。

## QuadSentinel

- 原文：[Sequent Safety for Machine-Checkable Control in Multi-agent Systems](https://arxiv.org/html/2512.16279v1)，重点 §3–4、附录A.5。
- 作者在原文明确链接[代码仓库](https://github.com/yyiliu/QuadSentinel)。本次下载提交 `1d4e29443284385a2505acb1e7986d9d80dc97f1`，17个受版本管理的文件，无许可证文件。
- [Guard入口](https://github.com/yyiliu/QuadSentinel/blob/1d4e29443284385a2505acb1e7986d9d80dc97f1/src/quadsentinel/agent.py)保存谓词、规则、主体威胁级别及缓冲上下文；消息更新状态，工具检查策略，必要时调用两级裁判。
- [原生拦截器](https://github.com/yyiliu/QuadSentinel/blob/1d4e29443284385a2505acb1e7986d9d80dc97f1/src/quadsentinel/utils/intervention.py)先把工具提案作为消息更新状态，再检查动作。只调用`handle_action`不等同原拦截流程。
- [逻辑核](https://github.com/yyiliu/QuadSentinel/blob/1d4e29443284385a2505acb1e7986d9d80dc97f1/src/quadsentinel/agents/verifier.py)对布尔谓词进行规则求值；不是学习未来事件概率。当前公开实现包含字符串替换和`eval`，本项目仅允许审核后的受限布尔策略进入，不把模型自由文本当Python执行。
- [提示词](https://github.com/yyiliu/QuadSentinel/blob/1d4e29443284385a2505acb1e7986d9d80dc97f1/src/quadsentinel/utils/prompts.py)和[各角色代码](https://github.com/yyiliu/QuadSentinel/tree/1d4e29443284385a2505acb1e7986d9d80dc97f1/src/quadsentinel/agents)是实际核对对象。不能把论文中的连续威胁更新公式、必查集合等逐项宣称为该提交已按相同形式实现。

本项目先使用公开支持的全谓词模式`k=-1`，而非借用不存在的完整检索复现。未经许可，不将下载的作者代码重新打包公开；记录入口和提交即可。

## SafeFlow：同名项目要区分

本页指Dai等人的[2607.25255v1](https://arxiv.org/html/2607.25255v1)，不是其他同名交易/安全框架。该方法保留跨Agent来源—去向依赖，结合上下文重建与全工作流策略判断。它直接覆盖我们关注的跨节点信息流，不应从相关工作中漏掉。

核读§3–4及附录A/B，未在原文中确认完整作者实现链接。本轮未实跑；即使重写其确定性传播部分，也只能称部分重实现，不能报告为作者原方法成绩。

## 另外三项复现资格

- [SentinelAgent原文](https://arxiv.org/html/2505.24201v1)：仍未确认完整作者监测器仓库。Phoenix、CrewAI示例不是完整方法。
- [AgentMonitor作者仓库](https://github.com/chanchimin/AgentMonitor)：主要是MAS表现预测/响应后编辑，不直接输出与我们一致的安全风险目标。
- [ShieldAgent原文](https://arxiv.org/html/2503.22738v1)是重要的知识—概率规则近邻；搜索找到的[AutoPolicy](https://github.com/BillChan226/AutoPolicy)README主要描述策略抽取流水线，不足以确认完整ShieldAgent推断、训练与基准复现。THU同名模型也不能自动视为同一方法。

以上“未确认”只说明本次核查范围，不能写成作者永久未开源。原生MAS身份不自动保证任务可比；应先对齐输入、判断时刻和标签。
