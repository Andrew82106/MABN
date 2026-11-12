# 完整 LUMINA 公式基线：最小补充计划

当前仅是源码核对后的计划，未启动 GPU、提取或训练。执行排在 QA 和 R23 之后，须先冻结具体方案。

直接复用 Round7 的 `collect_response_hidden`、`final_statistics`、`ipr_from_hidden`、`cosine_mmd_from_topk`，不另写一套 IPR。固定官方 commit 为 `c43ff41d872b05f659dcb3ad3a6dd78226954319`；这是该代码公式在当前任务上的适配，不能称为原论文整套实验的完全复现。

## 范围与最小增量

仅 R16 **actual split=train** 的 301 题、278 组、602 份固定回答，继续原五折和原标签。拒答及解析失败也提取全部原 token，是否可评由原金标口径决定。禁止以 row ID 内的上游 train/test 字样筛分，也不读取 R16 原 validation/test。

R19 已保存每个原回答 token 的两列单来源 MMD，但未保存所有层 hidden、IPR 或重建 IPR 所需分布，因此 IPR 不能从 ECS/PKS/MMD 倒推，至少需要 **602 次原输入回放**。原回放同时取得最终 top-100 分布；旧两列 MMD 直接复用原缓存，先核所有 token IDs、offsets、源生成 hash 与 R19 签名。

建议再做 **602 次联合正文扰动回放**。每条将 R19 两个互不相交的正文 token mask 同时应用，分别复用各自已冻结 donor 流；标题、问题、模板、答案、原 prefix token 数及答案位置全保持。这只增加一个扰动前向，却能检查多个来源互相替代导致单独扰动不敏感的问题。联合 MMD 不能由两个单来源 MMD 相加推算，Transformer 的响应并非线性。

总计最小推荐为 1,204 次 backbone 前向，加原输入 28 层的分块 lm_head/IPR。两次 selfcheck 若结果能严格复用可计入总数。IPR 的逐层词表运算会成为主要额外成本，先实际计时 2 条短/长训练回答后按总回答 token 数估算时间；不可只按 backbone 次数或旧 H100 速度承诺工期。

## 冻结公式与公平聚合

按 Round7 固定代码，token 风险排序分数为：

`score_t = 0.5 * IPR_t - 0.5 * MMD_t`。

保留全 28 个输出层、层深权重、熵归一化、最终 top-1 概率比截断、实际生成 token 概率校正。尤其保留官方代码的最后层再次 final norm 约定；不能静默换成论文中只求和 1..L-1 的另一公式。MMD 保留 top-100 **未重新归一化**概率和余弦核的质量差项。

建议只预定两个完整公式版本：

1. **单来源均值适配**：`MMD_t = mean(MMD_t,source1, MMD_t,source2)`。来源同权，不根据哪个是目标来源、资料条件、类别、标签或分数选择。原 R19 的 max 汇聚若列出，只作已有结构的次要对照，不能从 mean/max 中事后挑优冒称固定官方公式。
2. **联合正文扰动适配（建议主版本）**：使用两个正文同时扰动后的真实联合 MMD，与同一 IPR 配合。

窗口分数是固定 4 raw BPE 内 token 分数的算术平均；原短窗口规则不变，标点位置也保留。系统整答比较沿当前协议取全部输出窗口最大值，阈值只用对应 calibration 折搜索。原 LUMINA 的整答 token 均值可另报一个提前命名的辅助口径，不能与 answer-max 混称同一指标。无需新 LR、PCA 或调 lambda；旧基线保持原系数/阈值。两个完整公式版本采用相同分组、校准预算与报告分母。

## “全 context” 的准确名称

联合两个**正文** mask 仍保留标题及正文边界 token，因此应明确叫 joint-body perturbation，不能称原始全检索 context 替换。R19 donor 拼接流是等长 token 干预，也不一定构成自然检索文章。

若要增加更接近官方随机 context 的基线，可在下一份冻结方案中增加单独分支：每条用固定、已审独立的两篇自然 donor 文档同时替换两个标题和正文，问题/系统提示保持，原答案 token IDs 不变；允许 prefix 长度变化但按各自 `P+t-1` 对齐同一回答 token。此分支需再加 602 次扰动前向及检索模板/来源审查，并明确长度和位置变化也是干预的一部分。它不是完成 IPR+MMD 两项公式所必需的最小工程步骤，不应把它与等长正文分支合并后挑结果。

## 输出与自检

新目录单独写入，旧 R7/R19 文件只读。建议每答 NPZ 保存：

- `token_ids`: int64[N]；`response_token_offsets`: int32[N,2]。
- `token_ipr`: float32[N]。
- `token_mmd_single`: float32[N,2]，等于已冻结 R19 数值。
- `token_mmd_joint_body`: float32[N]。
- `token_lumina_single_mean`、`token_lumina_joint_body`: float32[N]。

层 hidden 在 CPU 临时存储，完成 IPR 即释放，不落盘大张量。沿旧每行 JSON/NPZ 双哈希、代码/模型/公式/旧特征/原生成/donor/联合计划哈希、`complete` 与 602 计数的 manifest；保存联合改变位置与 prefix IDs 或其完整可复查计划。

自检应覆盖：原 prefix/答案精确回放；所有 N 个 token 包括首 token 在 P-1 对齐；两 mask 无交叉、联合只改 union mask、答案位置不变；identity 扰动 MMD 约为 0；同一输入重复值一致；少量单来源重算与旧 R19 MMD 比对；IPR 分块大小不改变公式（容许记录浮点误差）；全部得分有限。沿现有 NF4/BF16定义，不能为了比对临时改 FP32 路径。长度变化的官方风格分支若做，应另记数值/位置效应。

## 结论边界

该补充能回答“此前是否漏掉 LUMINA 的内部项 IPR，以及联合来源扰动是否比单来源适配合理”。它不能保证提高 F1，也不能把检索敏感性等同于事实支持。仍然是反复开发的助手标注资料支持数据，后续人工 QA 新测试需要在开发冻结后独立检验。
