# ReDeEP(Token) 正式基线迁移审计

## 当前结论

ReDeEP **原生就有逐 token 幻觉分数**。论文第 4.1 节把 `H_t(t)` 定义为 token 分数，再用 token 均值得到整答分数；因此公共文档中“ReDeEP 只有整答分数、需要向窗口广播”的说法应更正。正式迁移应保留 `H_t(t)`，再用固定字符重叠把它映射为本项目 4-BPE 窗口分数。

本目录已经冻结两种身份，不能按结果择优：

- **A：论文明确公式迁移，正式主结果。** ECS 只看检索资料；PKS 使用标准 Jensen-Shannon divergence；具体头/层在共享 fit 的 4-BPE 标签上按 Pearson 相关性排序。Khead=1、Klayer=10、α=1、β=0.2 固定为论文的 RAGTruth Llama2-7B Token 配置，不重搜。
- **B：官方源码公式诊断。** ECS 看整个聊天前缀；PKS 保留源码实际计算的反向 KL 混合量；具体头/层按源码实际返回的 AUROC 排序。K/α/β与 A 完全相同。B 只解释论文和代码差异，不替代 A。

统一 calibration 结果如下。主结果是 A；B 不能按结果择优替换 A。

| 身份 | 4-BPE 窗 F1 | 窗 AUROC | 窗 AP | 整答 F1 | 整答 AUROC | 整答 AP |
|---|---:|---:|---:|---:|---:|---:|
| A：论文公式 | 0.3296 | 0.6693 | 0.2575 | 0.7724 | 0.5907 | 0.7326 |
| B：官方源码诊断 | 0.3325 | 0.6730 | 0.2394 | 0.7722 | 0.6207 | 0.7612 |

整答 F1 不能单独解释为“很准”：159 答中有 100 个正例，全部预测为正的 F1 已是 0.7722。A 只比它高 0.00016，且整答 AUROC 为 0.5907。ReDeEP 在本场景有一定排序信号，但局部定位较弱；更能反映定位能力的是窗口 AUROC 0.6693、AP 0.2575 和 F1 0.3296。以上阈值仍在反复使用的 calibration 上选择并评测，未触碰的独立 holdout 结果保持 N/A。

fit 冻结出的 A 具体头为 `(18,9)`，10 层依次为 `20,18,30,21,17,22,31,16,29,24`；B 为头 `(16,1)`，层 `22,17,29,23,21,16,20,30,18,24`。A 的 calibration 窗口/整答阈值为 0.135078/0.241463；B 为 -0.025730/0.010220。完整相关性、AUROC 排名和 fit MinMax 边界见 `SCORE_FREEZE.json`。

论文所称完整 greedy 搜索仍为 **N/A**：论文没有给搜索顺序、初值、优化目标和并列规则；官方脚本没有执行搜索，只写死最终参数，而且 `token_level_reg.py:223` 先切掉一半列再请求 PKS 与标签列，当前代码会报 `KeyError`。自行补一种搜索规则会改变基线。

## 方法层保持不变

- 骨干是完整、未量化 FP16 Llama-2-7B-Chat；候选 Copying Heads 固定为作者 `topk_heads.json` 中的 32 个头；PKS 使用全部 32 层再排序。
- ECS 对每个候选头取允许区域内注意力最高的 `floor(10%)` token，均值其最终层隐藏状态，再与当前 predictor 的最终层隐藏状态做余弦相似度。
- PKS 对每层 FFN 前后的残差分别经过最终 RMSNorm 和 LM head；最后一层 after 状态已经过最终 RMSNorm，不重复归一化。
- fit 上分别对选中 ECS 总和和 PKS 总和做 MinMax；这些边界原样用于 calibration。token 风险为 `normalized(PKS)-0.2*normalized(ECS)`。
- 作者原生整答复核仍报告 token 均值的 AUROC/Pearson，不能替代统一主比较。
- ReDeEP(Token) 没有另训分类器或损失函数；fit 阶段只做作者规定的头/层排序和 MinMax 拟合。这一训练结构未改。

GPU 实现不保存整张 attention matrix。它在作者固定的 32 个候选头处复算同一 Q/K 排名，只保存 top-10% 的位置；这不会添加新参数或新特征。CPU 小模型把该实现与 dense attention 逐项比较：两种 ECS 最大误差均为 0，两种 PKS 最大误差均为 0，字符映射误差为 0。详见 `CPU_SELFTEST.json`。

模型层保持不变，接口层只做五件事：换成本项目共享样本与 group split；用共享 fit 标签执行原生头/层选择；把原生 token 按字符重叠取均值映射到 4-BPE 窗；按统一规则取 `answer=max(window)`；按统一阈值和 F1/AUROC/AP 评测。映射无参数、标签无关，模型原生输出仍完整保留。

## 本项目接口适配

- 输入固定为 3,680 个 fit 回答和 159 个 calibration 回答，材料组数分别为 615/154，组间零重叠。全部来自 RAGTruth 原 train；提取器拒绝其他 split。
- 模型打分位置 `i` 对应它将要生成的目标 token `i+1`。原生 token 通过回答字符区间与固定 4-BPE 窗口相交；窗口分数取所有相交 ReDeEP token 的算术均值。该映射无参数、不读标签。
- 整答分数按本 benchmark 统一取 `max(window)`。窗口与整答各自在 calibration 上按 `F1 → precision → 更高阈值` 选阈值，并报告 F1、AUROC、AP。
- 作者模板先生成 `<s>[INST]...[/INST]`，再以默认 `add_special_tokens=True` 重新编码，因此 3,839 条输入全部是双 BOS。回答 17592 的开头引号与 prompt 末 token 合并；按作者的独立 prefix 长度切法，该 1 个字符没有原生目标分数。本实现披露该边界，不伪造分数。

无标签准备产物共 3,839 答、2,487,708 个完整前向 token、710,970 个原生回答目标 token。共享合格窗口为 fit 653,979、calibration 42,241；另有纯标点等排除窗 692/80。精确 join 和文件哈希见 `DATA_JOIN_AUDIT.json`。

## 代码与论文不一致

1. 论文 ECS 只看检索资料；代码看 BOS、系统提示、问题、检索资料和模板组成的整个 prefix。
2. 论文 PKS 是标准 JSD；代码执行 `KL(M||P)+KL(M||Q)`，按词表求 mean 后乘 `1e6`。
3. 论文写按相关性排序；代码虽然计算 Pearson，却返回并排序 AUROC。
4. 代码只处理名为 `test` 的行，并在同一 dataframe 上重拟合 MinMax，未实现训练到独立评测的传递。

所以 A 和 B 必须分名报告。共同 benchmark 只改变输入、fit/cal 接线和最终输出映射，不改变各自的特征和公式。细节与源码位置见 `METHOD_FREEZE.json`、`SOURCE_AUDIT.json`。

## 测试封存处置

作者仓库附带的 test JSON 曾在缓存连接审计中被本地进程整体反序列化。没有使用其中 QA 标签、回答、分数或统计做任何选择，但旧 official QA test 已不能继续称为严格“未打开”。它已退出最终评测，后续必须改用新建且未触碰的独立 holdout。完整记录见 `INCIDENT_OFFICIAL_TEST_SEAL.md`。

因此 `BASELINE_PROTOCOL.md`、`CURRENT_STATUS.md` 中“旧 official QA test 仍封存”的说法需要更正；`FORMAL_BASELINE_RESULTS.md` 中“ReDeEP token 信号不是定位输出”的说法也需要更正。本目录先保留独立审计产物，未并发修改公共正式文件。

## 执行状态

全量抽取与 calibration 评分已完成：3,839 答、710,970 个原生目标 token、696,220 个合格 4-BPE 窗，所有原始特征有限。正式运行正常退出，完整 FP16 抽取约 2.39 小时，原始特征 0.256 GiB。

身份链已经补齐：模型 shard、tokenizer、官方仓库提交与 clean 状态、runner/core/scorer、固定 32 头均由哈希绑定。另用 fit 最短、fit 中位、fit 最长和 calibration 中位共 4 条回答重新做完整 FP16 抽取；paper/code 的 ECS/PKS 四组数组与原文件逐元素完全相等。全量原始文件清单、fit MinMax、全部 42,241 个 calibration 窗映射、整答 max 和指标也由另一份不导入 scorer 的脚本复算通过。

关键证据：`METHOD_FREEZE_PRE_EXTRACTION.json`、`RUN_IDENTITY_EVIDENCE.json`、`EXTRACTION_IDENTITY_VERIFICATION.json`、`RAW_FEATURE_COMPLETE.json`、`SCORE_FREEZE.json`、`CAL_RESULTS.json`、`SCORING_INDEPENDENT_VERIFICATION.json`。完整 greedy 复现和独立 holdout 仍为 N/A，不能把当前 calibration 数值写成最终泛化结论。
