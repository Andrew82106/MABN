# Round9 训练与评测草案

状态：实现前协议；不包含真实训练或测试成绩。根确认数据、特征名和冻结清单后写入正式 protocol。Round7/8 文件保持原样。

## 1. 要检验的问题

在联网资料汇总的模拟场景中，判断回答的哪些词元缺乏当前资料支持。分别检验：把整项标签改成错误位置标签是否有效；把整资料注意力细分到问题所问主体与属性的候选资料是否有效。候选资料由可见问题与来源推导，可能匹配错误，不能当标准证据。

计划约 200 个全新题组，每组 complete/partial 两条件，120/40/40 组训练/验证/测试，五类尽量均衡。最终数量按数据可行性预先冻结。关联主体、原文、事实及同题两条件不跨划分；新留出题避开 Round7/8 已见题。每份回答只有一个预定回答项。提示词正常提问，不加入“资料不足”回答暗示。

## 2. 严格 2×2 与固定结果数量

四组核心方法全部使用同一批可评词元、同一当前位置特征、相同训练样本权重规则。粗监督把该回答项的 risk 广播给其词元；细监督只将标注风险表达覆盖到的词元标 1。没有把句末位置训练与词元位置训练混在一起。

| 编号 | 方法 | 训练标签 | 输入 |
|---|---|---|---|
| 1 | lb_coarse | 整项 risk 广播 | 全局 Lookback，784 维 |
| 2 | lb_fine | 风险位置 | 同上 |
| 3 | binding_coarse | 整项 risk 广播 | 同一全局 Lookback + binding |
| 4 | binding_fine | 风险位置 | 同上 |
| 5 | hidden_lr | 风险位置 | 第 28 层 hidden，3584 维 |
| 6–8 | hidden_mlp_seed_* | 风险位置 | 同一第 28 层，三个预定种子 |
| 9–12 | nll、entropy、redeep、lumina | 对应固定信号，ReDeEP 在训练集确定排序及缩放 | 当前词元信号 |
| 13–14 | all_positive、all_negative | 无 | 常数 |
| 15–16 | lb_coarse_broadcast、binding_coarse_broadcast | 不另训练 | 每项全部词元风险分数的算术均值，广播回该项 |

共 12 个方法家族、14 个预测器，加 2 个整项广播对照，共 16 条预测结果。MLP 另报三种子的指标均值和样本标准差，不新增概率集成方法，不挑最好种子。广播对照使用已完成整项的信息，是事后对照，不是当前词元因果定位。

这里的“粗监督”仅指训练损失。粗、细两行都使用细标验证集选参数和定位阈值，因此不宣称粗监督整条流程不需要细粒度标注。旧 Round8 句末分类头只作历史诊断，不混入新主表。

## 3. 权重与有限搜索

基础权重按题组→条件→回答项→词元逐级等分，使长回答不因词元多而压过短回答，每题组基础总权重相同。StandardScaler 仅在训练特征上用基础权重拟合；粗、细同特征版本共享同一 scaler，不能按标签重做输入缩放。

训练损失正负系数仅由训练集基础加权类别总量计算。基础权重乘类别系数后，再逐组归一，使每题组最终总权重相同、全体样本权重均值为 1。此处理不保证最终类别总质量正好各半，应保存实际类别质量。LR 使用 sample_weight，不能再叠加 balanced class_weight。

- 五个 LR 家族各试 C={0.1,1}，共 10 次拟合。L2/liblinear/max_iter=2000，固定随机种子 20260910。
- MLP 固定 3584→128→64→1、ReLU、dropout=0.1、AdamW/weight_decay=0.01；仅试 lr={1e-4,3e-4}，种子 20260910/11/12，共 6 次。最多 60 epoch，patience=8；建议固定 batch=128。提前停止按验证 BCE，使用训练导出的类别系数及验证基础组权重，不由验证类别数量重估权重；保存最小 BCE checkpoint。每个种子各自校准阈值，以三种子验证 F1 均值选共同学习率，并列选平均 precision、更小学习率。
- ReDeEP 只试 heads={1,4} × layers={4,8} × beta={0.1,0.5,1}，共 12 组。ECS 与 1-y、PKS 与 y 的加权 Pearson 排序只读训练；选定信号求和的 MinMax 也只在训练拟合。风险方向固定 scaledPKS−beta×scaledECS，不按测试结果翻转。
- LUMINA 的 lambda=0.5，沿既有公式；NLL/entropy 原方向。无额外层、结构、种子搜索。

参数和阈值仅按验证集**无权词元 micro F1**选择，必须保留正常回答词元。阈值并列选 precision，再选更高阈值，包含全报和全不报端点。LR 参数并列选较小 C。每种方法一个全局阈值，不分属性、条件或是否风险项调阈值。两个广播对照分别用同一细标验证集校准阈值。全正/全负固定 0.5。

## 4. 最小数据接口与禁止信息

- `data/inputs.jsonl`：row_id/question_id/group_id/split/condition/questions[1]/passages[{title,text}]/prompt/system/dataset/expected_items=1/category。
- `data/generated.jsonl` 和 `generation_records/{row_id}.json`：原始生成文本、精确 input/response token IDs、response_token_offsets、预定 items 的 text/start/end/parse_ok。
- `data/features/{row_id}.npz`：hidden_28[N,3584]、lookback_features[N,784]、binding_features[N,D]、token_nll[N]、token_entropy[N]、token_ids/token_start/token_end。JSON 侧车至少有 source_generation_sha256、arrays_sha256、signatures、binding_feature_names。
- `data/attention/{row_id}.npz`：token_redeep_ecs[N,784]、token_redeep_pks[N,28]；`data/lumina/{row_id}.npz`：token_lumina_score[N]。需要侧车或冻结清单绑定原生成和数组哈希。
- `data/annotations_{train,validation,test}.jsonl`：每个预定 item 一条，精确文本/span、source_generation_sha256、original_risk、original_stance、localization_status、claim_scope、risk_spans，与 Round8 对齐。

候选解析只接收白名单可见 question/passages（及重建展示所需 prompt/system），禁止 row.subjects、参考答案、隐藏 supporting facts、condition、category、标签或未来回答。category/condition 只用于报告。空候选是合法失败情况，保留数值占位及 validity，不删除困难词元。D、names、缺省含义和候选规则版本必须在正式拟合前冻结。binding 中存在性或长度线索可能解释成绩，未做匹配变量独立消融前不把全部增益解释为注意力机制创新；本次不擅自扩大已确认 16 方法预算。

## 5. 标签、时点和严格计数

标签是当前资料支持关系，unsupported/contradicted 与现实事实错误分开。助手盲标并复核，不能称专家人工金标。明确拒答、未决项不进入主定位分母，但保留文字报警描述。粗、细训练均只取同一批 resolved asserted 项及其词元，不能用测试覆盖情况挑样本。

沿 Round8 精确原始 BPE 偏移计数：含 Unicode 字母或数字的词元可计，数字被拆开的词元保留，纯标点/空白/编号排除；功能词保留。词元与风险 span 的字母数字字符相交即为正，不做分数平移或容忍窗口。正常项内词元和风险项内非风险词元都要保留。非 span 标 0 只表示不是圈定风险表达，不声称每个词都经事实验证。

hidden/Lookback/ReDeEP 采用读入当前词元后的原时点；NLL/entropy/LUMINA 保持原预测时点。原生分数不可见未来文字。整项均值广播可见整项后文，明确另列。

## 6. 主结果、避免全句标红和冻结

主表报告所有 resolved 项的 token micro P/R/F1、TP/FP/FN/TN、AUROC/AP、缺失预测及覆盖；另报 risk_items_only 相同指标，并在每条同时含正负词元的回答内部单独算 AUROC/AP 后取宏均值，防止风险项之间的排序掩盖句内不区分。还报逐回答宏均值、风险 span 命中/词元和字符覆盖、额外报警覆盖、正常回答任何误报率。宏 F1 只对存在风险词元的回答定义，正常回答误报单列；不使用 accuracy 作高分依据。来源类别及 complete/partial 分层采用同一阈值。

组 bootstrap 2000 次，种子 20260911；同题两条件和全部词元整体抽样。预定对照是两个监督差、两个特征差和差中之差；另报原生粗监督与自身广播之差。主要特征增益看 binding_fine−lb_fine，同时公开完整四格，不挑测试最好格子。置信区间条件于固定分类器和助手标签，不覆盖重新采样训练或标注误差。

`fit` 只解析 train/validation 标签；提前核对 test 标签文件哈希不等于解析测试标签。正式参数、模型权重、scaler、权重规则、候选词典、全部输入/生成/特征/三套标签哈希及 PLAN/guide/protocol 在 test 前冻结。`test` 验证冻结后一次生成 16 方法结果、逐词元预测、误报案例，不再改阈值。未齐备不自动执行真实 fit/test。

计时分别记录数据生成、候选构建、信号提取、全部 CPU 候选训练、最终纯评分、文件加载；CPU 至多 4 线程。纯评分耗时不能冒充联网端到端延迟。复查预算和复查策略保持原样，本轮只研究定位。
