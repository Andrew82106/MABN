# ReDeEP 正式基线可复现性协议

**结论：当前公开实现不足以直接完成可信的整套复现；现有本地 ReDeEP 适配也不能冒充正式基线。** 论文定义及主要抽取代码可核实，但公开评分路径存在截列错误、分区接线与论文不一致，且没有完整提供选择过程的身份记录。本轮只下载官方源码、核 SHA-256/语法并写协议，没有模型调用、训练、特征提取或测试数据读取。

来源：[ICLR 2025 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7daf60e805e596c3bd1e843e72ea5560-Abstract-Conference.html)；[官方仓库](https://github.com/Jeryi-Sun/ReDEeP-ICLR)。锁定提交 `4d081915b8fb4430fda65c411da61540cc73cc57`，核查时仍为 main，提交日期 2025-06-03。14 个源文件/配置/论文已保存于 `source_files/`，逐字节哈希见 `source_checks.json`；全部源码只作 AST 解析，没有 import 执行。包含 `test_*` 的已读文件仅为头号与参数 JSON，没有下载仓库中的回答、测试分数或数据集内容。

## 1. 模型、输入与读取时点

| 项目 | 论文与锁定实现 |
| --- | --- |
| 生成/重放模型 | Llama-2-7B-Chat、Llama-2-13B-Chat、Meta-Llama-3-8B-Instruct；未提供精确权重 revision/hash |
| 7B 几何 | 32 层、每层 32 头、隐藏维数 4096、词表 32000 |
| 模型精度 | `torch_dtype=float16`，`device_map=auto`；使用作者修改的 Transformers；没有 NF4 |
| 输入 | 原 source prompt，加系统消息“You are a helpful assistant.”和模型 chat template，再拼已发布回答，teacher forcing 重放 |
| 截断 | 脚本先取 `prompt[:12000]` 字符；并非保证全资料保留 |
| Token 读取时点 | 若单独分词前缀长为 P、完整输入长为 T，循环 `seq_i=P-1,…,T-2`；用该位置的 attention、最终状态及 FFN 分布给下一位置预测时刻计信号 |
| 候选来源范围 | 代码 `start_p/end_p=None`，实际在 **整个前缀 [0,P)** 内取 top 10%，包括问题、系统与模板；论文式子则写检索 context 范围 |
| 回答轴 | 代码单独分词 prefix 再定边界；不处理边界跨 token，且 token 金标使用同 seq_i 与闭区间比较，存在时点/标签错位风险 |

见 [token 抽取](https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/4d081915b8fb4430fda65c411da61540cc73cc57/ReDeEP/token_level_detect.py)。循环包含每个下一词预测时点，不能简单说“漏掉末词”；它**没有读取末词读入后的状态**。若迁移到我们的原生 BPE，必须显式把 predictor P+j−1 对应到答案 token P+j，保留全部原坐标，不混用本地 post-read 状态。这属于必要接口说明，不能悄悄改变信号时刻。

原实验主要让指定模型检测它对应的回答；我们的 added3046 含其他生成器，以同一 Llama2 重放它们是额外跨生成器适配，须另列。NF4 重建也不是当初生成时的原始轨迹。

## 2. 两类内部信号及代码/论文差异

**ECS（外部资料利用）**：每个候选头在前缀/资料内按 attention 排序，选 `floor(0.1×长度)` 个位置；平均这些位置的**最终层状态**，与当前 predictor 位置的最终状态作 cosine。作者代码没有“至少一个”的下限；短输入不足 10 个候选时必须报告其定义问题，不能静默加新规则。每个 token、每个候选头各一个值。已有 7B `topk_heads.json` 提供 32 个候选头，抽取按层号/头号遍历排序，并非输出全部 1024 头。最终参数选中 **(layer25,head0)**，均为 0-based。

**PKS（参数知识利用）**：取每层 attention 残差加和后、进入 FFN 前的 residual，及 FFN 残差加和后的状态；分别过最终 RMSNorm、同一 lm_head，再比较完整词表分布。末层输出已经过最终 Norm，作者完整层遍历分支避免再次归一化；不能拿 FFN 输出张量本身代替“前后残差”。[自定义模型](https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/4d081915b8fb4430fda65c411da61540cc73cc57/transformers/src/transformers/models/llama/modeling_llama.py#L1321)

论文 Eq.5 是标准 JSD：
`0.5 KL(P||M)+0.5 KL(Q||M), M=(P+Q)/2`。
作者 token 代码实际为：
`10^6/V × [0.5 KL(M||P)+0.5 KL(M||Q)]`，
来自 `F.kl_div(logP,M).mean(-1)`，再乘 `10e5`。它不是标准 JSD；作者源码注释明确知道该问题仍保留原实现，并链接 [Issue #2](https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2)。**忠实代码版必须保留方向、词表分母和倍率；标准 JSD 只能另名为论文公式重实现。** 现 R7/R19 的标准 JSD、Qwen 与层段均值均为本地适配。

**Chunk 版另有额外模型及不同结构**：论文用 RecursiveCharacterTextSplitter，chunk_size=256、overlap=20；由头的 chunk-pair attention 选择一个资料块，再用 `BAAI/bge-base-en-v1.5` 的归一化嵌入计算回答块与资料块 cosine。PKS 先对块内词元平均，再平均各块。官方实际代码却：

- 读取预存 `prompt_spans/response_spans`，仓库没有生成这些 spans 的脚本。
- 选资料块用 attention **总和**，论文为均值；不同块长时会改变选块。
- PKS 用反向 KL 的词表求和，再对块内 token **求和**，论文为块内平均；之后整答各块等权平均。
- 用模板局部重分词和固定 `-4` 计算 prompt span 坐标。
- Llama3 chunk 只抽层 0–15，但 reg 期待 32 列；13B 从第 8 层起抽，部分列名/偏移也不一致。

见 [chunk 抽取](https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/4d081915b8fb4430fda65c411da61540cc73cc57/ReDeEP/chunk_level_detect.py)。因此 Token 与 Chunk 不能互相替代；BGE 也不能换成现有 MiniCheck 或 ModernBERT 后称原基线。

## 3. 训练器、参数选择、归一化与整答评分

**论文正式实验没有训练 MLP/LR 网络。** §4.1 为选中 PKS 之和减加权 ECS 之和；附录 J 明确采用网格搜索：先在 validation 根据关联排序，再搜索头数/层数 K∈[1,32]，α=1，β∈(0,2)、步长 .1。论文写 Pearson 关联排名；代码最终按 **AUROC** 排序，计算的 Pearson 值没有用于选择。函数 `linear_regression` 虽定义 LogisticRegression(max_iter=10000)，但主路径**从未调用**，不能当作作者正式分类器。[token reg](https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/4d081915b8fb4430fda65c411da61540cc73cc57/ReDeEP/token_level_reg.py)

代码的评分链为：

`E_t=sum(选中 ECS); P_t=sum(选中 PKS)`

`z_t=MinMax(P_t)−β×MinMax(E_t)`

`z_answer=mean_t(z_t); score_answer=MinMax(z_answer)`。

MinMax 在传入整个 dataframe 上重新拟合；最终整答又以同批回答的 min/max 归一化，不是概率，也不是 answer-max。保存配置有相应 extrema，但公开 reg 不读取它们作独立新数据推断。常量最终分数的零分母也没有处理。不得擅自加 scaler、逻辑回归、窗口 max、后验平滑或新阈值。

论文 RAGTruth 参数及阈值：

| 模型 | Token：头数/层数/β/阈值 | Chunk：头数/层数/β/阈值 |
| --- | --- | --- |
| Llama2-7B | 1 / 10 / .2 / .6 | 3 / 4 / .6 / .3 |
| Llama2-13B | 2 / 17 / .6 / .6 | 9 / 3 / 1.8 / .6 |
| Llama3-8B | 3 / 30 / .4 / .4 | 2 / 5 / 1.2 / .4 |

7B token 参数文件给出层 **[23,24,21,25,16,17,28,18,20,29]**、头 **[[25,0]]**、β=.2；完整三个模型参数与 min/max 已归档。文件没有选择用例/划分清单，不能据路径或参数值证明与我们 cal159、official_test150 无重叠。不得直接继承这些监督选择结果并声称具有我们的材料组隔离。

主表指标是**整答** AUC/PCC/Accuracy/Recall/F1；Token 是内部计算粒度，不等于主表做了 token/span 定位验证。公开 reg 最后只输出 AUC/PCC，没有 Table5 阈值下分类计数的完整入口，也没公开阈值搜索目标/同分规则。若以后完整方法身份先冻结并运行，统一任务只允许把冻结整答连续分数原值广播到该回答全部合格 4-BPE 窗口，再按共同规则取窗口 max 得到整答分数；这不会改变原整答值。内部 token 信号不得包装成正式定位分数。映射无参数、无标签且确定性，统一 F1/AUROC/AP 在映射冻结后计算；作者指标另列。当前窗口与整答都保持 N/A。

## 4. 可复现性阻断：不可照原脚本直接运行

1. detect 和 construct_dataframe 都固定筛选 `split=="test"`；reg 随后对这些行的标签排名、MinMax 并评分。论文写 validation 选参。**公开代码不是一个隔离 validation→test 的实现**；这只是源码事实，不据此断言作者实际实验泄漏。
2. 两个 reg 都把 `df.iloc[:, :int(df.shape[1]*0.5)]` 送进需要全部 PKS/标签列的函数。它截的是列，标签等列已被删除，按当前文件布局会触发 KeyError。不能猜成“前半行”并悄悄修正。
3. 论文采用标准 JSD/context-only/相关性排名，代码是反向 KL/整前缀/AUROC；chunk 还存在 sum/mean 差异。两种口径必须单独固定。
4. 候选头有公开列表，但生成头列表的完整选择脚本、训练/validation 身份清单及 chunk 分段生成器没有发布在锁定树中；参数文件也没有 provenance。权重和 BGE 未锁 revision。除7B外还存在层列数/偏移接线错误。

这些事实已足够判断：**可重建信号与打分结构；不能声称只换数据文件即可完整复现官方实验。** 最小恢复需核实选参和归一化分区、修复明确的列/轴接口错误并保留记录、补齐分段与参数来源。若仅用公开参数做固定迁移，应明确为“作者参数迁移”，不能宣称分区独立或原结果重现。若我们自行改成 fit-only 排名、标准 JSD、缩小网格、NF4 或新分类器，则名称必须包含“本地适配”。

## 5. 现有 Llama2 缓存复用盘点

| 现成数据 | 可复用范围 | 不能替代的部分 |
| --- | --- | --- |
| 原793与新增3046输入计划、raw BPE、字符坐标、材料组身份 | 数据接口和共同评测轴 | 作者模板/12k字符截断与当前输入并非已证明相同 |
| `lb/lb_prefix_pre_header` 等逐头比例 | 边界/时点对照资料 | 只有聚合比例，无法恢复前缀内 top-10% 具体位置或 chunk-pair attention |
| `hidden_last[N,4096]` | 同 NF4 输入的回答 post-read 状态检查；非作者 FP16数值 | 未存全部资料 hidden；首 predictor 的前缀状态也缺失。不能直接形成完整 ECS |
| `nll[N]` | 坐标及选中 token 概率检查 | 一个 NLL 标量无法恢复每层 FFN 前后完整词表分布 |
| HARP、GHOST、LUMINA 标量/投影 | 各自独立方法 | 无法逆推出 ReDeEP 的 top-k attention 与 FFN 两侧分布 |
| R7/R19 ReDeEP 值 | 旧 Qwen 适配结果保留 | 模型、时点、标准 JSD、聚合均不同，不能移作本次 Llama2 正式基线 |

已核 `src/feature_qa.py`、`src/run_feature_qa_all.py`、`fit_expansion/run_llama_expansion_v3.py` 的实际保存代码，未加载任何分数/测试回答。原缓存确实只保存回答状态并释放全输入状态。

**必须新抽取**：①对应精度/模板的全前缀最终状态；②指定 copying heads 在 predictor/块上的 attention 及关注位置；③选定/候选 FFN 层前后残差经 norm＋lm_head 的完整分布差；Chunk 另需精确分段与 BGE 嵌入。可以分块即时归约，不必永久保存完整大张量，但不能改公式、dtype、分母或目标时点。

## 6. 单 8GB 的忠实执行边界

原 FP16 7B 权重约 13GB 量级，**仅权重就超过 8GB**。以当前最长输入 1232 为例，原脚本同时保存全部 32层×32头 FP16 attention 约 **2.90GiB**，32层 FFN 前后完整词表 logits 约 **4.70GiB**，尚未计 hidden/KV、临时 softmax 与 BGE。这是静态数组估算，不是实际峰值测量；原实现不可能全部驻留单张 8GB。

保持 FP16 权重的 CPU/offload＋逐层/逐 query/vocab 归约，是**可能的工程路线**，但需验证自定义 residual/logit 接口、dtype及实际显存，不保证原 `device_map=auto` 脚本直接成功。若使用现有 NF4 来省显存，便是明确的量化适配，不能写“仅数据接口改变的完整正式基线”。重新跑7B或CPU offload均未在本轮授权/执行。

最终分类：**ReDeEP 正式基线：公开实现缺项，尚未完成。** 当前已有的 ReDeEP-token/Qwen 标准JSD、四层段均值、LR 融合均归本地适配。本协议不改变任何已有成绩，也不以新方法替补冒名。
