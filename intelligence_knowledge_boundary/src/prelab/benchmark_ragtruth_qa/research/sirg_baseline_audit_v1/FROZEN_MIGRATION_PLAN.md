# SIRG 官方代码结构迁移：冻结执行计划

版本：v1，2026-09-13  
状态：仅完成审计；未启动 GPU  
目标身份：**SIRG 官方代码结构迁移（Llama-2-7B teacher-forced common replay）**

## 不可改变的边界

### SIRG 方法层

以下内容按作者提交 `0e5e310891f16d81b8f91c874125bdcc8000c259`冻结：

1. 完整 BF16 Llama-2-7B-chat 和 LXT 2.0 AttenLRP；
2. 对固定回答逐 token teacher forcing，以温度缩放后的实际 token logit 反传；`max_steps=500`；
3. 作者的换行 + spaCy 句界、最短 5 字符规则、实体抽取和现存实现行为；
4. 回答实义 token 的绝对 LRP 均值、源片段 token 最大值、作者归一化；
5. Top-15 来源片段、贡献权重保留两位小数、`Context i` 与 `Previous Generate:`模板；
6. AlignScore-base/RoBERTa-base pooled output、dropout 0.1、二分类线性头；
7. 类别权重 `[0.1, 0.9]`交叉熵、batch 16、长度 512、`only_first`截断；
8. seed 2022、AdamW `lr=1e-5`、`eps=1e-6`、weight decay 0.1、warmup 6%、FP32、100 epochs；
9. 以内部验证集片段 F1、原生阈值 `p(risk)>0.5`选择单一最佳 checkpoint；
10. 冻结 checkpoint 的连续 `p(class 1)`作为原生片段风险。

正式运行不得加入平滑、校准器、NLI 替换、额外信号、窗口分类头或参数搜索。不得修正 `noun_spacy`字段错误等会改变特征的代码行为。

### 本项目评测层

以下由共同 benchmark 决定：

- fit：3,680 个回答、615 个材料组；
- cal：159 个回答、154 个材料组；fit/cal 材料严格不相交；
- 4 个原始 BPE、步长 1 的合格窗口；
- 片段标签、片段到窗口映射、整答 `max(window)`；
- 窗口和整答在 cal 上分别 F1Opt，阈值并列时按 precision、较高阈值决胜；
- 风险为正类，统一报告 F1、AUROC、AP；
- 正式 test 在模型、适配器和阈值全部冻结前保持封存。

## 阶段 0：冻结来源和运行资产

在任何 GPU 工作前完成：

1. 保存作者仓库 bundle 或 tar，记录提交和 [`official_source_hashes.json`](./official_source_hashes.json)。
2. 冻结 LXT 2.0 release/commit，不跟随 `main`。
3. 冻结完整 BF16 Llama 资产：`NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee`，保存每个 safetensors 哈希。
4. 冻结官方 `AlignScore-base.ckpt`，保存 SHA256；它是起始点。
5. 在单独环境做依赖兼容试装。作者未给完整版本，选定能执行原代码语义的一组版本后生成 lockfile，并把它明确写成“本地复现约定”。
6. 安装并冻结 spaCy `en_core_web_lg`和 Stanza 英文模型文件；记录版本和资源哈希。
7. 只允许建立路径/设备/文件名/入口封装补丁。为每个补丁写明：原行、修改、为何不改变张量或文本语义。

**门 0：** 所有资产能离线加载；环境和补丁有哈希；若需要改方法语义才能运行，则停止，状态保持 N/A。

## 阶段 1：CPU 数据接口

### 1.1 共同输入

对每个回答生成冻结输入记录：

```json
{
  "answer_id": "...",
  "group_id": "...",
  "prompt": "exact benchmark prompt/context/question",
  "response": "exact published answer",
  "temperature": 0.7,
  "answer_char_length": 123,
  "fit_or_cal": "fit"
}
```

文本必须逐字符保留，不能重写资料、问题或回答。温度沿用记录值，不缺省为新值。

### 1.2 fit 内部 train/validation

作者需要内部验证集选 checkpoint，但 cal 禁止参与。对 615 个 fit 材料组按固定字符串 `sirg-v1:<group_id>` 的 SHA256 排序，前 80% 组进入 train，后 20% 组进入 validation。该划分只看 group id，不看标签；组内所有回答同行。

保存：

- 每个组的哈希与去向；
- train/validation 的回答数、组数、片段数和正负数；
- 与 cal、未来 test 的组交集必须为 0。

这是项目的数据隔离规则，不改变 SIRG 的训练目标。禁止根据类别比例重新挑组。

### 1.3 片段监督

先按冻结 SIRG 代码得到回答语义片段及字符跨度。每个片段的训练标签固定为：

```text
fragment_risk = 1
    iff the fragment character span overlaps any positive 4-BPE gold window
else 0
```

重叠定义为半开区间交集长度大于 0。该规则只在 fit 内读取 gold，产生作者二分类器所需的片段标签。不得用 cal 标签改变片段边界、过滤片段或选择映射。

**门 1：** 文本哈希、材料组隔离、4-BPE 几何和片段标签可由独立脚本逐值复算；未读取 test。

## 阶段 2：短样本 LRP 兼容试验

在 40 GiB A100 上先运行一个固定的 8 回答无标签样本，只检查管线，不评估效果：

1. 模型必须显示 BF16，且没有 4/8-bit 量化模块；
2. LXT 版本必须为 2.0；
3. 每个输出 token 的相关度维度与当前前缀长度一致；
4. 相同输入和 seed 重跑时，关键输出在声明的数值容差内一致；
5. 与作者 `max_steps=500`一致；
6. 不出现 NaN、空向量或越界；若出现，记录失败，不静默替换为其他归因方法。

**门 2：** 8/8 样本产生结构合法的 LRP 文件；峰值显存和单位样本时间已记录。24 GiB 设备只有通过此门后才可考虑，不使用量化或 CPU offload 冒充正式运行。

## 阶段 3：全量 LRP、实体和图

对全部 3,839 个回答执行冻结 teacher-forced replay。每个阶段保存逐样本 manifest：输入哈希、token 数、处理 token 数、输出哈希、运行时间、失败原因。

预先接受作者限制：18 个回答超过 500 回答 token，尾部约 203 token 没有 LRP。不得延长上限；这些尾部后续没有片段风险覆盖时按 0 分映射。

随后按冻结代码执行：

1. `get_entity.py` 实体抽取；
2. `get_score.py` 语义切分和图权重；
3. `generate_training_data.py` Top-15 线性化。

需要把脚本硬编码的 Qwen/Llama 入口、目录和文件名改成显式参数，但输出文本、数值和排序必须与原函数相同。所有零分母、空片段、重复偏移和截断事件单列计数；不能为提高覆盖率修改规则。

**门 3：** 每个回答都有成功或明确失败状态；随机抽查字符跨度、token 偏移、Top-15 排序和模板；训练文件只来自 fit，cal 文件无标签。

## 阶段 4：AlignScore 训练

从冻结的官方 `AlignScore-base.ckpt`初始化，按方法层参数训练。训练数据只来自阶段 1 的 fit-internal-train；验证只用 fit-internal-validation。

每 epoch 保存：loss、片段 precision/recall/F1、checkpoint 哈希。选择规则严格使用 `p(class1)>0.5`的验证片段 F1，保留一个最佳 checkpoint。若 F1 并列，使用首次达到者作为预先约定，避免事后挑 epoch。

不得：

- 用 cal 早停或选 epoch；
- 搜索类别权重、学习率、epoch、Top-k 或随机种子；
- 在 checkpoint 选择后用全部 fit 再训练一遍；
- 添加本项目窗口损失或回答损失。

**门 4：** 最佳 checkpoint 可从日志自动复核；输入组均属于 fit；模型头、损失和参数与冻结代码一致。

## 阶段 5：冻结原生输出

用最佳 checkpoint 对 fit 和 cal 产生每个 SIRG 回答片段的连续 `p(risk)`，先保存不含 gold 的独立文件：

```json
{
  "answer_id": "...",
  "fragment_index": 0,
  "char_start": 10,
  "char_end": 42,
  "risk_score": 0.731,
  "fragment_text_sha256": "..."
}
```

另做作者原生诊断：片段 `p>0.5`；整答幻觉片段比例及 α 网格。诊断必须注明作者整答 F1 的正类方向，不进入统一主表。

**门 5：** 原生输出文件在读取 cal gold 前冻结哈希；每个分数有限且位于 `[0,1]`。

## 阶段 6：确定性 4-BPE 适配和统一评测

映射器只读冻结原生输出、回答文本和 tokenizer 几何，不读 gold：

1. 对每个合格 4-BPE 窗口，找与其字符跨度相交的 SIRG 回答片段；
2. 窗口风险为这些片段风险的最大值；没有覆盖则为 `0.0`；
3. 整答风险为全部合格窗口的最大值；
4. 保存映射后的无标签分数文件及哈希。

独立评分程序之后才打开 cal gold：窗口与整答分别 F1Opt，按 `F1 → precision → 较高阈值`处理并列，报告 F1/AUROC/AP和阈值。还要报告：片段覆盖率、无覆盖窗口数、500-token 截断影响、按生成器分组结果。

**门 6：** 对同一冻结分数独立复算差异小于 `1e-12`；answer score 逐条等于 window max；适配器代码中不存在 gold 字段访问。

## 正式结果身份与失败处理

正式表中的方法名固定为：

> SIRG official-code-structure transfer, Llama-2-7B teacher-forced common replay

报告同时注明：

- 不是作者原数据/原 checkpoint 的逐值复现；
- 闭源生成器回答的白盒状态来自共同 Llama-2-7B 重放；
- 原生是语义片段定位，4-BPE 是无参映射；
- 论文/代码冲突按提交 `0e5e310`处理；
- 方法分数无论高低均保留，不为压低基线而改实现。

任一道门失败，正式成绩保持 N/A。可报告失败原因和已经完成的资源统计，但不能拿 `token_source_attribution_v4`、简化 LRP、普通注意力或未微调 AlignScore 填入 SIRG 行。

## 预计资源

| 资源 | 计划值 |
|---|---:|
| 首选 GPU | 40 GiB A100 |
| GPU 预算 | 35–50 A100 小时（工程预留） |
| LRP 主体估算 | 约 27.2 A100 小时 |
| 全预处理与判别估算 | 约 29.5 小时，不含未知训练时间余量 |
| 磁盘 | 25–40 GiB |
| 初始 AlignScore checkpoint | 约 1.97 GB |

当前 8 GiB RTX 3070 不满足正式完整 BF16 LRP 条件。等待可用 GPU 前，可以完成阶段 0、阶段 1 和阶段 6 的无标签适配器单元验证，但不得产生或宣称正式分数。
