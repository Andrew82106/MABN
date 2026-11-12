# RAGognize 同域共同划分审计

本轮统一数据已生成于 `../data/`。所有方法使用同一批官方 train 中的 Llama-2 回答和同一划分；所有模型从通用基座及新检测头开始。该数据以前已被读取并用于研究，准确名称是**共同数据内部重划分测试**，不是新外部盲测。

## 可用官方数据

2026-09-28 重新查询官方 HF API，当前 revision 仍为 `aab54518c2a7c0d25fff8bffbf5337d0321de142`。只有 train 1,842 行和 test 2,781 行，无 validation 或新增未见 split。每行包含同一个问题/资料条件下四个模型的回答，因此官方 train 的全部四模型回答为 7,368 条，本轮只用其中 1,842 条 Llama-2 回答。

原数据位于 `../../auxiliary_ragognize_train_v1/train-00000-of-00001.parquet`（该相对位置以 benchmark 根目录定位更稳妥）；SHA-256 为 `6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44`。模型为 Llama-2-7b-chat-hf、Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.1/v0.3。问题来自近期 Wikipedia 资料，包含同题可回答/不可回答配对。回答自然生成；标签为 Gemini 自动标注的字符片段，经既有本地 tokenizer 转为 token/window 标签，属于银标。

已读取和测分的官方 test 2,781 条只可作历史参考；`RAGognize-with-samples-test` 复用同一测试问题，额外采样没有提供新的独立带标测试集。本轮转换程序不访问官方 test 或新的候选测试标签。

来源：[官方数据卡](https://huggingface.co/datasets/F4biian/RAGognize/blob/aab54518c2a7c0d25fff8bffbf5337d0321de142/README.md)、[官方论文](https://arxiv.org/html/2604.15945v1)。实时 API 快照见 `HF_DATASET_API_CURRENT.json`。

## 最终分组与划分

旧 `GROUP_INDEX.jsonl` 的 917 组先连接同一问题、来源身份及完整检索资料。进一步在全 1,842 行上，将**目标资料段**与任何其他样本中的相同规范化资料段连接，取传递闭包；规范化为 NFKC、casefold、合并空白，不依据标签。干扰段之间的共享不单独建立边。

共补充 61 条跨旧组的样本连接，最终 883 组，最大 28 回答。新 group ID 为有序成员原始行号的 SHA-256 前 16 位。按 `SHA256('20260928|' + group_id)` 排序，前 `ceil(0.15*883)=133` 组为 test，随后 133 组为 validation，剩余 617 组为 fit。

| split | 回答 | 组 | no-hint 回答 |
|---|---:|---:|---:|
| fit | 1,292 | 617 | 681 |
| validation | 271 | 133 | 135 |
| test | 279 | 133 | 155 |

规范化问题、来源 identity、来源 URL、来源标题、完整 context 均无跨 split 精确重复；目标资料到任何资料的精确重复也无跨 split 边。该检查不证明语义改写或近重复完全不存在。划分与 no-hint 计数均未使用幻觉标签；没有计算 test 风险比例或其他标签统计。

## 文件接口

- `data/MANIFEST.json`：完整性、来源哈希、训练与测试口径。
- `data/split_assignments.jsonl`：response_id、train_row_index、split、新 group_id、legacy_group_id、no_hint；不含幻觉目标。
- `data/GROUPS.json`、`GROUP_AUDIT.json`、`SPLIT_FREEZE_BEFORE_LABEL_COPY.json`：分组与读取标签前的划分冻结记录。
- 三个 split 都有 `inputs.jsonl`、`geometry.jsonl`、`window_geometry.jsonl`。它们逐行原样复用已审计的 `external_eval/train_fixture`，保留原始 question、documents、retrieved_passages、released_prompt、original_response、prompt_prefix、document_char_spans 及完整分词几何，不含目标或 hint 标签。
- fit 与 validation 另有 `answers.jsonl`、`sequences.jsonl`、`tokens.jsonl`、`windows.jsonl`，保留旧转换训练字段，包括完整官方 `released_hallucinations`、answer_risk、risk_mask 和 raw/clipped offsets；group_id 更新为新组，并保留 legacy_group_id。
- fit 与 validation 的 `scoring.jsonl` 可供验证评分；test 目标只在 `test/sealed/labels.jsonl`，包含原 spans、token risk_mask、window_labels，预测器和调参流程不得读取。

1842 条 predictor 输入均核对了 input hash、完整 prompt+空格+answer hash、context 字符区间、回答 token IDs、raw/clipped offsets；原始输入和几何文件的来源哈希均与旧审计清单相符。没有模型前向或 GPU 操作。

## RAGognizer 可比性

官方发布检查点已经通过官方 train 的训练/验证流程，不能作为本轮内部留出比较的初始化。必须从同一通用 Llama-2 基座重新训练原 LoRA+检测头结构，并显式读本轮 fit 与 validation 文件。官方 `ft.py` 先扩展回答再 `train_test_split(test_size=0.1, seed=42)`，这会拆开同来源样本；本轮只替换数据划分入口，保持结构和目标清晰可追溯。

官方 `--allentries` 会加入四个生成模型的回答。当前共同数据协议是 Llama-2-only；该选项不能只对基线开启。双方只能用 fit 目标更新权重，validation 选模型和阈值；test 在模型、特征和阈值冻结后只评分一次。训练使用全 fit；no-hint 为主测试，all 为预先声明的补充结果。

官方代码固定 revision `1d3e8fdd6de638dc2d06265829a4d9ced387e0ba`，快照见 `official_ft.py`、`official_demo_ft.sh`、`OFFICIAL_CODE_TREE.json`。来源：[官方训练代码](https://github.com/F4biian/RAGognizer/blob/1d3e8fdd6de638dc2d06265829a4d9ced387e0ba/fine-tuning/ft.py)。

生成程序：`build_same_domain_data.py`。若数据清单已经存在，程序拒绝覆盖，防止训练后悄然重切。
