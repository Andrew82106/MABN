# 语义外部基线适配

所有步骤使用现有 `prelab/.venv/Scripts/python.exe`，未更改共享 Python 依赖。固定 fit 1292、validation 271、test 279；本目录不读取 test 标签。GPU 由根任务统一排程。

## LettuceDetect

实际使用论文初版同期的官方 **0.1.6，2025-02-27，d96e413aaf39f97739207759e8f393df1ecc38af**。较新的 2096ed28 版本包含 v2 内容，只归档，运行不加载。源文件、下载链接和 SHA 见 `LETTUCE_V1_SOURCE_MANIFEST.json`。

直接调用上游 `RagTruthDataset` 和 `Trainer` 的 AdamW 构造；保留原生双 logit 分类、回答 token 交叉熵、prompt 的 -100 遮罩及末尾 SEP 的负类标签。上游 `predict_prompt` 路径接收原生成提示，因此本适配使用数据自带 released_prompt，未添加“资料不足”提示。所有答案均没有截断，首答案 token offset 都为 0。

与官方训练程序的差异只有运行和统一评测适配：固定本项目的源文档分组划分；统一 seed；原 batch8 用 microbatch1 按整个有效 token 分母精确累积；BF16 前向、FP32 参数/梯度/Adam、非重入梯度检查点、SDPA、不启用 torch compile。无 scheduler/clip/group-weight/answer-weight/lexical-weight。训练仍为 AdamW lr1e-5、默认 wd.01、6轮；验证选轮和阈值用共同评价规则，保留原生 token argmax F1。不是把已有 Context-large checkpoint 改名，必须从通用 ModernBERT-large 开始。

CPU 检查覆盖全部1842条编码、无截断、原跨度标签、映射，以及 batch CE/梯度/AdamW 更新与微批累积的等价性。续训保存每轮末模型、优化器和随机状态；程序中断可从最近完成轮恢复。

```powershell
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\lettuce_v1.py cpu-test
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\lettuce_v1.py smoke
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\lettuce_v1.py train
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\lettuce_v1.py predict --split test
```

训练自动产生选中轮的 `lettuce_v1_run/validation_predictions.jsonl`；最后命令产生 `test_predictions.jsonl`。每行含 response_id、原 Llama BPE 的 token_scores，同时保留原生 token 分数和偏移。

## MiniCheck

官方 MiniCheck-Flan-T5-Large checkpoint revision 为 `96eafd01cee2d16cf81aaa2fb226b14f422a37b3`，模型每个文件 SHA 由下载完成后的 `MODEL_MANIFEST.json` 记录。代码锁定 `b58b9fa69acbd1015ec970fa65dd752413a053d2`，见 `SOURCE_MANIFEST.json`。只使用官方权重，不使用旧 MiniCheck-tail2、TCN 或其他自制头。

调用未改动的官方 `Inferencer.inference_per_example`：按换行与 NLTK 断句；材料按500词成块；原模板 `predict: document</s>claim`；最大2048token；decoder起点0，标签token3/209；材料块支持度取最大。仅把官方自动联网模型加载换成锁定的本地快照与明确设备，保持 FP32。

按官方 doc+claim 接口核查“资料＋原句”。问题不作为证据、不拼成断言，保留在准备文件中以说明输入约定；这是原句核查基线，可能无法完全解释依赖问题的简短回答。没有外部 LLM 改写。

输出风险为1−支持概率，保留原句跨度、每个材料块的概率；句级结果通过原字符位置映回 Llama BPE。它不是原生逐 token 检测器。所有550条准备已通过，2482句，分块后最大968token，没有2048截断。

```powershell
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\minicheck_official.py cpu-test
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\minicheck_official.py smoke
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\minicheck_official.py predict --split validation
& 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe' -u -B -X utf8 .\minicheck_official.py predict --split test
```

输出在 `minicheck_run`；每条完成即独立原子写盘，重跑按输入和checkpoint绑定恢复。本方法不在本地 fit 上再训练，主表注明“公开模型迁移”。

## 共同坐标适配

有文字的原生成 token 按其非空白字符的原生分数加权。纯空白 token 若有原生覆盖就按实际字符覆盖映射，无覆盖间隙取最近原生区间，距离相同时取更早者；不人为填0影响 mean4。每个 token 权重和为1。风险分数方向始终为越大越可疑。统一 mean/max 四词元适配只在 validation 选择。

官方链接：[LettuceDetect v1](https://github.com/KRLabsOrg/LettuceDetect/tree/d96e413aaf39f97739207759e8f393df1ecc38af)；[MiniCheck](https://github.com/Liyan06/MiniCheck/tree/b58b9fa69acbd1015ec970fa65dd752413a053d2)；[MiniCheck权重](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large/tree/96eafd01cee2d16cf81aaa2fb226b14f422a37b3)。
