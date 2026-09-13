# RAGLens 正式基线下载前门禁

核查日期：2026-09-13。这里只做源码、远端权重元数据、依赖和本机资源检查；没有下载 SAE/模型，没有加载 CUDA，也没有训练或评分。当前结论是 **N/A**：方法身份可以冻结，但本机尚不能按作者默认链路忠实执行。

## 冻结的方法身份

- 论文：[ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f6bfbd260fbf2f5acb0a1d6439ca0e-Abstract-Conference.html)。作者在 RAGTruth 的 Llama2-7B 上报告回答级 AUC/平衡准确率/macro-F1 为 `0.8413/0.7576/0.7636`；没有报告词元或跨度 F1。
- 官方仓库：[Teddy-XiongGZ/RAGLens](https://github.com/Teddy-XiongGZ/RAGLens/tree/e10ed5824ae35bac4957dfe13615d9e40c0555cf)，commit `e10ed5824ae35bac4957dfe13615d9e40c0555cf`，tree `d52491b911f61cbd7a508eaa5f2ad1179db76988`。上游 Git blob 内容 SHA256：`requirements.txt`=`457f05dfd60e95e51c78b7b5736ea10dd5a474ae648d64803fc95ee688bc3196`，`src/RAGLens.py`=`07d840f302bc649feb4e8c6c132ae8152abbd0e0821cac42e14874e498ea7c4f`，`src/sae_encoding.py`=`d47ada1a97a2770eae14b90f6886ab61eb3ba0bb21cbb2e8820bf7648532bfe8`，`src/utils.py`=`f194747a31b31f3b6ef8fb67b5e617682b82db4788dfb701e0e74f264df31f94`。
- 唯一允许的 SAE 是作者论文指定的 [`yuzhaouoe/Llama2-7b-SAE`](https://huggingface.co/yuzhaouoe/Llama2-7b-SAE/tree/7dbfc87e7fade7c154a24353a3afe98f47600cd1)，revision `7dbfc87e7fade7c154a24353a3afe98f47600cd1`，`layers.15/sae.safetensors`，LFS SHA256 `65a310a99afcbf4a836373a96dc5b87c38868bf8b7bcb46a4a21256a1f14f898`，`4,295,508,312` bytes。配置固定为 `d_in=4096`、expansion `32`、Top-K `192`、normalized decoder。2026-09-13 通过 Hugging Face 官方 API 确认文件仍可获得，本机未缓存该文件。
- 底座固定为本项目已有的 Llama-2-7B-Chat 完整 safetensors：`NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee`。其权重文件元数据哈希已在 [CHECKPOINT_AND_QUALITY.md](../../CHECKPOINT_AND_QUALITY.md) 中核为与 Meta 官方 revision `f5db02db724555f92da89c216ac04704f23d4590` 相同。必须按作者代码用未量化 BF16 权重；现有 NF4 激活不能复用。SAE 模型卡没有记录其训练底座的精确仓库/revision，这一上游出处缺口必须披露，因此即使执行成功也只能称“论文结构迁移”。

方法层固定如下，不能为适应本项目改动：

1. 读取 `layers.15` 的输出，送入上述 SAE；使用 SAE **pre-activation**，不改成 Top-K 后激活。
2. 只对回答词元逐通道取最大值，得到 `131,072` 维整答向量。
3. 用 50 个 bins 估计互信息，选 top `1,000` 特征。
4. 使用 `interpret==0.7.2` 的 `ExplainableBoostingClassifier`：`interactions=0`、`max_bins=32`、`random_state=0`、`early_stopping_tolerance=1e-5`、`validation_size=0.1`、`max_rounds=1000`；其余默认值由固定依赖版本决定。
5. 训练标签是回答级“是否含任一人工 hallucination span”。fit 用于特征选择和训练，calibration 只按本项目统一规则选阈值与计分。

若未来放行，必须先让冻结的Llama2＋SAE链路为本项目全部3,839答生成原生特征，再只用3,680条fit（615个材料组）做MI选特征和EBM训练；159条cal（154组）不得进入拟合。当前fit包含原634条回答和同一项目数据扩展出的3,046条跨生成器回答；它们都按完全相同的冻结Llama2检测链路重放。这是统一数据接入，不改变RAGLens模型、特征或训练法。主开发计分仍只在cal159上进行。

作者的 `explain()` 会从推动整答风险上升的特征中最多取 5 个，再标出各特征激活峰值附近默认 `±3` 个词元。它是整答判定的局部解释，没有逐词元概率，也没有定位 F1；不能把它改写成训练过的词元探针或当作正式定位分数。

## 统一 4-BPE 适配

正式主比较只采用无歧义的零参数映射：保留 `predict_proba()` 的整答风险原值，将其原值广播到该回答全部可评 4-BPE 窗口；整答仍是同一原值。随后按本项目固定标签和阈值规则评分。这样不会给 RAGLens 新增局部模型，也会如实暴露其原生检测器只在整答级受监督。

`explain()` 的峰值位置以后可以单列为解释覆盖率诊断，但不得替换上述正式分数、不得借人工 span 选峰值，也不得把 GAM contribution 与其他信号重新融合后仍称 RAGLens 基线。

## 本机门禁

| 项目 | 核查值 | 结论 |
|---|---:|---|
| GPU | RTX 3070，8,192 MiB；检查时空闲 7,281 MiB | 不足 |
| Llama2 完整权重 | `13,476,872,576` bytes，约 12.55 GiB | 已在本机磁盘；单独已超过显存 |
| layer-15 SAE | `4,295,508,312` bytes，约 4.00 GiB | 远端可得；未下载 |
| 两组静态权重下界 | 约 16.55 GiB，未计激活、CUDA和临时张量 | 作者默认 `.to(cuda)` 路径必然超过 8 GiB |
| SAE 单样本 pre-activation | 每 1,000 输入词元约 500 MiB（FP32） | 还需额外峰值内存 |
| fit/cal 聚合特征 | 3,680/159 答约 920/39.75 MiB（FP16） | 磁盘不是瓶颈 |
| 系统内存 | 63.79 GiB，总空闲 38.17 GiB | 静态权重可容纳；CPU速度与峰值尚未实测 |
| 磁盘 | D: 空闲 78.45 GiB；C: 空闲 205.13 GiB | 足够下载 layer-15 SAE |

官方 `data_encoding.py` 对非 70B 模型先把完整 BF16 Llama2 移到检测到的 GPU，再把 FP32 SAE 移到同一设备。因此本机 8 GiB 不是估算风险，而是明确的默认链路阻塞。CPU运行或 CPU/GPU offload 尚未做等价性和耗时验证；`decoder=False`、NF4、换用 1B 模型或其他 SAE 即使能省显存，也都不是当前冻结基线。

当前 `prelab/.venv` 也不满足作者环境：Python `3.11.15`；已有 `torch 2.5.1+cu121`、`numpy 2.4.6`、`transformers 4.51.3`、`huggingface_hub 0.36.2`，而作者锁定 `torch 2.7.1+cu128`、`numpy 1.26.4`、`transformers 4.56.1`、`huggingface_hub 0.34.4`，并缺 `pandas 2.1.3`、`interpret 0.7.2`、`datasets 3.0.0` 等依赖。以后执行必须使用隔离环境，不能覆盖现有预实验环境。

## 放行条件

保持 N/A，直到依次满足：精确 SAE 下载后核 LFS SHA256；建立作者版本隔离环境；在不量化、不换 SAE/层/特征/分类器的条件下解决设备放置；用小样本证明设备迁移前后 SAE 特征与官方代码等价；为全部3,839答完成冻结特征推理，只用3,680条fit训练、保留159条cal计分并冻结原生整答概率；最后才执行上述零参数广播和统一计分。任何替代 SAE、NF4 激活或自建词元头只可列为本文方法/消融。
