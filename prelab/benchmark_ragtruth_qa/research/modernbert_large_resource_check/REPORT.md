# ModernBERT-large：8GB 资源与定位证据

**结论：不能把 large 当作现有 base 的可直接替换项并保证 8GB 可跑。** 按当前 FP32 参数／梯度／Adam 状态、BF16 前向和逐层检查点方案，它接近显存边界；静态计算不证明必然 OOM，也不等于实测通过。本次仅下载约 1.2KB 的官方配置、读取公开元数据，并在 meta device 上数参数；没有下载大权重、训练或初始化 GPU。

## 实际参数与显存

锁定通用模型 `answerdotai/ModernBERT-large` revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`：28 层、hidden 1024、16 头、MLP 2624。官方原 MLM 权重元数据是 395,881,664 参数；按本地 Transformers 4.51.3 换成二分类 token 头，实际是 **395,833,346**。同方法数 base 为 **149,606,402**，与本地已运行记录一致。[官方配置](https://huggingface.co/answerdotai/ModernBERT-large/blob/45bb4654a4d5aaff24dd11d4781fa46d39bf8c13/config.json)

| 口径 | Base | Large |
|---|---:|---:|
| 一份 FP32 参数 | 0.557 GiB | 1.475 GiB |
| 参数＋梯度＋Adam 两个矩，16 字节/参数 | 2.229 GiB | **5.898 GiB** |
| 再加一份参数大小的 FP32 临时张量，20 字节/参数 | 2.787 GiB | **7.373 GiB** |
| 本地实际训练最高 allocated | **3.117 GB＝2.903 GiB** | **未测** |

现脚本 `AdamW(...)` 没指定 `foreach`。本地 PyTorch 2.5.1 默认 CUDA 路径通常选 foreach，其平方根分母列表会额外分配约一份参数大小的张量；仅上表 20P 已让 8GiB 卡只剩约 **0.63GiB**，还没覆盖 CUDA 上下文、分配器保留空间、激活和其他临时量。[对应官方实现](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py#L606)

把 base 实测峰值直接按参数比放大得到约 **7.68GiB allocated**，这只是粗估，不是 large 实测，也不包含全部驱动占用。激活开销不与参数严格等比例。已有 microbatch=1，继续增加梯度累积不能减少固定的 16P；检查点也主要节省激活。若日后另立 `foreach=False` 方案，仍可保持全参 FP32 AdamW 状态并减少批量临时量，但必须单独记录实现差别、重新做最坏输入单步测试，不能本次宣称可用。

本地证据为 `results/full_context_encoder_v2/GPU_SELFCHECK.json`（979 token，3.109GB）及 `full_finetune/epoch_*.json`（六轮最高 3.117GB）。这些是 **allocated**，不是 reserved 或整张显卡总占用。精确计数及配置哈希见本目录 `RESOURCE_COUNTS.json`。

## 代码适配范围

- `run_full_context_encoder_v2.py` 的实际全模型前向和二分类头从 config 取尺寸，没有限定 768 或 22 层；`character_map`、`mapped_logits`、权重和评测也不依赖隐藏宽度。需新建版本并修改模型路径、revision／哈希、协议和 GPU 检查记录，不能覆写旧基线。
- `tail_finetune.py` 确有 layer22/23、hidden1024，但属于另一个 RoBERTa 两层缓存路径。完整输入 ModernBERT 只借用其数据、映射与权重函数，不走那条模型加载路径；不能机械把这些 22 改成 28。
- 若以后加入残差融合，`generation_residual_fusion{,_v2}.py` 的语义支路和 `cache_frozen_context_fusion.py` 的缓存确实写死 768，须适配为 1024并重建缓存／测试；`modernbert_generation_bridge.map_hidden` 本身按 `hidden.shape[-1]` 动态映射。**Llama LB 的 1024 维不变**，它不是 ModernBERT 的头数。

## 原论文的 QA 定位结果

LettuceDetect 原论文 v1 表3：

| QA 字符重叠定位 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base | 62.65% | 60.40% | **61.50%** |
| Large | 66.85% | 62.14% | **64.41%** |

定位 F1 提高 **2.91 个百分点**。这是 §5 定义的字符重叠指标，阈值 0.5，不是我们的四个原始 Llama BPE 窗口 F1；表2的 65.52%→70.18% 是整答结果，不能代替定位证据。论文跨 QA／摘要／数据转文本训练，本地数据和选型规则也不同，不能据此预测本地升级收益。[原论文表3与§4–5](https://arxiv.org/html/2502.17125v1#S5)
