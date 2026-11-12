单配置窗口TCN已固定训练30轮；直接监督每个4原始词元窗口，不再由词元概率取max得到窗口分数。以下均为校准选参成绩，不是最终测试。

| 模型 | epoch/C | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |
|---|---:|---:|---:|---:|---:|
| 同1089输入旧LR | 0.001 | 0.641 | 0.554 | 0.790 | 0.868 |
| 原tokenTCN | 6 | 0.602 | 0.593 | 0.741 | 0.827 |
| 直接windowTCN | 4 | 0.632 | 0.601 | 0.775 | 0.860 |

634 fit回答/168123窗口，159 cal回答/42241窗口；119个训练批次、952条fit链，每一轮每个fit窗口恰好出现一次。
输入是旧LR完整1089维窗口矩阵，PCA/scaler/标签/权重均复用并核对；没有新大模型生成或GPU调用。394处raw-start缺口断链，不跨答传播。
Versus token_tcn this also changes193->1089 features,64->32 width,rawBPE RF7->10,weighting and direct supervision. It is a predefined practical alternative, not a clean one-factor estimate of the supervision mismatch.
卷积使用前后窗口，是离线检测；RF7个窗口在连续处覆盖10个原始BPE，并非宣称模型内部信息只来自10个词元。全部30轮状态与概率均保留。
