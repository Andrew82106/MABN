# Lookback Lens：作者结构8-BPE迁移

按作者代码固定8个原生BPE滑窗、stride1，均值1024维；唯一分类器为`LogisticRegression(max_iter=1000)`，其余默认。没有scaler、sample_weight、C搜索、NLL/HARP或平滑。

| 划分 | 8-BPE窗 | span AUROC（主指标） | span AP | 窗口F1（额外） | 整答F1（额外） |
|---|---:|---:|---:|---:|---:|
| fit | 639955 | 0.914414 | 0.677509 | 0.621449 | 0.731419 |
| calibration | 41685 | 0.855601 | 0.624460 | 0.587706 | 0.817734 |

校准窗口TP/FP/FN/TN=3853/2673/2733/32426；整答=83/20/17/39。
两个阈值只由fit内分数确定：窗口0.298034296，整答0.429758702；未用cal选阈值或参数。整答采用全部完整8-BPE窗max，是项目接口适配，不是作者原输出。
保留3680/159答的615/154材料组隔离；2条fit短答（12818、14641）不足8BPE，按作者完整窗口规则无span，记录N/A，未补0。cal159全部有窗。主指标为8-BPE sliding-span AUROC，不能把这里的窗口F1当作原4-BPE成绩。
真实拟合99.67秒，迭代[334]；警告[{'type': 'DeprecationWarning', 'message': 'scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.'}]。数据采用已有Llama NF4统一重放、真实header及pre-read特征，存在骨干/软件/读取环境迁移差异，不宣称逐数值复现。
材料组拆分、人标风险OR及fit-only F1阈值均为明确的数据/报告适配；fit阈值来自训练内分数，可能过拟合，cal也已反复开发。sealed test150未读。

[原论文§2.1、Table2、AppendixC.1](https://aclanthology.org/2024.emnlp-main.84.pdf)；[锁定作者代码](https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step03_lookback_lens.py)。本目录AUTHOR_SOURCE.json记录下载hash；没有下载模型或新数据。
