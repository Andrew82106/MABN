# Expanded-v4 microclaim cross-encoder

训练器直接复用原 crossencoder 的模型、risk logit、优化器、批处理、单轮训练、推理、checkpoint 和 GPU smoke。输入绑定到冻结 v4 的 question + three-passage top-2 evidence + contextualized claim；标签、source-connected 五折、层级/类别权重和 4-BPE 回投关系也绑定 v4。

当前只完成 CPU prepare/check/tiny。GPU smoke 和六个正式训练进程必须另行显式启动。official test 与正式 baseline 均不动。
