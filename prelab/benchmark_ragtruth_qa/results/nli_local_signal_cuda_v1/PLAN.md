# 冻结 NLI CUDA 执行版（我们的方法候选）

同793输入/8829片段/四视图/35316对、同ModernBERT参数与E/N/C；CPU产物只读。CUDA FP32 batch8，禁混精度、TF32及编译，使用SDPA math。原读出源码通过私有模块复用，新输出目录独立。

prepare/check无模型和GPU；gpu-smoke比较原CPU108对（最大概率差≤3e-5）、生产逐答批次和最长683词元批。同路径重复exact，峰值allocated/reserved均≤6.5GiB；失败保留，不降批或放宽。extract须独立显式授权，只接受带新CUDA签名的逐答缓存。score另起CPU进程；verify只复算，不重新拟合。

5折只针对读出层，current上游仍in-sample且曾用cal选型；本分支不是正式baseline或独立最终测试。
