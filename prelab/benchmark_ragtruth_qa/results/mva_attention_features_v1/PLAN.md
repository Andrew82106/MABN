# MVA作者代码统计：CPU准备阶段

本目录只准备MVA的三类输入特征，**不等于完整MVA基线**。没有检测器、标签训练、阈值选择或正式特征提取循环。作者代码的统计严格锁定在 `f8b871a06b6c18dabe5881bd02a68854b2940b81`。后续正式MVA基线必须另建目录，保留作者结构和后处理；256维小Transformer、partial-label CRF等改造都只能标作我们的适配方法，不能混入正式基线。

## 锁定的输入和计算

- 仅原634 fit＋新增3046 fit＋原159 cal，共3839条已公开开发输入；材料组615/154不交。只读原plans中的original视图，不使用随机资料、标准答案、金标或封存test。
- 完整原token序列最长1232、总2408466，原回答共708506 BPE。答案跨边界首词元、标点及最后词元全部保留，计算完成后才按原answer_positions切片。
- 逐层逐头复算因果注意力。Q/K使用当前固定NF4/BF16模型，点积沿模型dtype，softmax用FP32，然后转FP16作为作者raw统计输入。这是量化模型的重建轨迹，不能称原发布生成trace。
- 作者 `key_avg` 分母为完整输入T、query行号从1起；incoming entropy用原A列归一；outgoing entropy用原A行归一。第一次归一后FP16舍入，再进入作者熵函数第二次归一，epsilon均1e-9，熵按非零项数归一。**不使用论文中的另一分母或校正熵公式。**
- 输出列顺序为 `key_avg`、`key_entropy`、`query_entropy`；每类按layer/head展开，原答案BPE为行，float16 `[N,3072]`。这是读取当前词元之后的Q行；incoming还使用后续回答，因此仅适用于离线。

## 有界内存方案

一次只处理一层、2个head；每32个query重建attention块，但暂存这2个head的**全部T×T FP16矩阵**。完成完整列后，按单head的2D算子执行作者统计。没有沿query截断、分块各自归一、先删未来词或用简化熵恒等式代替FP16中间步骤。

在T=1232时，两头attention buffer约6.07MB；单head FP32矩阵约6.07MB；一层旋转Q/K约20.19MB，统计中间张量会额外占用内存。预留512MiB特征工作区是工程估算，**不含模型、前向激活和CUDA运行时，也不是实测峰值**。真正加载后仍须单独执行最长样例gpu-smoke才能判定8GB是否够用。原作者堆叠全层矩阵和norm变体不在本实现内。

全量FP16特征约4.353GB，若转FP32约8.706GB；坐标和JSON另计，建议6GB磁盘余量。此轮正式存量为0。未来吞吐需按实际smoke报告，本轮不虚报GPU耗时。

## 三个互不自动串联的阶段

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/mva_attention_features_v1.py prepare
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/mva_attention_features_v1.py cpu-selfcheck
# 本轮禁止执行下行，等待根代理审码与GPU调度：
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/mva_attention_features_v1.py gpu-smoke
```

`prepare`绑定3839原plan与原raw坐标、官方源码SHA、原模型资产身份/加载精度/软件版本，输出protocol、输入计划与资源清单。当前尺寸及历史资产hash身份用于CPU准备，gpu-smoke在加载前再读取实际权重文件核完整SHA。

`cpu-selfcheck`执行锁定源文件中未经修改的 `get_features` 函数AST，只给其不用的第四个Lookback输出提供一个合法虚拟边界；三项MVA分支保持原函数。手工小因果、全零、单token、单非零、稀疏/极小概率与随机矩阵必须逐值FP16一致。还检查完整分母手算、答案最后切片、NumPy接口、非法非有限输入拒绝、TinyLlama真实全层eager attention独立核对、GQA、不同分块、hook退出清理及临时原子写回。Tiny模型运行在CPU，未加载Llama7B。

`gpu-smoke`只在显式调用时读取已通过的CPU报告，选择原输入首、末和最长样例，做有限性、坐标、重复同路径exact与真实峰值检查。独立dense数学核验是CPU TinyLlama；不把GPU同路径重复冒称独立算法oracle。该入口没有全量提取或训练选项，成功后也不会自动继续。

## 后续可接的缓存接口

`extract_one(model,row)`返回 `mva_raw` 和 `token_ids/answer_positions/token_start/token_end/token_start_raw/token_end_raw`，没有任何概率、风险标签或窗聚合。生产调用仍需另行调度。

`save_record`按记录写NPZ.pending→fsync→原子替换，再写绑定record/input/source/code签名及NPZ完整SHA的metadata，最后写commit。只有commit与全部hash/轴检查通过才算可恢复记录。匹配已有记录不重写；不匹配或缺commit却有已完成文件时停下保留，不静默覆盖。未提交.pending不是有效缓存。全3839最终manifest/complete必须由未来被授权的提取协调器在完整核验后生成；当前脚本不生成伪complete。

任何公式、数据或数值失败都保存FAILURE报告，不放宽容差掩盖。当前仅授权prepare和CPU-selfcheck；没有训练模型、修改既有baseline、读取sealed test或占用GPU。
