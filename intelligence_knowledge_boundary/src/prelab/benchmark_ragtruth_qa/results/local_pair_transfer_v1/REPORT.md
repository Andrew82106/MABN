# 局部修复成对训练入口：CPU准备完成

**两个固定对照已具备可执行入口，尚未运行真实训练或GPU自检。** 一组只用两侧局部银标BCE；另一组在完全相同的数据和计算流程上增加固定排序项。入口：[run_local_pair_transfer_v1.py](../../src/run_local_pair_transfer_v1.py)。

## 冻结数据和预算

最终使用分词及局部范围都通过的 **10036对、5274条原答、5237个原材料组**。原10040候选全部仍在上游；4个标点目标对未进入词元损失。过滤后的组→原答→pair权重重新计算，逐值等于分词阶段的可用集合权重，总质量1。

两模式分别从generic ModernBERT-base、seed20261005重新开始。每pair先后单独前向原错误版和仅修一处的版本，保留两图后一次反向；两版本从不拼成一个输入。仅各自`target_raw_lexical_indices`参与辅助损失，修后整答和范围外字符不补0。

| 固定项 | 每个模式 |
|---|---:|
| 辅助轮数 | 1 |
| pair数量 / 前向次数 | 10036 / 20072 |
| 辅助有效batch | 8对，最后4对 |
| 辅助更新 / 后续QA更新 | 1255 / 3×460 |
| 总更新 | **2635** |
| 辅助encoder输入token | 15897749 |
| 总训练逻辑输入token | 21205199 |
| QA评估逻辑输入token | 5572614 |
| 最长完整pair两侧长度 | **1481 / 1483** |

逻辑token数不计梯度检查点重算；实际时间和显存将另记。旧QA输入、权重和完整6行顺序文件均逐字节复制；后续只取前3轮顺序。辅助和QA共享同一个AdamW，lr1e-5、wd0.01、clip1，无scheduler、无阶段重置。每模式只在3个QA轮全部完成后按原两级同候选规则选一个轮次；辅助阶段不看校准或挑轮次。

## 损失及公平范围

`paired_bce`使用冻结损失模块的λ0；`paired_rank`使用λ1，margin都为1：

`局部BCE = 0.5 × [原范围BCE(1)均值 + 修复范围BCE(0)均值]`

`排序项 = softplus(1 − 原范围平均risk logit + 修复范围平均risk logit)`

两组均计算相同局部BCE和排序项，仅乘数不同；不重新归一化两项之和。每batch实际B对，单pair缩放为`Npairs / B × normalized_pair_weight`，最后batch使用真实4对。pair内目标长度不同不改变两侧各半的质量。

作者修复侧0是**局部银标假设**，不是人标事实认证。增加排序项同时改变梯度强度和方向，不能把未来改善单独归因为“学会了对比推理”。范围外输出没有直接辅助损失，但源资料及其他上下文仍通过共享编码器收到间接梯度。这是额外语义核查编码器的辅助训练，不是原生成模型白盒探针。

## 实际CPU检查

- prepare：session9655，实际exit0。仅完整上游20080行分词及全量输出检查通过后才准备；过滤集合和每侧字节/哈希/原目标检查通过，未重新分词或扫描来源隔离。
- cpu-test：实际exit0，约8.1秒。两个TinyModernBERT模式使用相同初态和pair顺序；调用**实际生产辅助循环**处理11对的8＋3短batch，逐pair核目标梯度和范围外梯度0，并验证组→原答→pair质量、实际batch缩放。两组各22次前向。随后实际调用共享QA训练循环，**同一个优化器**步数2→3→4→5，未重置状态。[CPU_SELFCHECK.json](CPU_SELFCHECK.json)
- check：实际exit0，约7.7秒；数据/代码/协议/QA复用文件哈希仍一致。

## 后续显式命令与资源

已执行的是`prepare`、`cpu-test`、`check`。未来获得GPU调度后才执行以下入口，任何命令均不会自动串联下一阶段：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_local_pair_transfer_v1.py gpu-smoke
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_local_pair_transfer_v1.py train --variant paired_bce
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_local_pair_transfer_v1.py train --variant paired_rank
```

GPU自检使用最长完整pair、生产两图路径和λ1，检查两侧重复差≤2e-6、有限非零梯度、BF16前向及非重入checkpoint重算、FP32参数/梯度/Adam状态。没有通过自检无法训练；失败记录保留，不截断、不降精度门槛、不改超参、不自动覆盖或恢复。

两个模式共保留8份完整model/optimizer/RNG checkpoint，按现有同模型文件约 **14.36GB**，另加分数和日志；准备时可用空间118.39GB。GPU时间暂估每模式2–5小时，属于规划区间；最长两图是否适合8GB及实际速度必须等自检，不提前保证。[WEIGHTS_AND_RESOURCE_PLAN.json](WEIGHTS_AND_RESOURCE_PLAN.json)

所有旧源码、原QA文件、分词阶段文件保持不变；未准备额外36组合或新增超参。官方test未打开。
