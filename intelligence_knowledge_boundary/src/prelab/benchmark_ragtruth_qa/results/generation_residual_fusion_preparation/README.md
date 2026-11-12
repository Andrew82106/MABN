# 原生成信号残差融合：仅模块准备

目的：在现有 FullContext ModernBERT 的风险分数上增加一项可学习修正，原基线仍单独保留比较。目前没有训练、效果结果或真实测试访问。

每个原 Llama token 输入：语义末层状态 `h[768]`、原基线风险 logit `z`、Lookback `lb[1024]` 和 `nll[1]`。768 维状态与原基线 logit 要分别使用既定非空白字符重叠映射对齐，不能从平均后的 hidden 重新计算 logit 来替换原基线。

固定结构（无宽度或层数搜索，共 52,940 个参数）：

- 语义路：`s = SiLU(Linear32(LayerNorm768(h)))`。
- 生成路：`a = SiLU(Linear24(LayerNorm1024(lb)))`；`b = SiLU(Linear8(log1p(nll)))`；拼成 32 维。NLL 非负，单独处理，不与 1024 个 LB 一起归一化。
- 拼接 `u = [s, a, b, z]`，共 65 维；`gate = sigmoid(Linear1(u))`。
- 输出 `z_new = z + gate × Linear1_zero(u)`。最后一层权重与偏置均零初始化，因此初始化时严格等于原 logit。模块不做 sigmoid 概率转换、窗口汇总、掩码或阈值选择，这些沿用调用方原口径。

两路输入不是“两个原模型注意力头”：LB 是同一 Llama 的实际 32 层 × 32 头读资料比例；语义状态来自额外的 ModernBERT。ModernBERT 阅读资料、问题和完整回答，整个方案仍是离线检查，不能声称生成到当前 token 时即可获得其语义状态。其他生成器的答案经 Llama 统一重放，也不是恢复其原生生成轨迹。

CPU 合成检查通过：单答及批量形状、零修正精确等价；测试性设置非零读出后，语义、LB、NLL 和基线输入都能改变输出，所有分支梯度有限且非零。零初始化时第一步只有最后读出层先获得非零梯度，这是预期行为；读出离开零后上游分支可学习。检查没有优化器、训练数据或模型权重保存，不证明检测效果会提高。

代码：`src/generation_residual_fusion.py`；核查记录：本目录 `CPU_SELFCHECK.json`。现有冻结基线脚本未改。后续训练、特征对齐接线及是否冻结语义编码器尚未由本模块决定。

补充接线已实现于 `src/modernbert_generation_bridge.py`：通过临时hook取得同一次原生ModernBERT前向的末层状态，分别映射真实logit与hidden，再接融合头。CPU真实tiny ModernBERT检查中，原基线分数与零残差输出逐值相同，字符映射和骨干/残差读出的反向梯度通过，hook正常移除；见 `BRIDGE_CPU_SELFCHECK.json`。仍未加载正式权重或进行此融合模型训练，也未决定后续骨干是否冻结。
