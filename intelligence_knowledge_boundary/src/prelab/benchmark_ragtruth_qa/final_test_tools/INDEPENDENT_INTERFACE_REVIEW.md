v1 校准数值核对通过，但正式解封前需补齐依赖绑定。

发现：原始文件预期哈希取自 development_manifest，tokenizer 预期取自 feature_preparation/manifest；v1 未直接冻结校验两文件，也未实际遍历 gold manifest 内的传递哈希。父代理将另立 v2 增加两 manifest 和 plans 的显式绑定，保留 v1。

已独立复核159答、42,798原token、42,241可评窗及80纯非词窗；首token跨界、字符与4 raw BPE标签、固定分数和整答max均正确。点指标最大差2.34e-15；154组×5,000次bootstrap区间精确一致。

实际推理适配器仍须另行审查冻结。本次未调用release、读取或哈希真实raw test，未训练或使用GPU。
