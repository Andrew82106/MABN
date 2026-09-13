# 已完成large及公平组合错误诊断

仅使用已选模型及其原阈值。反复开发校准159答；不训练、不调阈值、不读取官方测试。

| 方法 | 定位F1 | 整答F1 | 正常答内误报窗 | 风险答内误报窗 |
|---|---:|---:|---:|---:|
| generic_base | 0.641796 | 0.829493 | 261 | 1619 |
| generic_large | 0.672782 | 0.864078 | 256 | 1718 |
| lookback__large_lr | 0.672237 | 0.860000 | 256 | 1706 |
| lookback__large_tree | 0.670858 | 0.858537 | 279 | 1771 |
| harp_claim__large_lr | 0.673836 | 0.857143 | 277 | 1816 |
| harp_claim__large_tree | 0.670858 | 0.858537 | 279 | 1771 |
| semantic_claim__large_lr | 0.674838 | 0.859813 | 272 | 1799 |
| semantic_claim__large_tree | 0.670858 | 0.858537 | 279 | 1771 |

| 类型 | 原风险窗 | base检出 | large检出 |
|---|---:|---:|---:|
| Evident Baseless Info | 4086 | 2986 | 3180 |
| Subtle Baseless Info | 814 | 543 | 578 |
| Evident Conflict | 997 | 149 | 211 |
| Subtle Conflict | 109 | 48 | 85 |

类型可能重叠，只报告召回，不将其他风险类型充作负类。训练曲线和全部六种组合的类型计数保存在summary.json。
base和large同时改变尺寸及AdamW foreach执行方式；组合还包含其他语义/生成信号，不能视为单一因素因果结论。
