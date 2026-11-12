v2 增量审查通过，v1 的依赖绑定缺口已补齐。

五个脚本逐字节不变；initialize 仅增加 development_manifest、feature_preparation/manifest、plans 三个哈希及版本号。实际11项哈希全部匹配。require_release 会在返回前验证这些哈希，因此上游期望文件变化会在读取或哈希 raw 文件前被拒绝。数据、gold与评分公式均未改变。

核对了现有159答/42,241窗模拟的完成绑定（报告点指标差0、整答max精确），未重复全量数值审计。仍须冻结并审查实际推理适配器和最终方法，获得明确授权后才能解封。本次未读取或哈希真实raw test。
