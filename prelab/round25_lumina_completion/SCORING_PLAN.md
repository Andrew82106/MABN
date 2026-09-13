# R25 固定公式评分接口

本阶段只新增 CPU 评分代码和 `scoring_protocol.json`，不改已冻结的提取 `protocol.json`、`extract25.py`、联合计划或源数据。没有新 LR、PCA、回答生成或 GPU 调用。

方法共十项：R22 六个冻结旧基线、R24 `slots_base_smooth`，以及：

- `lumina_single_mean_lambda05`：固定 lambda=.5，两个 R19 单来源 MMD 同权均值。
- `lumina_joint_body_lambda05`：固定 lambda=.5，双正文联合 MMD；预定主方法。
- `lumina_joint_body_lambda_cal`：仅 calibration 在 {.25,.5,.75} 中选择；独立命名的次要对照。

窗口先对原 float32 token 公式得分按原 4 raw BPE 取算术均值，再导出 float64。标点仍占 BPE 位置，原短窗口规则不变。整答分数取所有原候选窗口的最大值；不先按风险、可评性或拒答筛窗口。安全拒答按原规则作为负整答，窗口不进定位分母；未决仍排除。

实际阈值 helper 支持负数，端点为 nextafter(实际最大值,+∞) 与 nextafter(实际最小值,−∞)。所以不需要 expit、minmax 或截断，保留原有符号 LUMINA 排序分数，不称为校准概率。

五折事件分组完全沿用旧 assignment。每折只用 calibration 的 window/answer F1 设各自阈值，优先 F1、precision、更高 cutoff。次要 lambda 依次比较 min(window F1,answer F1)、window F1、window precision、优先 .5、再较小 lambda。外层标签不参与选择；全部五折的阈值、lambda、候选表及全量分数文件先冻结，之后独立 `test` 入口才计算外层指标。

## 命令

```powershell
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/run25.py initialize
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/run25.py baseline-check
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/run25.py status
```

新特征完整并通过哈希与坐标检查后：

```powershell
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/run25.py calibrate
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/run25.py test
```

`fit` 只是 `calibrate` 的兼容别名，没有模型拟合。特征缺任一份即停止，不能用已完成子集先报结果。原 R16 validation/test 内容不解析；仅 actual-train 源记录、train 标签及固定旧结果用于该 runner。

## 已完成检查

- 有符号阈值、负整答 max、包含不可评窗口的整答 max、4 BPE/短窗口均值、lambda 并列选择规则 CPU 自检通过。
- R22 六基线手工按冻结系数/公式回放，五折每个分数都完全一致；R24 平滑按冻结 stay/temperature 回放完全一致。
- 七旧基线原 calibration 阈值复算完全一致。记录为 `results/baseline_replay_check25.json`。
- 真正的新 LUMINA 校准和 outer 尚未开始。后续 `test` 会再次检查七个旧基线汇总与原报告完全一致、每回答/每窗口 OOF 恰好一次。

产出为 `results/calibration_freeze25.json`、`fold_*_calibration25.json`、`fold_*_scores25.npz`，随后 `summary25.json`、`window_scores_oof25.jsonl`、`answer_scores_oof25.jsonl`、`complete25.json`。仍是反复开发的助手标注资料支持数据，不是新独立测试。
