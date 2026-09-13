本模块提供可审计的本地强基线，**尚未正式拟合或评测**。官方实现于本次在线核对；下面明确区分公式保留与实验适配，不将 R19 四段压缩称为完整 ReDeEP，也不将 MMD 单信号称为完整 LUMINA。

| 方法 | 输入与候选 | 冻结后名称 |
|---|---|---|
| Lookback | 4 原始 BPE 内每头比值均值，784维；LR C={.001,.01,.1} | Lookback Lens，本地 Qwen/4BPE 适配 |
| ReDeEP | 完整 ECS784 / PKS28；Kh={1,4,16} × Kl={4,14,28} × β={.2,.6,1}，27个公式候选 | ReDeEP-token 信号，Qwen/标准JSD/4BPE 适配 |
| slots 强本地参考 | R18 已冻结的4×785+mask头、原参数原阈值 | 原 slots，不冒充官方方法 |
| LUMINA | 必须有逐token IPR 和兼容 MMD；当前缺 IPR | 不纳入完整方法成绩，待新增提取 |

**官方依据与本地差别。**

Lookback 官方先对资料侧及生成侧注意力分别求均值，再取前者除以两者之和；检测区间内对每个头取时间均值，训练逻辑回归。官方代码支持逐token及滑动窗口，并非只能整答判断。我们使用已缓存的 Qwen post-read 状态、4BPE窗口、精确原token坐标、事件组划分及加权标准化，标签正类统一为风险；这些不是原 Llama 预训练检测器的原样运行。来源：[比值提取](https://github.com/voidism/Lookback-Lens/blob/main/step01_extract_attns.py)、[窗口与LR](https://github.com/voidism/Lookback-Lens/blob/main/step03_lookback_lens.py)。

ReDeEP 官方 `token_level_reg.py` 的实际排序返回值是 AUC：ECS 与 1−y、PKS 与 y 对应，按降序选头/层，各自求和、MinMax，风险方向为 scaledPKS − β×scaledECS。不要把源码中另算的 Pearson 当实际返回排序。我们在拟合组按基础组权重计算 AUC，从全部784头及28层选取；27组参数只在校准组比较。常数列AUC=.5，同分按原列号。MinMax只用拟合样本，未见样本不重新缩放、不裁剪到[0,1]；原范围为0时除数取1。原脚本按回答均值汇总，而本任务按窗口/整答max汇总，这是明确适配。来源：[官方排序与组合](https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/main/ReDeEP/token_level_reg.py)。

已有 ECS 是各层各头对注意力最高10%资料位置的最终表示与回答表示的余弦；PKS 是 FFN 前后残差经过最终norm/词表投影的标准 JSD。官方旧提取代码保留了反向 KL 写法及额外数值倍率，注释也链接了问题讨论；本地此前采用标准 JSD，不能承诺原代码逐数值重现。R19原始列按层优先、头其次保存，选头前没有四段压缩。来源：[官方提取](https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/main/ReDeEP/token_level_detect.py)、[JSD问题](https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2)。

LUMINA 默认 score=.5×IPR−.5×MMD。IPR需要各层词表分布、层深度/熵项及最终token概率信息；最后一层hidden、NLL、PKS均不能直接当IPR。MMD采用top100未重新归一化概率与余弦核。已逐文件确认R19共602行、14968个原token，只有 MMD[N,2]、ECS[N,784]、PKS[N,28]与坐标，没有任何IPR数组；对应manifest SHA256为 `bcc79f15c60f05830570be635c9f5a261a0eddfa20b3f006d1f393254a7d8ab1`。即使未来补出IPR，R19逐来源等长替换与官方整份随机资料干预仍不同，组合必须注明本地扰动适配。来源：[官方LUMINA实现](https://github.com/deeplearning-wisc/LUMINA/blob/main/lumina.py)。

**拟合与选择契约。** 模块不读文件，不接收评测标签。调用者先由冻结划分准备：`fit_rows`仅可评拟合窗口；`fit_lb/ecs/pks`与它们逐行对应。基础权重保持每事件组、条件、回答、窗口层级平衡；标准化及AUC用基础权重，LR损失另加拟合类别因子并按组归一，总量3854。默认liblinear、L2、seed20260913、最多2000次，CPU最多4线程；实际配置需进R21协议。

每个候选先独立在校准集选窗口阈值与整答max阈值，各自最大化风险F1，阈值平手依次看精确率及更高阈值。候选选择统一最大化 `min(window_F1, answer_F1)`，再看窗口F1、窗口精确率、较小复杂度/较小C。ReDeEP复杂度预定按 `(Kh+Kl, Kh, Kl, β)` 字典序，较小优先。完整校准表保存；选中后不在fit+cal上重拟合，不再用外层结果换参数。不得仅汇报某个粒度最好的不同候选却说是同一个模型。

校准窗口评分包含全部输出可见的候选窗口，包括安全拒答；定位指标只取原可评断言，安全拒答只作为整答负类；未决覆盖单列。整答max不能先用金标筛窗口。`select_calibration`检查拟合/校准事件组不重叠，并拒绝缺失预测；评测数据只在选择冻结后交给`predict`。ReDeEP分数可能超出[0,1]，它不是概率。

**最小调用示例。** 调用者负责哈希冻结、候选窗BPE几何、外层标签隔离、模型pickle和逐项分数落盘。

```python
import baselines21 as b

# 以下是API说明，尚未在真实数据上执行。
candidates = b.fit_lookback_candidates(fit_features['lb'], fit_windows)
lb = b.select_calibration(candidates, cal_features, cal_windows, cal_answers)

candidates = b.fit_redeep_candidates(
    fit_features['ecs'], fit_features['pks'], fit_windows)
redeep = b.select_calibration(candidates, cal_features, cal_windows, cal_answers)

# 先保存选中对象、两个阈值、完整校准表及其哈希，再单次评测。
scores = b.predict(redeep, evaluation_features)
window_alert = scores >= redeep['thresholds']['window']['threshold']
# 整答风险另按所有候选窗口max与answer阈值比较。
```

`python src/baselines21.py` 只作合成算术/选择检查，不拟合LR、不打开真实数据、不调用GPU。实际正式调用须由已冻结的R21协议统一启动。比现有四段信号更充分使用原始信息，是合理强基线；它不保证高于既有模型，更不能因结果不佳而继续在外层扫描参数。
