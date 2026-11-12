# 可复用输入数据

- `raw/`：保持原样的输入；纯外部数据发布在 `raw/external/`。
- `processed/`：清洗、标注、特征等派生输入；记录对应原始来源与处理配置。
- `splits/`：冻结的数据划分及样本 ID。

一次运行生成的轨迹、预测、模型和报告属于 `experiments/<主题>/runs/<运行ID>/`，不要再复制到这里。
现有自包含实验包保持内部布局不变；只有明确供多次实验复用的数据才进入本目录。

上游代码与数据混合的仓库完整放在 [code/external/](../code/external/README.md)。
早期预实验数据和说明已随旧系统迁入 [archive/early_pre_exp1/data/](../archive/early_pre_exp1/data/)。

原始与派生数据默认不纳入 Git；可公开的划分按许可与隐私要求审查，不能包含 API 凭据。
