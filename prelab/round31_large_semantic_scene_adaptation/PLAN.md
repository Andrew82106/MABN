# R31：large骨干的场景适配对照

只使用已反复开发的R16原train：602答、301问题、278事件组；`human_gold=false`。保留原BPE、字符映射、12,222候选窗口、9,526可评窗口和598可评整答。安全拒答仍是整答负例，未决不补零。原R16 validation/test与公共QA test均不打开。

骨干固定为公共QA ModernBERT-large六轮完整结束后，原规则已经选中的唯一轮次。R31不选择上游轮次。CPU设计不读取任何未完成轮次或预测。GPU必须等待前任务实际退出，并由root另外授权。

一次离线完整资料/问题/回答前向，分别保存原BPE映射的最终hidden1024、实际分类logit差与GPU概率。1024维隐藏均值使用窗口全部原BPE；logit最大值仅使用原lexical槽。合成1025维，每折一个LR，固定C=.01、原seed、原fit-only scaler及3854损失总质量。两个原基线分别使用六个既定alpha；阈值和alpha只在各折cal选，五折全部冻结后才合并eval。

R29保存的模型、五折分数和选择均按哈希保留，不重训。比较同时改变骨干、维度及上游训练方式：R29为base辅助→QA，R31为large QA-only。不能将差异单独归因于模型大小、维度或某个特征，也不能称为独立人标验证。

必要检查：base/large分词器文件逐字节一致；原输入和坐标不变；分组隔离；fit外NaN不影响scaler；原始窗口均值与lexical logit最大值；alpha0精确回到基线；缺complete不加载模型/GPU、不拟合。GPU阶段固定检查原QA第一个fit答的旧概率，以及场景首末答的无hook logit和重复hidden/logit，绝对容差2e-6。没有旧large场景缓存，不能声称与旧aux概率相等。

入口（项目根目录，Python为`prelab/.venv/Scripts/python.exe`）：

```text
prelab/round31_large_semantic_scene_adaptation/src/run31.py design
```

以上CPU准备已完成。后续命令均未执行，需要按阶段授权：

```text
prelab/round31_large_semantic_scene_adaptation/src/extract31.py infer
prelab/round31_large_semantic_scene_adaptation/src/run31.py prepare
prelab/round31_large_semantic_scene_adaptation/src/run31.py fit
prelab/round31_large_semantic_scene_adaptation/src/run31.py evaluate
```

`infer`完成实际退出后即可释放GPU，其余均CPU四线程。各阶段拒绝覆盖已有正式产物；不自动恢复、不变更超参数。
