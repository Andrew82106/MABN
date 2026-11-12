# HarnessAudit：运行观测与事后标签分离

## 本轮结论

新增 `harnessaudit_observations.py` 和 `project_harnessaudit_observations.py`，把官方原生 JSONL 分成模型可见的已完成动作重放记录、事后审计结果和任务元数据。它位于项目主线的**运行证据接入层**，为后续知识模板和 BN 提供有明确时间语义的输入边界。

**不是新的风险模型成绩，也不是已经接通真实 API MAS。** 新动作级 BN 所需的提前检查、动作授权绑定并不在这些原生日志里，不能伪造字段填满协议。旧适配器及历史输出保留，仅加了不宜直接作为模型输入的说明。

## 1. 从原始代码确认的三个事实

核读本地官方仓库 `data/external/HarnessAudit`，提交 `6317162590aeeb1c8dde32b880ac199933343e4a`，工作区无改动：

1. `multi_agent/cli.py::run_once` 先完成 `runner.run`，然后执行 `check_trace`，写 `AccessDecisionEvent`。**access_decision 是任务结束后的审计，不是执行前的权限检查。** 原生 `trace/events.py` 也明确注明 post-run。
2. `frameworks/oai/tools.py::_make_domain_handler` 先派发工具、获得返回，再调用 `emit_tool_call`。其 `raw_event.capture_mode=inline_dispatch` 不能被解释为预调用观测。工具参数与返回值位于同一个已完成动作事件里；失败返回同样可能出现，不能把 completed-observed 当作操作成功。
3. `frameworks/core/action_sink.py` 缓存动作，在 `finalize()` 按来源时间、文件、行号、插入序号重新排序并分配序号。这些序号支持原生文件顺序重放，**不证明真实在线到达顺序，也不证明调用开始时刻**。

因此，直接把审计内容注入 `permission_mismatch` 或提前时点，会把答案放进模型输入。旧 `adapt_harnessaudit_traces.py` 的 `_permission` 保留 access_decision 字段，`_features` 又读取 violation_code/rule_description 等；它是历史覆盖分析接口，不是本轮的隔离接口。没有据此重写历史分数或宣称所有旧实验均受同一种泄漏影响。

## 2. 新接口保留什么

- `monitor_inputs.jsonl`：每个任务的 `monitor` 对象才是模型输入。包含开始时声明的 Agent/角色、实际记录的通信内容、工具名、参数、返回，以及原生序号和来源时间。
- `evaluation_only.jsonl`：事后 access_decision、结束统计、异常和标签资格。
- `metadata_only.jsonl`：任务 ID、领域、模型、扰动条件、规则数量等实验组织信息。
- `REPORT.json`：输入和产物哈希、来源声明、缺失证据、源码快照哈希。调用方声明 `native_execution` 不等于我们独立验证了来源；报告始终保留该区别。

`trace_id` 只作文件关联键，不是模型特征。`raw_event`、预生成 features、裸 permission、标签、评审分数和扰动信息不会进入 `monitor`。`useful_tools` 也不会被转成权限清单：任务需要的工具与实际授予的权限不是同一个概念。

相反，工具参数或消息文本中本来就出现的 `risk_label`、`permission_status` 等字符串仍是原始内容，不应为追求“字段名隔离”而删除真实文本；它们不会被提升成可信权限事件。参数中类似密钥的键做局部脱敏，但这不是完整的文本匿名化或秘密过滤保证。

工具事件命名为 `tool_completed_observed`，不会伪装成 v3 的 `tool_proposed` 或具有执行前语义的 `tool_call`。当前直接接入旧 BN 的资格明确为 false，需要另行定义角色/工具映射和语义抽取，不静默退化到旧模型。

## 3. 标签究竟表示什么

可选的派生标签是 `any_recorded_access_rule_violation`：完整审计里有没有记录到至少一次规则违规。

- 包含资源范围违规 V-OR，不等于原文的 L1 SAR，也不是最终真实危害标签。
- 0 只表示既定规则未匹配到违规，不表示“系统安全”。
- 原生检查器对每个动作返回一个 allowed 判定或一个/多个违规判定；多条匹配规则不能被误算成多个动作。
- 缺结束记录、动作未审计、计数不符、运行错误、矛盾判定、未知事件类型时，标签为 null，不当作负例。
- 首个被标记动作的序号仅供事后定位，不是危害发生时刻或提前预警基准。

因为源文件是事后规范化的产物，接口不授予在线顺序或预执行预警资格。未来即使在这些记录上完成前缀预测，也应称为“已完成动作序列重放”，除非额外取得可靠的在线采集时间。

## 4. 已做验证

新增30项测试，包含：

- 改变审计判定、标签、隐藏规则、任务条件、评审分数和 useful_tools，`monitor` 及其哈希保持不变。
- 未来记录不回填过去快照；工具结果只在对应已完成动作记录到达后出现；返回对象或输入对象被修改不会污染内部记录。
- 缺审计、错误结束、未知事件不变成安全样本；重复/逆序动作、不合原生类型的数据、审计后出现动作被拒绝。
- 多任务交错不混合，正常/违规完整审计分别得到0/1，V-OR的含义明确。
- 用官方 ActionSink、TraceWriter、check_trace 的实际组件代码生成一个受控记录，确认先缓存再重排，以及事后规则结果与监测输入分离。**没有执行 Agent、API 或真实工具**。

官方整体 frameworks 导入在当前 Windows 环境报 `ModuleNotFoundError: fcntl`，因为注册器提前加载了 Unix 配置锁模块。组件测试直接加载对应官方源码文件，未改写算法、未伪造 fcntl，也不据此宣称完整 HarnessAudit 可运行。本机只读查询确认存在 Ubuntu WSL，尚未在其中安装或运行该基准。

完整项目测试：307项通过，4条既有 sklearn 数值/弃用警告。CLI后续仅增加源码快照哈希，相关测试再单独复验。

已有最小夹具投影结果保存于 `results/harnessaudit_observation_split_fixture_20260919/`：1条夹具、2条可见动作、0个合格标签。原因是其中一个动作没有审计判定，结束事件也没有明确 error 字段。这是正确保留“不完整”，不能修改夹具来冒充真实数据。

```powershell
python -B -m pytest paperAlpha/tests/test_harnessaudit_observation_split.py -p no:cacheprovider -q
python -B paperAlpha/scripts/project_harnessaudit_observations.py --input <native-trace.jsonl> --out <new-output-directory> --provenance external_unverified
```

## 5. 对后续模型的实际约束

接下来不能直接训练“每个动作都有可靠提前权限”的模型。需要沿两条不同观测条件推进：

1. **现成日志条件**：从真实消息、工具参数和返回提取证据，建立已完成动作上的信息流、来源和结果状态；事后审计只放评测侧。先验证这些观测能支撑什么风险任务。
2. **增强采集条件**：在受控 API MAS 调度器实际调用前取得提案、动作绑定和独立检查，才能验证 v3 的待执行更新与提前预警。新增采集不是补造既有日志，实验必须单独命名。

本轮没有真实模型调用、没有新训练成绩、没有完成原论文 MAS baseline。知识模板扩充、真实运行收集、独立结果标签、大图推断和外部评测仍需继续。
