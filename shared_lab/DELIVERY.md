# 共享实验平台交付说明

状态：等待主 Agent 验收。

## 实际环境

- 开发/主测试解释器：`C:\Users\贾济铭\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Python：3.12.13（代码声明并保持 Python 3.11+ 兼容）
- 交叉测试解释器：`D:\anaconda\python.exe`
- Python：3.13.5
- 任务书指定的 `D:\anaconda\envs\multi_agent_graph\python.exe` 在当前机器不存在，因此未能在精确的 3.11.15 环境复跑。
- GPU、模型与网络：测试均未使用。

## 依赖清单

- 运行时第三方依赖：无，仅 Python 标准库。
- 构建依赖：`setuptools>=68`，仅用于可编辑安装/打包。
- 测试框架：标准库 `unittest`，无新增测试依赖。

## 测试与验收证据

公开测试命令：

```powershell
cd shared_lab
python -m unittest discover -s tests -v
```

测试数：19。Python 3.12.13 与 Python 3.13.5 各运行一遍，均为 `Ran 19 tests ... OK`。另执行 `pip wheel --no-deps --no-build-isolation`，成功构建 `lab_kernel-0.1.0-py3-none-any.whl`，随后清理构建产物。

1. 场景解耦：`test_kernel_source_has_no_first_scenario_constants` 扫描内核源码；场景数据仅位于 `scenarios/`。
2. 全离线 stub：`test_offline_stub_scheduler_and_deterministic_replay`。
3. 追加账本与回放：`test_append_only_ledger_refuses_reopen`、`test_offline_stub_scheduler_and_deterministic_replay`、`test_replay_rejects_state_hash_tampering`。
4. 权限前置与拒绝留痕：`test_permission_denial_is_recorded_before_execution`、`test_message_must_use_live_configured_edge`、`test_unauthorized_defense_is_denied`。
5. 配置/代码/环境溯源：`test_receipt_verifies_config_code_environment_and_ledger`。
6. 篡改 fail-closed：`test_ledger_tampering_fails_even_if_event_hash_is_recomputed`、`test_configuration_tampering_fails_against_receipt`。
7. 首个场景与程序判定：`test_scenario_is_declarative_and_programmatically_evaluated`。
8. 三类钩子：`test_message_gate_replace_drop_and_safe_rewrite`、`test_observation_noise_missing_and_false_label`、`test_all_defense_actions_are_permission_checked_and_replayable`。
9. `paperAlpha/` 字节不变：实施前后对 653 个文件按相对路径排序并计算逐文件 SHA-256 后再聚合；基线与最终值均为 `90372c7a53158b1745dfab0ade03dcffc0cbf9a7c73c6b2f0caf5881e34c6c73`。`git diff HEAD -- paperAlpha doc README.md .gitignore` 为空。

## 未实现项与已知限制

- 未集成真实模型适配器；这是刻意保留的场景/部署层扩展点，不影响离线内核验收。
- 未集成 AutoGen 或 LangGraph；标准库薄内核已经覆盖本任务的审计性质，避免增加运行时依赖。
- 收据是内容寻址的可信锚，不是数字签名。若攻击者能同时替换账本、配置、代码和调用方保存的原始收据，则纯本地哈希无法证明外部真实性；生产归档应把收据哈希写入独立只读存储或使用组织签名。
- 精确 Python 3.11.15 解释器在当前环境不可用，等待验收环境复跑。

本文不包含论文结论；等待主 Agent 验收。
