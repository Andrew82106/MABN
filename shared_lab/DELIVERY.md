# 共享实验平台权限映射返工交付说明

状态：主 Agent 已于 2026-08-18 独立验收通过。

独立验收使用 `D:\anaconda\envs\multi_agent_graph\python.exe`（Python 3.11.15）
复跑 22 项测试全部通过，并额外复验：`state.write` 伪装执行 `isolate_agent`
被拒且状态不变；正确 `defense.isolate` 仍可执行；未知 action kind 被拒；空 canary
被拒。功能验收与整体交付均通过。

## 实际环境与依赖

- Conda 环境：`multi_agent_graph`
- 解释器：`D:\anaconda\envs\multi_agent_graph\python.exe`
- Python：`3.11.15`
- 运行时第三方依赖：无，仅 Python 标准库。
- 测试框架：标准库 `unittest`；未安装任何新依赖。
- GPU、模型与网络：全部未使用。

## 本轮返工

### 内核强制 kind–capability 绑定

映射由 `src/lab_kernel/permissions.py` 内核内置，并使用只读
`MappingProxyType` 保存。当前所有可执行 action kind 的唯一合法映射为：

| action kind | required capability |
|---|---|
| `set_value` | `state.write` |
| `append_value` | `state.write` |
| `send_message` | `message.send` |
| `isolate_agent` | `defense.isolate` |
| `cut_edge` | `defense.cut_edge` |
| `revoke_permission` | `defense.revoke` |
| `rollback_state` | `defense.rollback` |

`Kernel.dispatch()` 在 hook 前的首次授权及 hook 后复检两处都调用
`PermissionEnforcer.require_action()`。内核先按 `kind` 取得必需 capability，再要求
调用者声明值与之严格相等，最后才检查该 actor 是否实际持有必需 capability。
未知 kind、声明不一致或 actor 缺权都会在任何状态转换前拒绝，并追加
`action_denied` 账本事件；调用者即使持有其他合法 capability 或通配能力，也不能用其
替代 kind 的必需 capability。

### 空 canary

供应商审查场景 evaluator 现在要求 canary 必须是非空字符串；空值或非字符串会直接
抛出 `ValueError`，不再触发空字符串包含判断恒真的误报。

## 测试结果

公开复跑命令：

```powershell
cd shared_lab
conda run --no-capture-output -n multi_agent_graph python -B -m unittest discover -s tests -v
```

结果：`Ran 22 tests ... OK`。

新增定向证据：

- `test_every_action_kind_rejects_other_owned_capability`：对全部 7 个 action kind，使用
  coordinator 已拥有的另一项合法 capability 伪装执行，全部拒绝、状态不变，且逐项
  产生 `action_denied`。
- `test_defense_kinds_cannot_be_spoofed_with_state_write`：明确验证隔离、切边、撤权、
  回滚四种防御动作均不能被 `state.write` 绕过，并核对拒绝账本顺序。
- `test_scenario_evaluator_rejects_empty_canary`：空 canary 明确报错。
- 原 19 项账本只增不改、确定回放、溯源收据、配置/代码/账本篡改 fail-closed、
  场景声明式加载、程序判定器与三个钩子测试全部继续通过，未重做其实现。

## 边界与已知限制

- 本轮只修改 `shared_lab/` 内核权限、场景 evaluator、测试和本交付说明。
- 未修改 `paperAlpha/`、`doc/`、根 README 或 `.gitignore`；工作树中既有的
  `paperAlpha` 删除状态不属于本任务，也未被触碰。
- 未接入真实模型、网络、GPU、AutoGen 或 LangGraph。
- 收据仍是内容寻址可信锚而非数字签名，此限制与首轮交付相同。

本文不包含论文结论；本次交付仅确认共享实验平台的工程基础满足任务书要求。
