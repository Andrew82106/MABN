# 代码

- `src/paperalpha_monitor/`：知识编译、风险推断与解释。
- `src/paperalpha_runtime/`：观测记录、API 接入、适配与命令入口。
- `commands/`：数据准备、采集、训练、评测与审计命令。
- `tests/`：当前 MAS 实现的测试；命令专用测试在 `tests/commands/`。
- [external/](external/README.md)：第三方作者仓库，保留原版本，不混入自有算法。
- `workspace_paths.json`：旧路径到新位置的唯一机器映射。

当前评测从 [实验协议](../experiments/protocol.json) 和[复现说明](../docs/usage/REPRODUCIBILITY.md)进入。
`commands/maintenance/` 中的日期迁移脚本只作历史记录，不要重复执行。
历史冻结源码仍在各实验运行包里，不能用当前源码替代后沿用旧哈希。
