# 环境与复现

所有命令在 `paperAlpha/` 根目录执行。主设置以 [experiments/protocol.json](../../experiments/protocol.json) 为准。

## 本机测试

继续使用现有 Python 3.11.15，未新建环境。基础环境没有 `agents` 模块；完整采集相关测试需要已有的 OpenAI Agents SDK 依赖。可在当前 PowerShell 进程中临时借用原采集环境的包目录：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = (Join-Path $PWD 'code/src') + ';' + (Join-Path $PWD '.local/environments/collection/harnessaudit_oai_20260919/Lib/site-packages')
& 'D:\anaconda\envs\multi_agent_graph\python.exe' -X utf8 -B -m pytest -q
```

这只适用于本机已有依赖，不是跨机器安装方式。测试日志与实际验收结果见[迁移验收](STRUCTURE_MIGRATION_20261014.md)，不能将缺依赖的测试记成通过。

其他机器使用 Python 3.11+ 安装当前源码：

```powershell
python -m pip install -e ".[test,collection]"
python -X utf8 -m pytest -q code/tests/test_package_entrypoints.py code/tests/test_workspace_layout.py
paperalpha-evaluate --help
```

上述命令检查安装与目录接口。完整测试还依赖本地第三方仓库、采集和基线环境，须按[外部资产说明](EXTERNAL_ASSETS.md)恢复后执行全量 pytest；仅克隆源码不保证这些集成条件齐备。

核心计算依赖见 [pyproject.toml](../../pyproject.toml)。外部基线按各自原仓库准备；主模型不要求 GPU 或调用在线 API。原生 QuadSentinel 测试在独立进程中执行，避免其依赖与采集 SDK 在同一进程里混用。

默认测试只覆盖 `code/tests/`。早期预实验已单独归档，运行方式见 [archive/early_pre_exp1/README.md](../../archive/early_pre_exp1/README.md)。

## 新建主基准复现

复现脚本强制指定新的运行 ID，拒绝覆盖已有目录：

```powershell
./code/commands/reproduce_main.ps1 -RunId my_fresh_reproduction
```

结果在 `experiments/synthetic_mas/runs/my_fresh_reproduction/`。脚本按原设置生成 4,000 条合成轨迹，再运行五折评测；没有调用真实 API。这是开发基准复现，不补足独立人工确认集。其他机器通过 `-Python` 指定解释器。

`paperalpha-evaluate independent`、`binding`、`costs` 也可从源码工作区运行。
核心 wheel 包含模型与静态知识，可在工作区外导入；整套评测命令依赖 `code/commands/`，不能仅靠核心 wheel 复现。

## 数据与历史记录

- 实验位置见 [experiments/README.md](../../experiments/README.md)，旧路径用 [resolve_historical_path.py](../../code/commands/resolve_historical_path.py) 查询。
- 第三方资产的本地分区见[外部资产说明](EXTERNAL_ASSETS.md)。恢复时遵循原始 revision/hash，不用最新版替换后沿用旧结果。
- `.local/environments/` 保留原环境文件，可能含旧绝对路径。移动目录不等于保证环境中的所有启动器可移植。
- 原始轨迹、失败记录和大部分运行包不进入 Git。完整档案须另作授权备份。
- 凭据只通过本机环境提供，不公开 `.env`、认证头或未经审查的完整轨迹。
