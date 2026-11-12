# 项目结构

当前根目录只有七类内容，各有一个入口。

```text
paperAlpha/
├── docs/                 说明与阅读材料
│   ├── design/           方法设计
│   ├── references/       文献、调研、既有论文
│   ├── manuscript/       论文汇报与发布清单
│   └── usage/            使用与迁移说明
├── knowledge/            机器读取的知识
│   ├── risks/
│   ├── templates/
│   └── schemas/
├── code/
│   ├── src/              两个 Python 模块
│   ├── commands/         命令与评测脚本
│   ├── tests/
│   └── external/         第三方实现
├── data/                 可复用输入
│   ├── raw/
│   ├── processed/
│   └── splits/
├── experiments/
│   ├── protocol.json     唯一主协议
│   ├── baselines.json    当前基线清单
│   ├── index.json        实验索引
│   └── <主题>/
│       ├── configs/      该类实验的设置（有配置时建立）
│       ├── notes/        实验分析（有分析时建立）
│       └── runs/<ID>/    一次运行的完整证据包
├── archive/              已退役的整套系统
└── .local/               本机环境、缓存与备份
```

根级 `README.md`、`AGENTS.md`、`pyproject.toml`、环境与 Git 配置是项目入口和安装配置，不再另建同用途文件夹。

## 内容边界

1. `docs/` 解释方法；`knowledge/` 是算法实际读取的定义。
2. `data/` 存可复用输入；`runs/<ID>/` 保留那次实验的全部输入、输出、冻结源码和日志，不按文件类型拆散。
3. `archive/` 收退役系统；普通旧运行留在原实验主题，不因日期早就归档。
4. 当前算法只在 `code/src/` 维护；运行包中的源码只是不可改写的历史证据。
5. 上游代码与数据混合的仓库整包保留在 `code/external/`，不复制出第二份数据。

## 旧位置到新位置

| 旧入口 | 新入口 |
| --- | --- |
| `literature/` | `docs/references/` |
| `research/` | 设计进 `docs/design/`，实验分析进对应 `notes/`，可执行定义进 `knowledge/` |
| `configs/` | 通用模板进 `knowledge/templates/`；实验设置进对应 `configs/` |
| `src/`、`scripts/`、`tests/` | `code/src/`、`code/commands/`、`code/tests/` |
| `submission/` | 写作进 `docs/manuscript/`，协议/基线进 `experiments/`，复现命令进 `code/commands/` |
| `results/` | 各主题的 `experiments/*/runs/` |
| `pre_exp1/` 与其数据 | `archive/early_pre_exp1/` |
| `.venv-baselines/`、运行环境、整理备份 | `.local/` |
| `data/external/` | 作者仓库进 `code/external/`；纯数据进 `data/raw/external/` |

完整映射只维护在 [code/workspace_paths.json](../../code/workspace_paths.json)。
在项目根目录可查询旧路径：

```powershell
& 'D:\anaconda\envs\multi_agent_graph\python.exe' -B code/commands/resolve_historical_path.py 'results/independent_mas_journal_v1/REPORT.md'
```

历史 manifest 和冻结源码中的旧路径不改写。查询只帮助定位，不会修改原记录或绕过其哈希检查。原冻结脚本如果硬编码旧绝对路径，需要在另一个工作副本还原布局；不能直接宣称移动后全部历史命令均可运行。

本机资产完整保留，不意味着克隆 Git 就包含原始数据、环境、模型和全部历史运行。分享范围见 [PACKAGE_MANIFEST.json](../manuscript/PACKAGE_MANIFEST.json)，复现条件见[环境说明](REPRODUCIBILITY.md)。
