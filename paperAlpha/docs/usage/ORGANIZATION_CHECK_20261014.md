# 目录整理验收记录（2026-10-14）

结论：已识别的 MAS 正式资料、旧平台和外部参考仓库已收拢到本项目，原始实验记录未删除。本次验收只验证目录、文件和运行入口，不重新验收论文模型效果。

## 文件完整性

- 9 组目录搬迁、2 份共享规范快照，搬迁完成时核对 2,432 个文件（含外部 Git 元数据），全部 SHA-256 一致。
- 导航更新后再查，原清单没有缺失文件；只有 8 份已声明的文档修改了路径或历史归档提示，源码、PDF 和外部仓库未改写。
- 原文献 manifest 的 66 个已下载 PDF 均重新通过 SHA-256 校验；原先未下载的第 56 项保持缺失状态，没有以其他文件冒充。
- 本地投稿包清单的 49 项文件均存在；协议中的两处文献目录路径均改为项目内位置，没有改实验参数、划分或分数。
- 已检查项目 README、投稿入口、文献、研究索引、档案导航与父目录导航的本地链接。修正后未发现失效目标；这些检查基于本机完整工作区，不能据此保证被 Git 忽略的资产也存在于外部克隆。
- 迁入的 `literature/`、`archive/` 中没有指向父目录的链接。

搬迁逐文件收据：[BEFORE.json](../../.local/backups/organization_20261014/BEFORE.json)、[AFTER.json](../../.local/backups/organization_20261014/AFTER.json)。位置映射见[项目结构](PROJECT_STRUCTURE.md)。原始 manifest 中的旧路径作为历史溯源保留。

## 运行检查

统一使用已有 `D:\anaconda\envs\multi_agent_graph\python.exe`，Python 3.11.15；没有安装依赖、新建环境或调用 API。

| 检查 | 实际结果 |
| --- | --- |
| 旧平台 `unittest discover -s tests` | 36 通过 |
| 当前项目的打包、独立 MAS、同信息对照、两层输入/传播/语义及 MAST 相关测试 | 36 通过，1 失败 |
| 三个评测命令的 `--help` | `independent`、`binding`、`costs` 均能打开 |
| 主模型 wheel 的隔离安装与静态资源测试 | 包含在上述通过项中，可在源码目录外导入模型与静态资源 |

失败项为 `test_default_pytest_collects_both_suites_without_writing_legacy_report`。它收集完整测试集时遇到统一环境缺少 `agents` 模块，以下四个测试文件不能导入：

- `test_contract_forecast_judge_scoring.py`
- `test_harnessaudit_node_executor.py`
- `test_paired_harnessaudit_collection.py`
- `test_submission_harnessaudit_audit.py`

收集阶段识别到 2,628 项测试及上述 4 个导入错误，不代表 2,628 项测试已执行。当前 `.[test]` 没有声明该可选采集依赖，需按相关采集环境清单单独补齐后再验证全套；本次未修改依赖或用跳过测试来宣称通过。

## 未改变的范围

没有修改模型算法、重训、重新评分、补写测试标签、触发真实 API 采集或处理另一个论文项目。没有删除冻结证据，也没有提交或推送 Git。整理前已有的代码修改保留，不把它们当成本次整理成果。

本地完整目录仍不等于完整 Git 备份：多数原始数据、结果、外部仓库、权重和环境被忽略，迁机与分享范围见[复现说明](REPRODUCIBILITY.md)。
