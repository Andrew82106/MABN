# 《情报杂志》项目工作区

本目录是“面向开源情报问答的大语言模型事实风险监测与定位研究”的统一工作区。论文、代码、数据、模型、实验结果与参考资料均已归入本目录。

公开 Git 分支只交接研究材料。下列 Word、PPT 和历史稿链接仅在原本机工作区可用；研究结果从[同域实验状态](src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/STATUS.md)进入。原始数据、环境和权重按实验清单另行准备。

## 常用入口

- [正式论文](reports/paper/面向开源情报问答的大语言模型事实风险监测与定位研究（正式论文）.docx)
- [图1可编辑PPT](reports/paper/图1_FRML总体框架（可编辑）.pptx)
- [论文写作交接提示](reports/paper/PAPER_WRITING_PROMPT_CN.md)
- [写作风格要求](reports/paper/WRITING_STYLE_CN.md)
- [实验汇总与交接说明](reports/paper/PAPER_HANDOFF.md)
- [当前主实验代码](src/prelab/benchmark_ragtruth_qa/src/)
- [当前实验状态](src/prelab/benchmark_ragtruth_qa/CURRENT_STATUS.md)
- [正式基线结果](src/prelab/benchmark_ragtruth_qa/FORMAL_BASELINE_RESULTS.md)
- [参考论文](reports/reference_papers/README.md)
- [从旧 doc 收入的稿件](reports/history/doc_manuscripts_20261014/README.md)：单独归档，不替代上面的正式论文。

## 目录分工

| 目录 | 内容 |
|---|---|
| `src` | 实验代码。现有各轮实验保留在 `src/prelab`，主实验为 `benchmark_ragtruth_qa` |
| `scripts` | 运行入口与目录兼容工具 |
| `configs` | 实验配置及目录映射 |
| `data` | 训练数据、评测数据及特征，主实验数据位于 `data/benchmark_ragtruth_qa` |
| `artifacts` | 模型、Python 环境、外部依赖、日志及本地迁移记录 |
| `reports` | 论文、实验结果、文献、调研与历史方案 |

论文集中放在 `reports/paper`。当前主实验结果入口为 [reports/experiments/benchmark_ragtruth_qa](reports/experiments/benchmark_ragtruth_qa/)。早期方案保存在 [reports/history/initial_proposal](reports/history/initial_proposal/)。

部分历史结果目录内含依赖原目录位置的审计脚本，因此保留代码与结果的内部布局，通过项目内目录链接接入分类；同一份文件只有一份实体。代码实体在 `src/prelab/`，补充报告在 `reports/research/`，根目录旧入口已取消。

原根目录 `tmp/` 中属于本项目的缓存、专用环境、旧审计和迁移记录已归入 `artifacts/legacy_tmp/`。根目录重复的交接入口已移除，正式交接说明继续从上面的“实验汇总与交接说明”进入。

## 运行

在本目录执行主实验脚本，例如查看可用参数。

```powershell
.\scripts\run_experiment.ps1 benchmark_ragtruth_qa/src/run_development.py --help
```

已有 Python 环境位于 `artifacts/environments/prelab`，模型位于 `artifacts/models/initial`。新检出仓库先运行 `scripts/restore_compatibility_links.ps1` 恢复目录链接；模型、数据和环境按各实验清单另行准备。

旧实验清单中保存了根目录 `prelab/` 等绝对路径，原始清单不改写。统一运行入口会临时建立 `prelab/`、`research/`、`tmp/` 链接，等待脚本结束后移除；同一仓库的此类运行不并发启动。它不会覆盖已有的普通目录，也不会删除实际代码和数据。强行终止 PowerShell 进程可能遗留临时链接，需先确认没有运行中的实验再清理。

旧命令请通过上述入口运行，不再直接使用根目录的 `prelab/.venv/...`。正常目录恢复脚本只恢复长期保留的链接，不重新创建已取消的根目录入口。

迁移清单和检查结果见 [reports/migration/README.md](reports/migration/README.md)。

本次根目录整理及临时文件归属见[整理记录](reports/migration/ROOT_CLEANUP_20261014.md)。

根目录 `doc/` 已撤销，两份稿件及参考文献均在本项目内。共享指南的历史快照归入 MAS 项目，本项目运行方式以上文为准。后续文档归位与原件校验见[文档整理记录](reports/migration/DOC_CLEANUP_20261014.md)。
