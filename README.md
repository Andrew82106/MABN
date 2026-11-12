# MABN

本仓库包含两个独立研究项目。MAS 论文从 `paperAlpha/` 进入；《情报杂志》论文从 `intelligence_knowledge_boundary/` 进入，不要混用二者的代码、结果和交接文档。

## 交接入口

```sh
git clone --branch codex/mas-risk-monitoring-charter https://github.com/Andrew82106/MABN.git
```

- **MAS 风险监测**：[复现方法](paperAlpha/docs/usage/REPRODUCIBILITY.md)、[开发实验结果](paperAlpha/docs/manuscript/FINAL_RESULTS.md)、[独立确认进度](paperAlpha/experiments/submission_audit/runs/final_eval/CONFIRMATION_COHORT_STATUS.md)。160 条轨迹的双人盲标尚未完成；标注包和源轨迹保留在原工作区。
- **事实风险监测（《情报杂志》）**：[项目说明](intelligence_knowledge_boundary/README.md)、[同域实验记录](intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/STATUS.md)。正式 Word、可编辑 PPT、历史稿仅在本机，需由原持有人另行交接。

这是研究材料的公开交接分支。第三方文献保留来源索引和笔记；数据、完整运行记录、外部依赖和权重按各项目清单恢复，凭据自行配置。下文的目录描述包括本机资产，不代表全部内容都在 Git 中。

## Structure

| 路径 | 用途与处理建议 |
| --- | --- |
| `paperAlpha/` | MAS 论文完整本地工作区：当前方案、代码、数据、结果、文献和旧平台归档。主要工作在这里进行。 |
| `intelligence_knowledge_boundary/` | 另一篇《情报杂志》论文，有用，独立保留。 |
| `.pytest_cache/` | 测试缓存，可重建，不属于论文资产。 |
| `.git/`、`.gitignore`、`.gitattributes` | 版本历史、忽略规则和二进制/归档规则；保留。 |

原 `shared_lab/` 已迁入 `paperAlpha/archive/shared_lab/`；原 MAS 文献和个人论文已迁入 `paperAlpha/docs/references/`；早期设计与任务书位于 `paperAlpha/archive/project_docs/`。没有删除实验记录。

原 `doc/` 已清空并移除：共享指南和环境说明保留在 `paperAlpha/archive/project_docs/`，两份稿件归入《情报杂志》项目的 `reports/history/doc_manuscripts_20261014/`，参考文献实体仍在其 `reports/reference_papers/`。稿件未覆盖项目内的正式论文，旧导航和说明原件可恢复。详见[文档归位记录](intelligence_knowledge_boundary/reports/migration/DOC_CLEANUP_20261014.md)。

根目录的空 `outputs/`、旧 `prelab/` 与 `research/` 入口、重复 `PAPER_HANDOFF.md` 已移除。原 `tmp/` 已按归属收进两篇论文：MAS 阅读/下载缓存位于 `paperAlpha/.local/cache/literature_20261014/`，另一篇的缓存、环境和旧审计位于 `intelligence_knowledge_boundary/artifacts/legacy_tmp/`。

旧实验仍引用根目录绝对路径时，使用《情报杂志》项目的 `scripts/run_experiment.ps1`；它仅在运行期间建立兼容链接，正常退出或异常返回后移除。完整交接说明仍保留在该项目内。详见[根目录整理记录](intelligence_knowledge_boundary/reports/migration/ROOT_CLEANUP_20261014.md)。

## Quick Start

MAS 从 [paperAlpha/README.md](paperAlpha/README.md) 开始，目录用途和搬迁记录见 [项目结构说明](paperAlpha/docs/usage/PROJECT_STRUCTURE.md)。历史公共说明见 [工作指南归档](paperAlpha/archive/project_docs/README.md)；当前运行方式分别以两个项目的 README 为准。

The current paper direction and its non-negotiable scope are fixed in the
[paperAlpha project charter](paperAlpha/README.md). New agents and contributors
must read that charter before interpreting legacy experiments or adding work.

## Paper handoff

The current factual-hallucination monitoring experiment, its fixed evaluation
protocol, baseline table, limitations, and suggested paper framing are in
[the unified project](intelligence_knowledge_boundary/README.md) and
[the collaborator handoff](intelligence_knowledge_boundary/reports/paper/PAPER_HANDOFF.md).
Large model weights, virtual environments, raw/generated arrays, and local
logs are intentionally excluded; download locations and hashes are recorded in
the experiment manifests.
