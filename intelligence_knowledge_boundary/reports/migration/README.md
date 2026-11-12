# 工作区整合记录

2026-09-27，按用户要求将《情报杂志》项目统一迁入 `intelligence_knowledge_boundary`，保留原有六类顶层目录。

**2026-10-14 后续整理：**根目录旧入口已取消，原 `tmp/` 已分归两篇论文，见[根目录整理记录](ROOT_CLEANUP_20261014.md)。下文 85 个链接等数字是 9 月迁移时的历史验收记录；当前长期链接以 `workspace_layout.json` 的 `directory_aliases` 为准。

同日又完成[旧 doc 归位](DOC_CLEANUP_20261014.md)：两份稿件单独保存，参考文献旧入口取消，根目录不再保留 `doc/`；当前长期目录链接为 81 个。

## 内容归属

| 内容 | 统一位置 |
|---|---|
| 各轮实验代码 | `src/prelab` |
| 当前主实验代码 | `src/prelab/benchmark_ragtruth_qa/src` |
| 主实验数据 | `data/benchmark_ragtruth_qa` |
| 实验配置 | `configs` |
| Python 环境 | `artifacts/environments/prelab` |
| 下载模型 | `artifacts/models/initial` |
| 主实验结果入口 | `reports/experiments/benchmark_ragtruth_qa` |
| 论文与写作交接 | `reports/paper` |
| 参考论文 | `reports/reference_papers` |
| 补充调研 | `reports/research` |
| 最初项目框架 | `reports/history/initial_proposal` |

含位置相关审计脚本的数据或结果目录保留原内部布局，通过目录链接接入分类入口。9 月迁移时曾保留旧仓库根目录 `prelab`、旧参考论文路径和两处旧调研路径作为兼容链接；10 月根目录清理后，前者及旧调研入口改为运行时临时兼容，旧参考论文入口直接取消。全部实体内容仍位于统一工作区内。

## 迁移检查

- 迁移前后核对 152,978 个原有文件，大小及修改时间一致。
- 对 224 个关键文件核验 SHA-256，包括全部主实验源文件、论文初稿、主数据文件与主要结果表，全部一致。
- 85 个目录兼容链接的目标全部通过检查。
- 迁移后的 Python 环境成功加载 PyTorch、scikit-learn 和 Transformers。
- 新运行入口通过 `--help` 与已有 `run_development.py self-test` 检查。
- 特征提取模块的 Llama/Qwen CPU 自检通过，40 处导航链接均有效。
- 2,009 个原跟踪项已在 Git 暂存区迁移到新路径，未提交或推送；其他项目的工作区改动保持原状。

完整性核验后，导航文档和论文交接文档更新了路径及写作风格入口，原稿备份位于 `artifacts/migration/document_originals`。实验方法、模型参数、数据标签和成绩未作修改。

目录映射见 [workspace_layout.json](../../configs/workspace_layout.json)，逐项校验见 [file_integrity.json](file_integrity.json)。本地完整清单和移动日志位于 `artifacts/migration`。

新检出仓库可运行 `scripts/restore_compatibility_links.ps1` 重建链接。Git 保留代码、协议、文献和报告，模型、环境、原始数据及大型运行产物继续保存在本地。
