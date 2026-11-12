# 根目录整理记录

2026-10-14，按用户要求清理根目录入口并把临时材料分归两篇论文。正式论文、代码、数据、模型及冻结实验清单保持原内容。

## 调整后的入口

| 原位置 | 处理 |
| --- | --- |
| `outputs/` | 确认只有空目录后删除 |
| `prelab/` | 移除根目录链接，代码实体仍在本项目 `src/prelab/` |
| `research/` | 移除两个旧链接及空父目录，报告仍在本项目 `reports/research/` |
| 根目录 `PAPER_HANDOFF.md` | 移除重复入口，完整交接说明保留在 [reports/paper/PAPER_HANDOFF.md](../paper/PAPER_HANDOFF.md) |
| `tmp/` | 全部分归以下两处，再删除空父目录 |

## 原 tmp 材料的归属

- MAS 的 `literature_archive/`、`pdfs/bayestrace_read_…/`、`pdfs/literature_explanation/` → `paperAlpha/archive/local_scratch_20261014/`，保留原相对层次。
- 事实风险论文的其余材料 → 本项目 `artifacts/legacy_tmp/`，保留原相对层次：RAGognizer 专用环境、U-head 作者仓库、Llama tokenizer 审计、原子证据路由器旧快照、论文阅读缓存、样本预览及早期迁移记录。

共迁移 7,458 个文件，前后 SHA-256 全部一致，没有丢弃临时材料。清单与旧入口/配置备份保存在本地 `artifacts/migration/root_cleanup_20261014/`，不随 Git 发布。

## 旧实验的路径兼容

部分冻结清单保存了根目录 `prelab/…` 的绝对路径，不能直接改写它们而沿用原 hash。兼容处理集中在本项目 `scripts/with_legacy_paths.ps1`，由 `scripts/run_experiment.ps1` 调用：

1. 运行时临时建立 `prelab/`、`research/`、`tmp/` 三个链接，全部指向本项目实体目录。
2. 等待实验脚本退出，无论正常结束还是抛出异常，移除本次创建的链接，不递归删除目标内容。
3. 同仓库的此类运行使用互斥锁，不能并发启动；若路径已经存在且不是预期链接，拒绝覆盖。不是本次创建的链接不擅自删除。

默认 `restore_compatibility_links.ps1` 只恢复长期保留的内部链接，不重建上述常驻根目录链接。参考论文旧入口在后续 [doc 清理](DOC_CLEANUP_20261014.md)中也已取消。强行杀死 PowerShell 进程可能绕过退出清理，遗留链接需确认无运行任务后再删除。

在本项目目录运行：

```powershell
.\scripts\run_experiment.ps1 benchmark_ragtruth_qa/src/run_development.py --help
.\scripts\test_legacy_paths.ps1 -CheckActualManifest
```

没有重跑论文训练或评分，也没有修改数据标签、模型参数、真实 API 调用配置。本次没有提交或推送 Git。

## 检查结果

- 7,458 个迁移文件哈希一致；仅移除空目录、已验证的目录链接和重复交接入口，实际目标内容保留。
- 4 项隔离检查通过：正常退出清理、异常退出清理、不删除借用的链接、不覆盖冲突目录。
- 真实旧清单的 7 条绝对源路径在兼容入口内可解析，原清单字节未修改；退出后根目录无临时链接。
- 本阶段 82 个长期目录链接检查通过；后续 doc 清理取消其中一个旧参考论文入口，现为 81 个。默认恢复配置不再要求根目录 `prelab/` 或 `research/` 常驻。
- 主实验启动器 `--help` 返回 0；无效参数检查保持原退出码 2；两种情况都清理了临时链接。
- 搬迁后的 RAGognizer 专用 Python 可从新位置启动，成功导入原有 torch 2.5.1+cu121 和 peft 0.18.1；未重新安装环境。
