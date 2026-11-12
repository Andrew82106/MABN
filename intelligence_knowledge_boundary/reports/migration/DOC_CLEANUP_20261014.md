# 旧 doc 文档归位

2026-10-14 按用户要求取消仓库根目录 `doc/`，材料按两篇论文归属收回，没有覆盖当前正式稿。

## 文件归属

| 原内容 | 当前位置 |
| --- | --- |
| 《面向开源情报问答的大语言模型事实风险监测与定位研究》PDF、Word | [历史稿件目录](../history/doc_manuscripts_20261014/README.md)；保持原名，不替代 `reports/paper/` 的正式稿 |
| `Doc工作指南.md`、`Python环境说明.md` | 阅读版本沿用 `paperAlpha/archive/project_docs/`；两份逐字原件归入 `paperAlpha/results/organization_20261014/doc_cleanup/originals/` |
| `doc/README.md` | 已由两个项目及仓库 README 接替导航；原件保存在上述 `originals/` 中 |
| `doc/ref_paper/intelligence_knowledge_boundary/` | 旧入口移入整理备份，全部实体仍在 [reports/reference_papers](../reference_papers/README.md) |

工作指南与既有归档逐字一致；环境说明仅有此前已说明的跨项目导航调整，历史资源及授权记录不变。当前工作仍以各项目 README、AGENTS 和当前协议为准，不沿用历史任务的调用授权。

## 完整性与恢复

五个原文件迁移前后 SHA-256 全部一致，清单与新位置位于 `paperAlpha/results/organization_20261014/doc_cleanup/BEFORE.json` 和 `AFTER.json`。参考库除导航 README 外的 81 个文件内容一致，当前正式稿哈希不变；本次只修正文献 README 的入口链接，不改文献 PDF、下载记录或实验数据。

更新后的导航通过 57 条本地链接检查，布局恢复工具的 81 个长期目录链接全部通过检查，配置不再要求根目录 `doc/` 存在。

导航与布局配置修改前的备份位于本项目 `artifacts/migration/doc_cleanup_20261014/originals/`。原 `doc/` 空壳及参考文献快捷入口已移入同级 `legacy_entrypoints/doc/`，没有递归删除或复制链接目标。恢复旧入口可将该空壳移回根目录；文献实体无需移动。

目录恢复配置已去掉旧 `doc/` 链接，历史 `moves` 记录保留原样；正常恢复脚本不再创建根目录 `doc/`。根目录已清理，没有永久删除稿件、说明、文献或冻结实验记录。
