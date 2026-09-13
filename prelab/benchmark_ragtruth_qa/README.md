# RAGTruth QA：原 Llama-2-7B-chat 回答的白盒重建基准

**当前训练、基线比较与数据扩充状态见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。** 下文保留最初数据准备时的说明；其中“尚未加载GPU/训练”等历史状态已由当前进展取代。原QA测试仍封存。

本目录准备独立的人工 span 问答基准，不替代原 R16 控制场景，也不占用 Round21。使用官方发布的 `llama-2-7b-chat` 原回答，文字与人工字符标注保持不变。目标重放是**固定模型、模板与 NF4 精度的教师强制重建**，不是原始生成 token 轨迹的精确复现。目前只做数据与模型准备，未加载 GPU 或提取特征。

## 已准备的数据

| 分区 | 来源数（质量筛选前） | `quality=good` 原回答 | 可用回答对应组 |
|---|---:|---:|---:|
| fit | 646 | 634 | 615 |
| calibration | 162 | 159 | 154 |
| 密封官方 test | 150 | 未检查 | 147 个来源关联组 |
| 与 test 关联的官方 train | 31 | 未检查 | 归入上述密封组，暂不使用 |

同一个 source 的六模型回答始终同组同区；本基准当前只导出 Llama-2-7B-chat 的一个原回答。Source 数不能乘六后称独立问题数。完整来源索引共 989 source、928 个当前关联组。`withheld_train_linked_to_test` 的 8 组也在 test 的 147 组中，两个组数不能相加。

质量口径在导出前固定：仅官方 `quality == good` 进入主数据。15 个 `truncated` 开发回答保留于独立审计文件，不是根据错误数量或检测分数删除。其余全部四类人工 span，以及 `implicit_true`、`due_to_null`、`meta`、原始 `start/end/text` 全保留。808 个可打开的开发回答共有 886 个 span；原始字符区间与 `text` 全部吻合。这里包含15个截断审计回答，不能当作793个主数据的 span 数。

793个主回答共有846个span：fit中328个回答有span、306个无span；calibration中100个有span、59个无span。回答中位长度约1,000字符，明显长于旧控制集单句输出。详见 [开发集统计](data/development_profile.json)；这些统计不用于反向改分区。

- [fit.jsonl](data/fit.jsonl)、[calibration.jsonl](data/calibration.jsonl)：原问题、3段检索资料字符串、发布 prompt、原回答和全部原标注。
- [source_index.jsonl](data/source_index.jsonl)：所有来源的组/分区及模型回答ID；[sealed_source_index.jsonl](data/sealed_source_index.jsonl) 只含身份和哈希，不含密封回答或标签。
- [quality_excluded_development.jsonl](data/quality_excluded_development.jsonl)：保留15个官方截断开发回答及原标注，主数据不使用。
- [development_manifest.json](data/development_manifest.json)：实际计数、质量规则、来源与产物哈希。

尚未根据原回答自动猜测“拒答/正常断言”子类，也未固定本项目 token/window 评分。后续在开发资料上预先确定如何处理完整拒答、四类 span、`implicit_true` 和 `due_to_null`；官方测试的同一口径必须在读取其标签之前锁定。把截断回答排除与把疑难/无依据回答排除是不同操作。

## 来源隔离做到了什么

审计覆盖734个旧新闻来源和4,024份旧文档/输入行，包含早期Trivia/SQuAD与R6/R7/R9/R10/R16实际资料。QA对QA查到20条共享整段或连续20词的边；QA对旧资料未发现这一标准的原文重复。来源缺原页面URL，因此不能仅靠URL追踪同篇文章。

另对140个候选问句对实读，78条按相同/疑似相连的平台、机构、窄主体或食谱族保守合组，62条只有通用操作词或不同对象而不由该边合组。比如同一城的天气近重复、同一账号/平台的操作题需要注意；“都问怎样做饭”不足以把所有菜谱当同一事件。原段复用即使只是天气模板也暂作保守合组。所有触及官方 test 的关联组保持密封，31个训练来源留存索引但不导出回答，不移动到拟合集。

这次审计**不是穷尽所有实体别名/事件的认证**。865个低门槛词面候选对中只有上述140个做了逐对复核，剩余单一通用词候选保留待审。因此可称“来源及已识别关联组隔离”，不能称“严格新主体、新事件测试”。MS MARCO 问答还包含大量烹饪、操作、健康解释题，也不是近时新闻情报任务。若论文要主张新事件泛化，仍需另建R20方案里的专用密封事件测试。

审核记录：[材料重复](data/material_overlap_audit.json)、[问句关联复核](data/semantic_group_review.json)、[审计概况](data/source_audit_summary.json)。分区只按来源关联与固定种子 `20260911` 构成，不按标签平衡。R20最初的671/168分区是精确整文检查后的提案，已被这里的细段与关联组划分替代。

## 模型与重放

使用原 `llama-2-7b-chat` 回答，可选公开 HF 转换镜像 `NousResearch/Llama-2-7b-chat-hf`，锁 revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee`。两份 safetensors 的公开LFS哈希与Meta官方固定快照一致，总13,476,872,576 bytes；实际下载后的逐文件哈希由下载 manifest 核对。Meta原入口需要许可，未修改账户权限；公开镜像仍受Llama2许可约束。

权重字节相同不代表整个分词环境相同。镜像 tokenizer 配置与Meta不同，且没有默认 chat template；`tokenizer.model` 相同仍不足以忽略配置。必须显式处理 README 的 `<s>[INST] {prompt} [/INST]` 外层、BOS重复、prompt/response边界、pad/attention mask和字符偏移。NF4及内核差异也需要披露。原始 prompt/response token IDs 没有发布，所以不能声称恢复了历史原始轨迹。

详见 [checkpoint、许可与质量依据](CHECKPOINT_AND_QUALITY.md) 和 [下载状态](model_download_manifest.json)。没有重新生成或改写原回答。人工 span 是对原文本的标签，只有保持文本和坐标不变才能沿用。

后续：根代理复核分组和数据口径后锁定重放配置；只对 fit/calibration 提取并调参。官方150题保持封存，禁止用其具体答案、风险标注或检测分数指导选型。
