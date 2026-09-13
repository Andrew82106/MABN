# Weakly Supervised Factual Error Monitoring

当前方向为“搜索资料不完整时，模型自行回答的事实项风险监测”。第六轮已完成30题、60份Qwen自生成回答及全部基线比较，使用正常提问，不提示“资料不足可以不答”。6题留出检查中探针事实项F1为0.762，简单预测熵为0.750，尚未证明探针有稳定优势。见[第六轮报告](round6_evidence_grounding/results/REPORT.md)、[全部资料与回答](round6_evidence_grounding/results/examples.html)和[运行前详细方案](round6_evidence_grounding/PLAN.md)。下文保留前五轮历史任务与结果，不能将旧分数与第六轮混用。

《情报杂志》方向的独立预实验：只用整段“有错误/无错误”标签训练内部状态探针，再独立检验能否定位错误片段。旧的新旧事件知识边界方案已被用户替换。

已完成：数据、模型、探针训练、独立测试及补充自然生成。**当前结果未证明可靠定位**，详见 [结果报告](results/REPORT.md) 和 [全部测试热图](results/token_risk_report.html)。

第二轮扩大实验已完成：7B、五组主实验和 110 条独立来源新闻复核，见 [第二轮报告](round2/results/REPORT.md) 和 [基线实现审计](round2/BASELINES.md)。位置监督改善排序，但低误报自动报警仍未解决。下文描述第一轮，不与第二轮结果混合。

第三轮结构调参与补充信息实验已完成，见[第三轮报告](round3/results/REPORT.md)：复杂网络和当前长解释融合未形成稳定优势；三选一短核查有回答级改善迹象，逐词定位和低误报检出仍待突破。新测试为80条不同来源的Llama摘要，由同一本地Qwen重放和核查。

第四轮扩展核查范围、事实点核查与F1优化已完成，见[第四轮结果](round4/README.md)：候选方案在第一批100条新来源达到回答级F1 0.703/0.709，但冻结后在另50条新来源复核降至0.588/0.562，尚未稳定达到0.7；逐词定位也未突破。两批结果和失败的预选主方法完整保留。

第五轮[人工选点诊断](round5_oracle/README.md)已完成：在相同50条回答和固定模型、阈值、核查次数下，人工指定错误句子使回答级F1从0.588升至0.737，检出10/20→14/20，误报不变；进一步圈准错误片段未新增检出。这是借助人工标注的诊断，支持继续检查自动选点瓶颈，不代表自动检测或逐token定位已达标。

## 数据与实验

- RAGTruth（ACL 2024）英文新闻摘要，固定检索材料。限定同一原始生成模型 Mistral-7B-Instruct，减少生成器文风与标签的混淆。
- 仅保留人工标注的 Evident Conflict，或完全没有幻觉标注的回答；排除混合类型、隐含正确、空值导致的问题、不合格回答和超长上下文。
- 394 条：训练 240（100 错/140 正常）、验证 65（25 错/40 正常）、测试 89（25 错/64 正常）。保留官方测试集，训练来源另划验证；按来源文本哈希隔离，原文不截断。
- 冻结 Qwen2.5-0.5B-Instruct，提取第 8、16、24 层。训练集拟合探针，验证集整段 AUROC 选择层和训练步；片段标签放在独立目录，训练代码不读取。
- 补充 16 个留出新闻来源的小模型自然生成样本；证据审阅由执行助手完成，单独标明，不冒称独立双人标注。

## 对比

1. 最后 token / 平均状态线性探针：整段基线，直接局部应用仅作为朴素对照。
2. token MLP + 平均、最大值、最高约 10% 均值三种汇总，只用整段二分类损失。
3. HaMI 官方 **ori 原始表征版本**：原网络、原 MIL 损失和 top-k 汇总。小模型、小数据、层筛选和训练预算为本地适配；未复现语义一致性增强，不声称复现论文完整实验或原数值。
4. token 惊讶度、文本 TF-IDF、回答长度、打乱标签对照。

四种 MLP 各跑 3 个随机种子。比较整段 AUROC、F1，以及错误回答内部的 token AUROC、AP、top-10% 精确率和召回率。分开报告整段分类与局部定位；风险分数不当作校准概率。

## 解释边界

- 主实验是 Qwen **重放已有 Mistral 回答**的白盒表征实验，不能直接代表 Qwen 自己生成时的检出率；自然生成补充实验另报。
- Evident Conflict 是相对证据的冲突标签，未独立核验所有现实事实；“材料未提到”不等于事实错误。
- token t 的特征来自读取该 token 后的状态，可用于生成后监测，不能解释为生成前预警；分数可能反映前文已出现错误。
- 固定检索材料模拟联网后的阅读摘要阶段；没有运行实时搜索代理。
- 完整输出 top-10% 排名是离线定位指标，不等于在线报警策略。逐位置探针使用因果状态。
- 本预实验不支持跨语言、跨模型或开源情报领域泛化，也不预设正结果或创新性。

## 环境与运行

用户已明确授权创建环境。解释器为 `D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe`，Python 3.11.15。

通过 `D:\anaconda\envs\CA\python.exe -m venv --system-site-packages prelab/.venv` 创建，继承 PyTorch 2.5.1+cu121；其余依赖见 [requirements.txt](requirements.txt)，实际版本见 [environment_freeze.txt](results/environment_freeze.txt)。新机器可创建独立 Python 3.11 环境，从官方 cu121 索引安装 torch，再安装其余依赖。

仓库根目录执行：

```powershell
$prelabPython = 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe'
& $prelabPython prelab/src/download_data.py
& $prelabPython prelab/src/download_model.py
& $prelabPython prelab/src/prepare_data.py
& $prelabPython prelab/src/extract_features.py
& $prelabPython prelab/src/fit_probes.py
& $prelabPython prelab/src/evaluate.py
& $prelabPython prelab/src/audit.py
& $prelabPython prelab/src/generate_own.py
& $prelabPython prelab/src/evaluate_own.py
& $prelabPython prelab/src/build_report.py
& $prelabPython prelab/src/verify_delivery.py
```

运行前将官方 HaMI 仓库检出到 `references/HaMI`，版本见下。数据下载脚本校验固定哈希；模型支持镜像续传并校验官方 LFS SHA-256。训练和特征缓存可续跑；更换配置须使用新目录，不能混用缓存。自然生成复用本次留存的审阅文件；若更改样本或解码设置，需要重新独立标注。

## 文件

| 路径 | 内容 |
|---|---|
| `configs/experiment.json` | 预先固定的对比、层、种子和指标 |
| `data/processed/` | 只有整段标签的固定划分 |
| `data/annotations/` | 与拟合隔离的片段标注 |
| `data/features/` | 内部状态、token 索引、字符偏移和惊讶度 |
| `data/own/` | 自然生成、单独标注和特征 |
| `models/` | 模型缓存，不进入 Git |
| `results/` | 指标、预测、探针权重、可视报告和审计 |
| `logs/` | 执行日志 |
| `references/` | 直接参考的论文和上游源码 |

## 来源

- [HaMI，NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/b7c43d4a79dede363a2d061c6158e5a5-Abstract-Conference.html)：[本地论文](references/HaMI_NeurIPS2025.pdf)，[官方代码](https://github.com/mala-lab/HaMI)，commit `3d277605534999d2756a6eeef759d0a91199ad58`。
- [RAGTruth，ACL 2024](https://aclanthology.org/2024.acl-long.585/)：[本地论文](references/RAGTruth_ACL2024.pdf)，[官方数据](https://github.com/ParticleMedia/RAGTruth)。哈希在 `results/dataset_manifest.json`，数据许可随原始下载保留。
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，revision `7ae557604adf67be50417f59c2c2f167def9a775`，模型许可随权重保留。
