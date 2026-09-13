# R18：固定规则选出的三条整答漏报

从602条训练集折外回答中，mean_lr漏报51条，其中28条被全部8种方法漏报。按风险窗口数降序、item_id字典序取前3条，未使用补选。它们属于原R16训练集；ID里的test是公开来源分区，不是本轮测试划分。

**共同发现：24个“案例×方法”的最高4词元窗口均碰到冻结风险范围，但都未达到各自整答阈值。整答漏报不等于完全找不到风险位置。** 局部窗口阈值与整答阈值分别校准；分数也不是已校准的真实幻觉概率。

## 1. r16_ragognize_test_0137__partial__1

问题：On what date were the returning contestants for MasterChef Australia series 17 announced?

原回答：1. The returning contestants for MasterChef Australia series 17 were announced on September 16, 2024.

资料：目标资料只说第17季节目于16 September 2024公布，并列出评委；另篇资料介绍纪录片上映。当前资料没有回归选手名单的公布日期。

问题所在：回答把“赛季公布日”直接当成“回归选手名单公布日”。日期本身来自资料，错误在日期与事件的对应关系。

冻结风险片段：`September 16, 2024`。

8种方法的最高4词元窗口都在年份内部，且都超过各自局部窗口阈值；它们有定位信号。整答阈值更高，所以整答层面8种方法全部漏报。

零起算折号 3；风险窗口 11 个。以下TP只表示窗口碰到风险，不能当作完整语义定位。

| 方法 | 峰值 | 整答阈值 | 局部窗口TP |
|---|---:|---:|---:|
| mean_lr | 0.804292 | 0.975469 | 1/11 |
| slots_lr | 0.842338 | 0.992523 | 5/11 |
| shuffled_slots_lr | 0.897558 | 0.991031 | 2/11 |
| mean_mlp | 0.854514 | 0.909676 | 5/11 |
| mean_surface | 0.833224 | 0.975844 | 2/11 |
| mean_alignment | 0.795156 | 0.976241 | 1/11 |
| mean_both | 0.826911 | 0.975976 | 2/11 |
| mean_hidden32 | 0.807721 | 0.971117 | 1/11 |

标注说明：当前资料不支持这项日期关系，原风险判定有清楚依据。问题已固定所问事件，因此只圈日期值符合原标注规则；并非认定该日期字符脱离关系后也是虚假事实。

## 2. r16_ragognize_test_0835__complete__1

问题：How were The Newsreader series 3 episodes released online compared with the weekly TV broadcasts?

原回答：1. All episodes of The Newsreader series 3 were made available on ABC iView on the same day as their weekly TV broadcasts.

资料：资料明确说2025年2月2日首播，此后电视每周播出；同一天全部剧集已在ABC iView上线。另一篇电影资料与发布安排无关。

问题所在：回答写成“on the same day as their weekly TV broadcasts”，按逐集发布安排理解，会把全集一次性上线改成随每周电视播出上线。

冻结风险片段：`on the same day as their weekly TV broadcasts`。

8种方法的最高4词元窗口均碰到原风险跨度；mean_lr、slots_lr、mean_surface有局部报警，其余5种未达到局部阈值；8种均未达到整答阈值。

零起算折号 1；风险窗口 10 个。以下TP只表示窗口碰到风险，不能当作完整语义定位。

| 方法 | 峰值 | 整答阈值 | 局部窗口TP |
|---|---:|---:|---:|
| mean_lr | 0.678313 | 0.876894 | 1/10 |
| slots_lr | 0.812710 | 0.953523 | 2/10 |
| shuffled_slots_lr | 0.660327 | 0.944430 | 0/10 |
| mean_mlp | 0.299715 | 0.862304 | 0/10 |
| mean_surface | 0.666117 | 0.872336 | 2/10 |
| mean_alignment | 0.614822 | 0.896359 | 0/10 |
| mean_both | 0.596485 | 0.933372 | 0/10 |
| mean_hidden32 | 0.676773 | 0.919851 | 0/10 |

标注说明：冻结标签经第三方裁决采用“对应各集每周播出日”的读法。句子也可能被读者善意补全为“周播开始的同一天”，因此有时间范围/指代歧义；这不是三个案例中最无争议的硬错误，报告保留原标签，不重新改标。

## 3. r16_ragognize_train_0388__partial__1

问题：What power-cut schedule did the Ceylon Electricity Board announce for February 10 and 11, 2025?

原回答：The Ceylon Electricity Board did not announce any power-cut schedule for February 10 and 11, 2025.

资料：目标资料只提供2月9日全国停电及Panadura变电站的事故陈述；另一篇资料介绍荷兰火灾。没有2月10日和11日的限电安排，也没有电力局“未公布安排”的证据。

问题所在：回答不是说“资料未给出安排”，而是断言电力局“did not announce any power-cut schedule”，把当前资料缺失变成现实中的未公布事实。

冻结风险片段：`did not announce any power-cut schedule`。

8种方法最高窗口仍在“power-cut schedule”一带，位于风险跨度；但都低于局部和整答阈值，连局部窗口也没有报警。最高窗口并未覆盖否定词本身，重叠风险跨度不等于读懂了否定语义。

零起算折号 1；风险窗口 10 个。以下TP只表示窗口碰到风险，不能当作完整语义定位。

| 方法 | 峰值 | 整答阈值 | 局部窗口TP |
|---|---:|---:|---:|
| mean_lr | 0.506656 | 0.876894 | 0/10 |
| slots_lr | 0.631219 | 0.953523 | 0/10 |
| shuffled_slots_lr | 0.507652 | 0.944430 | 0/10 |
| mean_mlp | 0.566200 | 0.862304 | 0/10 |
| mean_surface | 0.494381 | 0.872336 | 0/10 |
| mean_alignment | 0.522910 | 0.896359 | 0/10 |
| mean_both | 0.496734 | 0.933372 | 0/10 |
| mean_hidden32 | 0.560838 | 0.919851 | 0/10 |

标注说明：原标签曾经裁决，关键是现实否定和正常拒答的区别。这里没有“in the supplied sources”等限定，按字面是无依据否定，保留风险标签有依据；若对所有此类句子都作资料限定的宽松读法，任务定义会改变，不能据分数临时改变。

这三条是刻意挑选的共同失败，且优先风险跨度较长者，不能代表总体。它们显示关系绑定、时间范围和无依据否定的困难；其中有相对定位信号，也有整答阈值未触发，不能据此断言模型完全没有学到信息或某条训练数据造成了失败。

具体最高窗口原文、字符坐标、原资料、局部阈值、TP/FP/FN及全部来源哈希见 [CASE_REVIEW18.json](CASE_REVIEW18.json)。原标签、预测、模型和阈值均未修改。
