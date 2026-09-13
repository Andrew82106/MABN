# Round8固定7例：逐词元定位解释审阅

这7例在查看新token分数前已固定，全部保留；它们说明边界类型，不代表总体准确率。原文、gold和分数均读取冻结文件，没有重标、改阈值或重跑算法。

下面主表使用设置B：只在validation上选出的token阈值。设置A沿用Round7回答项阈值，作为阈值影响的并列说明；没有按案例择优。所有9个原生token方法均列出，三个MLP种子分别保留。整项广播留在总表中作为对照。LR指隐藏状态逻辑回归；MLP10、MLP11、MLP12分别对应种子20260910、20260911、20260912。

“命中/金标”按实际Qwen BPE词元计数；“额外”指报警落在gold片段之外，不等于断言那些功能词都已独立核验为真。相邻报警的可评词元合并展示，夹在中间的纯空格/标点按原文补回；不跨过未报警的可评词元，也不把碎片扩成完整词。表内o、63、AR等是实际分词边界，不是排版遗漏。

| 方法 | A：原回答项阈值 | B：验证集token阈值 |
| --- | ---: | ---: |
| LR | 0.620733086 | 0.99660596 |
| MLP10 | 0.74011755 | 0.98966521 |
| MLP11 | 0.377937704 | 0.991945088 |
| MLP12 | 0.435194075 | 0.975966394 |
| Lookback | 0.746901226 | 0.99585541 |
| ReDeEP | 0.48266934 | 1.04533188 |
| LUMINA | 0.00497216359 | 0.111631073 |
| NLL | 0.000514618936 | 0.00234079361 |
| Entropy | 0.00366670731 | 0.0106453057 |

阈值表仅作显示舍入，报警使用JSON中的完整精度及score ≥ threshold。ReDeEP分数不是概率，阈值可以超过1。LR、MLP、Lookback、ReDeEP使用读入当前token后的信号；LUMINA/NLL/熵按其原预测位置对齐，不作事后时移。

## 1. 缺少出生年证据：只抓到年份后半段

`hotpot_5adf2b325542993a75d2640b__partial__2`

原文：

> Christo van Rensburg was born in 1963.

Gold：`1963`。

依据：当前四段中，Christo只在1993 Open 13双打决赛记录中出现；另有Mary-Ann Eisel出生1946及其他赛事资料。没有Christo出生1963的证据。题目只问出生年，故仅圈年份。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `o`；`ensburg was born in`；`63` | 2/4 | 5 |
| MLP10 | `Christo`；`ensburg`；`in 1`；`63` | 3/4 | 4 |
| MLP11 | `63` | 2/4 | 0 |
| MLP12 | `Christo`；`1`；`63` | 3/4 | 2 |
| Lookback | `63` | 2/4 | 0 |
| ReDeEP | 无 | 0/4 | 0 |
| LUMINA | `ensburg`；`63` | 2/4 | 1 |
| NLL | `ensburg`；`63` | 2/4 | 1 |
| Entropy | `ensburg`；`63` | 2/4 | 1 |

Lookback在A下：`ensburg was`；`in`；`63`。命中2/4，额外3。

解读：Lookback在1963的四个数字上分别约为0.021027、0.031824、0.999985、0.999971，因此B阈值只圈63，命中2/4。MLP11也只圈63。这个例子有局部定位信号，但不能把“错误span被碰到”写成“完整错误年份已定位”。

## 2. 资料完整但算错差值：报警落在单位

`hotpot_5abd0cbd5542996e802b46c2__complete__3`

原文：

> Youssef Chahine was born 20 years after Gordon Douglas (director).

Gold：`20 years`。

依据：Gordon Douglas资料给1907，Youssef Chahine资料给1926。年份差是19，不是20；after方向有据，因此gold为数字与单位20 years。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `f Ch`；`ine`；`years` | 1/3 | 3 |
| MLP10 | `ine`；`years` | 1/3 | 1 |
| MLP11 | 无 | 0/3 | 0 |
| MLP12 | 无 | 0/3 | 0 |
| Lookback | `years` | 1/3 | 0 |
| ReDeEP | `ah`；`born 2`；`years` | 2/3 | 2 |
| LUMINA | `0`；`after` | 1/3 | 1 |
| NLL | `0`；`after` | 1/3 | 1 |
| Entropy | `0`；`after` | 1/3 | 1 |

Lookback在A下：`was born`；`years`；`Douglas`。命中1/3，额外3。

解读：Lookback只圈years，20的两个数字都未报警；LUMINA/NLL/熵则圈0与有据的after。各方法在同一个错误附近反应不同，严格同位置评估只承认各自真正命中的词元；本轮没有把分数前后平移去追认命中。

## 3. 同一句既有已知事实又有无据事实：命中伴随扩散

`hotpot_5ae1083e554299422ee995d7__partial__3`

原文：

> Eugene Hütz was born in 1972, which is later than the birth year of Alex Chilton (who was born in 1950).

Gold：`later than`；`1950`。

依据：Eugene Hütz资料明确给1972；Gravest Hits、Seekers and Finders和Third专辑资料没有Alex Chilton生年。保留Hütz的1972，只圈later than以及另一次无据填写的1950。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `Eugene H`；`z`；`born in`；`2, which is later than the birth year of Alex`；`ilton (who was born in`；`9`；`0` | 4/6 | 18 |
| MLP10 | `Eugene H`；`z`；`born`；`1`；`2, which is later than the birth year of Alex Chilton (who was born in 1950` | 6/6 | 19 |
| MLP11 | `Eugene H`；`z`；`1`；`2, which is later than the birth year of Alex Chilton (who was born in 1`；`50` | 5/6 | 18 |
| MLP12 | `Eugene H`；`z`；`1`；`2, which is later than the birth year of Alex Chilton (who was born in 1950` | 6/6 | 18 |
| Lookback | `in`；`2, which`；`later than`；`year of Alex Chilton (who was born in 1950` | 6/6 | 12 |
| ReDeEP | `born`；`which is`；`the`；`was`；`in`；`5` | 1/6 | 6 |
| LUMINA | `Eugene`；`üt`；`in`；`which is later`；`the birth year`；`Alex`；`ilton (who was`；`50` | 3/6 | 12 |
| NLL | `Eugene`；`was`；`in`；`which is later`；`the birth year`；`who was born in`；`50` | 3/6 | 12 |
| Entropy | `Eugene`；`was`；`in`；`which is later`；`the birth year`；`ilton (who was born in`；`50` | 3/6 | 13 |

Lookback在A下：`z was born in 1`；`72, which is later than the birth year of Alex Chilton (who was born in 1950`。命中6/6，额外20。

解读：Lookback覆盖later than和1950全部6个风险词元，同时还报了12个span外词元；其中包括有据1972的末位2，以及大量连接和主体文字。风险覆盖充分与定位精细是两回事。LR和三个MLP也有明显范围扩散，不能只展示两个gold被命中。

## 4. 属的属性嫁接到科：词面接近资料仍可有风险

`hotpot_5ae52beb5542990ba0bbb1e1__partial__1`

原文：

> Kunzea belongs to a family whose taxonomy is not settled and is complicated by the existence of hybrids.

Gold：`whose taxonomy is not settled`；`is complicated by the existence of hybrids`。

依据：Kunzea资料原句是“The taxonomy of the genus is not settled and is complicated by the existence of a number of hybrids.”。回答把genus的分类属性写成family的属性；其他三段仅提供Trollius、Typhonodorum、Hyoscyameae的信息。两个错配谓词分别圈出。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `a belongs to a family`；`taxonomy is`；`settled and is` | 4/12 | 6 |
| MLP10 | `Kun`；`a belongs to a family whose taxonomy is not settled`；`hybrids` | 6/12 | 6 |
| MLP11 | `a belongs to a family whose taxonomy is not settled`；`hybrids` | 6/12 | 5 |
| MLP12 | `Kunzea belongs to a family whose taxonomy is not settled`；`hybrids` | 6/12 | 7 |
| Lookback | `zea`；`to` | 0/12 | 3 |
| ReDeEP | `a`；`to`；`whose` | 1/12 | 2 |
| LUMINA | `Kun`；`a family`；`taxonomy`；`not`；`and`；`hybrids` | 3/12 | 4 |
| NLL | `Kun`；`a family whose taxonomy`；`not settled and is complicated`；`the`；`hybrids` | 8/12 | 4 |
| Entropy | `Kun`；`a family whose taxonomy`；`not settled and is complicated`；`the`；`hybrids` | 8/12 | 4 |

Lookback在A下：`Kunzea belongs to`；`family whose`；`not`；`the`；`hybrids`。命中4/12，额外6。

解读：Lookback在B阈值只圈zea和to，两个gold共12个词元全漏；A阈值能碰到whose、not、the、hybrids等4个gold词元，但仍未完整圈出错误关系。回答大量复用了资料中的谓词，却改变了其所属层级。这支持一种案例层面的推测：词面复用可能伴随较弱的局部风险信号，但这里没有做注意力干预，不能据此证明注意力造成了漏检。

## 5. 0188：LG合作对象被填成Manus

`ragognize_test_0188__partial__1`

原文：

> LG AI Research collaborated with Manus, an AI agent that functions as a multi-agent system powered by several distinct models, though specific language models used are not specified.

Gold：`Manus`。

依据：问题是“Can you tell me who LG AI Research collaborated with?”。当前只给Manus (AI agent)和Draft:Linagora资料：前者确实介绍Manus的multi-agent system、several distinct models及模型细节未披露；两段都没有LG与Manus合作的证据。按冻结规则，问题已经指定合作关系，只圈填错的对象Manus；后面关于Manus的说明有当前依据。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `LG AI Research collaborated`；`powered by several`；`models, though specific`；`models used are not specified` | 0/1 | 15 |
| MLP10 | `LG AI Research collaborated with`；`powered by several distinct models, though`；`language models used are not specified` | 0/1 | 17 |
| MLP11 | `LG`；`Research collaborated with`；`powered by several distinct models, though`；`models used are not specified` | 0/1 | 15 |
| MLP12 | `LG AI Research collaborated with`；`powered by several distinct models, though`；`language models used are not specified` | 0/1 | 17 |
| Lookback | `LG AI Research collaborated with`；`are` | 0/1 | 6 |
| ReDeEP | 无 | 0/1 | 0 |
| LUMINA | `LG`；`collaborated`；`Manus`；`agent that functions`；`powered`；`distinct`；`though specific language`；`are not specified` | 1/1 | 13 |
| NLL | `LG`；`collaborated`；`Manus, an`；`agent that functions`；`multi`；`powered`；`several distinct`；`though specific language`；`used are not specified` | 1/1 | 17 |
| Entropy | `LG`；`collaborated`；`Manus, an`；`agent that functions`；`multi`；`powered`；`several distinct`；`though specific language`；`used are not specified` | 1/1 | 17 |

Lookback在A下：`LG AI Research collaborated with Manus`；`agent that`；`models`；`specific`；`used are not`。命中1/1，额外12。

解读：Manus这个词元的Lookback分数是0.763148，并非接近零：高于A阈值0.746901，低于B阈值0.995855。B下它反而在LG AI Research collaborated with和are报警，漏掉唯一gold对象。LUMINA、NLL和熵命中了Manus，但也有许多额外报警。Manus及后续描述都能从资料找到，合作归属却没有依据；“材料词面可复用但主体关系未核对好”是与本例相容的解释，分数本身不是模型证据使用或因果机制的证明。

## 6. 1858：另一家庭的姓名与女儿死亡年被拼接

`ragognize_test_1858__partial__1`

原文：

> The spouse of Clarence Sexton, Montserrat Utrillo Raymat, died in 1940.

Gold：`Montserrat Utrillo Raymat`；`1940`。

依据：唯一资料Eduardo Schilling (footballer)说：Schilling于1919年与Montserrat Utrillo Raymat结婚，两位女儿分别是Maria Montserrat (1921–1940)和Maria Núria (1927–1938)。没有Clarence Sexton配偶资料。回答挪用了另一人的配偶姓名，又把其女儿的死亡年1940接到该配偶身上。姓名与年份是两段gold。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `The`；`Clarence Sexton`；`Raymat, died in`；`9` | 3/11 | 6 |
| MLP10 | `The spouse of Clarence Sexton`；`Utrillo Raymat, died in 19` | 7/11 | 8 |
| MLP11 | `spouse of Clarence Sexton`；`tr`；`mat, died`；`1` | 3/11 | 6 |
| MLP12 | `The spouse of Clarence Sexton`；`Utrillo Raymat, died in 19` | 7/11 | 8 |
| Lookback | `Clarence Sexton`；`died`；`4` | 1/11 | 4 |
| ReDeEP | `tr`；`9`；`0` | 3/11 | 0 |
| LUMINA | `spouse`；`Mont`；`died`；`4` | 2/11 | 2 |
| NLL | `The spouse`；`Mont`；`died`；`4` | 2/11 | 3 |
| Entropy | `The spouse`；`Mont`；`died in`；`4` | 2/11 | 4 |

Lookback在A下：`of Clarence Sexton`；`serrat U`；`died`；`940`。命中5/11，额外5。

解读：姓名7个词元的Lookback分数为0.185119至0.983108，全部低于B阈值；1940四位仅4报警，因此总共命中1/11。与此同时它圈了Clarence Sexton和died。资料中可找到整个人名及1940，但人物角色绑定错误；本例提示不能用“词在材料里出现过”替代“这个属性属于所问主体”。A阈值命中5/11，也没有完整定位姓名与年份。这里不能把姓名漏检都归因于阈值，因为降低到已冻结A阈值仍会漏掉大部分姓名词元。

## 7. 1029：SNARF内容被写成Twenty Ideas的服务方式

`ragognize_test_1029__partial__1`

原文：

> Twenty Ideas aided founders by providing a new model of digital content through the SNARF acronym, which fosters deeper and more meaningful user experiences.

Gold：`by providing a new model of digital content through the SNARF acronym`。

依据：唯一资料SNARF (acronym)说“The initiative aims to offer a new model of digital content that fosters deeper and more meaningful experiences for users.”。它没有Twenty Ideas或支持创始人、概念打磨的关系。冻结gold圈把SNARF数字内容作为Twenty Ideas帮助方式的完整表达。

| 方法（B） | 实际报警片段 | 命中/金标token | 额外token |
| --- | --- | ---: | ---: |
| LR | `Twenty Ideas`；`AR`；`which` | 1/14 | 3 |
| MLP10 | `Twenty`；`founders by providing a`；`digital content through the SNARF acronym, which fost`；`and more` | 11/14 | 6 |
| MLP11 | `Twenty`；`by providing a`；`digital`；`the`；`acronym`；`and more` | 6/14 | 3 |
| MLP12 | `Twenty`；`founders by providing a`；`digital`；`the SNARF acronym, which fost`；`and more` | 9/14 | 6 |
| Lookback | `Twenty Ideas aided`；`by providing` | 2/14 | 3 |
| ReDeEP | `Ideas aided`；`content through` | 2/14 | 2 |
| LUMINA | `aided`；`providing`；`new`；`through the`；`acronym, which fost`；`user` | 5/14 | 4 |
| NLL | `Twenty`；`aided`；`by providing a new`；`of`；`through the`；`acronym, which fost`；`user` | 8/14 | 5 |
| Entropy | `Twenty`；`aided`；`by providing a new model of`；`through the`；`acronym, which fost`；`user` | 9/14 | 5 |

Lookback在A下：`Twenty Ideas aided`；`by providing`；`which`。命中2/14，额外4。

解读：Lookback在B阈值圈Twenty Ideas aided和by providing，gold只命中by providing的2/14；从a new model…到SNARF acronym多数词元分数极低。资料词汇的连续复用与这一低分区段相邻，支持“局部词面有出处而跨主体的服务关系无出处”这一观察；不能断言注意力已识别或因果决定了资料依赖。LR只命中SNARF里的AR，三个MLP覆盖程度不同，因此需同时保留全表。此例另有预先记录的语义边界争议：初标与裁决把which fosters…视为原文内容功能的改写，独立复核曾提出目标与已实现效果的读法；本报告沿用冻结裁决，不因分数改变gold。

## 审阅与来源核验

逐例已核对当前全部可见资料、原回答、canonical gold及每个报警token坐标；生成文件SHA与gold记录一致，gold词元映射亦与score文件一致。未新增样本筛选、阈值搜索或独立性能指标；表内计数仅解释这7个既定案例。

输入：[Round7 inputs](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/data/inputs.jsonl)；案例名单：[illustrative_examples.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/data/illustrative_examples.json)；阈值：[freeze8.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/results/freeze8.json)。

Gold：[主测试canonical spans](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/data/spans_test.jsonl)、[外部canonical spans](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/data/spans_external_test.jsonl)。分数：[token_scores_main](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/results/token_scores_main.jsonl)、[token_scores_external](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/results/token_scores_external.jsonl)。

1029边界裁决：[external_span_adjudication.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/data/reviews/external_span_adjudication.json)；另一读法保存在[external_span_second_review.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round8_token_localization/data/reviews/external_span_second_review.json)。

| 审阅输入 | SHA256 |
| --- | --- |
| data/illustrative_examples.json | `e5e0bf588940cd3a7dbe0e7d7dbfd2fc55ad2a12510edda3ff8b5fbc4bc10499` |
| results/freeze8.json | `62b29362af26da3c151c54317fad3ef589961cbcb6fbae53717d870655dc174b` |
| results/test_complete8.json | `5a994e9aad5aafaa1274f66d010c10e27805a1e797f7fc8c2b316f54ba86f1c7` |
| data/spans_test.jsonl | `6bb4c80ae0e08dd0ab970c335e643fa541295077e37c431f6293c728f67bf25d` |
| data/spans_external_test.jsonl | `acd3839df7d1732d5336994b08d45558f7d6afea492a34d2fef83c0aee48d5d6` |
| results/token_scores_main.jsonl | `daaa229d1ba202a894c6917e2a1abdc657af8d034297b7c99bb2b5b80b85f29d` |
| results/token_scores_external.jsonl | `3db738670436a3c4f054aeaa8918ea6560e3c22372a767ab3aa41c11a9e4cb7e` |

| 案例 | 源生成SHA256 |
| --- | --- |
| 1 | `82e5d9fe54e12c7afd512db238e9d28d7a2d2201e8f59c706fb5e4e1edb63765` |
| 2 | `672243960cbeeb3f61f08209f3d7320090555a77592c64809987637ce5b7ea88` |
| 3 | `4dab7d16d019977e2930ba861fd1d395bcfcb0f382dfd2ac5f2d692a71c2d72a` |
| 4 | `a7772897f6e7c75e7e96230698f49fa729c4e914e1339bb92a23f174b2768aa9` |
| 5 | `5cabbb620949b272a9f924c7c933c213cb9919b3c5e3a0892ec32dcaeee939f4` |
| 6 | `eb5a4695e4bf6f4737fe9386200657ec334d7cbb11a5cf781d9c875a7383b965` |
| 7 | `8ce1875eb1f702cae9cf8cea281500467d6278f196a303b8d76502c30393deab` |
