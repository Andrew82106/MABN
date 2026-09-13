# 固定强组合在训练集完全漏检的 12 处片段

**有比“输入不够多”更具体的问题：资料中的动作被找到了，但动作属于谁、在哪个条件下、先后如何，未必被正确比较。** 同时，一些漏检涉及推导或术语边界，不能全部当成无争议的模型错误。

## 选择与范围

仅用原 **634 fit 回答、168,123 个窗口**。固定模型 `citation_alignment_lr_v1 / harp_claim__two_scores_and_citation__C0.1`，沿用原校准窗口阈值 **0.7370759393261086**；不重选阈值、模型或标签。选中模型的训练混淆计数复核一致：TP 14,927 / FP 7,906 / FN 6,550 / TN 138,740。

“完全漏检”指：**所有与该人工 span 的风险词元相交的 4 raw-BPE 窗口均未报警**。各类型先取具有内部完整四风险词元窗口的片段，再按数值 response_id、原字符 start 排序取最多四处；不按分数大小或案例易解释程度挑选。三类均取得四处，共 12 处、10 份回答；17319 的两个片段是同一错误在同答中的重复，不能算独立事件。

原训练 span 中，显性冲突 79/109 处完全漏检，隐性冲突 4/7，显性无依据 52/343；这些是本模型固定阈值下的检出统计，**不是标签噪声率**。以下是先按规则选例、再阅读解释的定性检查，不能据此估计争议比例。

完整问题、三篇资料、原回答、人工类别/meta/坐标、相交窗口索引及分数均在 [CASES.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/fit_relation_error_review_v1/CASES.json)。以下判断是助手解释，**不是新的人工标注或裁决**。资料本身的世界真实性未在本次核查。

## 显性冲突

### 1. 11991，字符 [734,814)：把已有建议说成资料未提供

问题：如何促进头发生长。原风险句是 “The passages do not provide specific information on how to speed up hair growth.”

资料 1 明写 “To increase hair growth, constantly massage the scalp daily” 及定期修剪、饮食建议；回答前文也已列出这些内容。**现有输入包含核查所需内容**。风险关系是“这组资料有没有给出针对问题的建议”，不是某个方法名是否出现。末尾拒答式表述可能压低风险，但本次没有做因果归因；资料 3 对修剪能否加快生长存在限定，仍不等于所有资料都未给建议。

16 个相交窗口全漏；其中 12 个内部四风险词元窗口。原文：[fit 第 216 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:216)。

### 2. 12213，字符 [378,502)：正文事实正确，段落归属错误

问题：如何烤 rump roast。标记句要求切 4–5 个口、填盐胡椒和半瓣蒜；它位于回答 **“Passage 2:” 标题下**。

资料 3 明写 “Prep the roast by making 4-5 slits ... 1/2 of a clove of Garlic”；资料 2 只介绍慢炖等方法及时间。**完整回答和资料提供了归属证据，但单取该句不含错误来源标题**。只判断任意资料是否支持动作会觉得它正确；需要把前面的标题作用范围带进比较。这是本例的具体盲点，不是食材知识不足。

40 个相交窗口全漏；21 个内部窗口。原文：[fit 第 238 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:238)。

### 3. 12405，字符 [88,393)：来源归属之外，还把操作顺序反转

问题：如何安装蜂群 NUC。风险段说 Passage 1 给出入口、等待 24 小时的步骤，并要求 **“before moving the NUC to its permanent location”** 先打开入口等待。

资料 3 的顺序是：**先** “place in its permanent location”，**再** “Remove the block ... and wait at least 24 hours”。资料 1 是安装后的描述。原人工 meta 指出来源编号错误；本次阅读还发现风险段内有 before/after 的顺序差异，**不新增 span 或改类别**。原输入足够比较两者；只看“位置、打开、24 小时”等词是否得到支持会漏掉先后关系。

76 个相交窗口全漏；47 个内部窗口。原文：[fit 第 257 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:257)。

### 4. 12417，字符 [459,523)：资料编号正确，内部步骤编号错误

问题：南瓜意面。风险句说 “Let ... cool and mash it (passage 2, step 5)”。

资料 2 是 **“4 Let cool & mash flesh. 5 Set aside.”** 动作确实存在，但属于第 4 步。**输入有准确的两级位置关系**。仅核资料 2 是否支持该动作，或只有 passage 级引用信号，都无法解决 step 4/5 的归属。

25 个相交窗口全漏；10 个内部窗口。原文：[fit 第 259 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:259)。

## 隐性冲突

### 5. 12969，字符 [1539,1601)：有条件的精细过滤变成一般步骤，同时存在边界

问题：巧克力镜面淋面。风险句为 “Strain the glaze through a fine mesh strainer before using it.”

资料 3 对这一具体器具的表述是 **“If you do get ... a lot of air bubbles ... strain it through a fine mesh strainer”**。回答删掉条件；此前又正确复述过这个条件。**资料包含条件比较所需信息**，而最终总结把它丢失。

但同一资料更前面也有一般性的 “stir until smooth and strain”。因此“过滤”动作本身并非无依据；争议在于是否能将它与后面的具体滤网说明合并为常规步骤。保留原隐性冲突标签，不把该例称为已证明误标或完全无争议错误。

17 个相交窗口全漏；11 个内部窗口。原文：[fit 第 318 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:318)。

### 6. 13359，字符 [1144,1158)：术语替换，未找到明确反向事实

问题：酥脆豆腐条。人工标记的是配料表中的 **“(for egg wash)”**。

资料 3 明写 **“Add beaten eggs to a small bowl”**，随后 **“dip in egg, and roll in cornflakes”**。回答同一配料位置写 Eggs，并说明用途。可见资料已经支持“鸡蛋用于外层处理”的动作，但没有定义 egg wash，也没有给出与该术语明确相反的事实。**仅凭本次原文，不能把这个命名差异可靠解释成事实关系冲突**。

这是需要原标注口径解释的边界，不能据漏报反推模型一定缺少鸡蛋用途知识；gold 不变。7 个相交窗口全漏；1 个内部窗口。原文：[fit 第 358 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:358)。

### 7. 17319，字符 [900,1009)：重叠资料被接成两次煮沸

问题：菲律宾式炖牛肉。原风险句要求 “boil **again** ... another hour”。回答前一段已要求煮沸、炖约一小时，之后才重新放回牛肉及酱汁，再执行一次。

资料 3 的连续过程是：**Return beef ... Add water, tomato sauce ... Bring to a boil then simmer, about an hour ... Add potatoes ... another 6 to 8 minutes**。资料 2 从同一过程的煮沸部分开始，并非另一次炖煮。**完整上下文可判断重复动作和先后关系**，但单独拿“一小时煮沸”与任意片段比对，会得到支持。需要区分资料重叠和事件重复。

29 个相交窗口全漏；14 个内部窗口。原文：[fit 第 146 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:146)。

### 8. 17319，字符 [1647,1771)：同一错误在总结步骤中再次出现

同一回答的第 5 步已经煮沸、炖一小时，第 7 步才放回牛肉，第 8 步又写 **“bring ... to a boil again ... another hour”**。证据与上一例相同，风险是把一段流程执行两次，不能只看第 8 步中每个动作都在资料里出现过。

29 个相交窗口全漏；19 个内部窗口。它与例 7 是同答重复，**不是另一条独立失效证据**。原文同第 146 行。

## 显性无依据

### 9. 11871，字符 [844,874)：新增问题限定，又否认现有液体清洁说明

问题只问安全清洁电脑屏幕。gold 圈 “with liquid or other solutions”，meta 指出问题原本没有这个限定。实际上下文是 **“The passages do not provide specific instructions for cleaning the screen with liquid or other solutions.”**

资料 2 却详细给出 **“1 part isopropyl alcohol and 1 part distilled water”**，随后喷在布上擦屏幕。**资料足以反驳这条“未提供”断言**。本例原类别仍是显性无依据；不能为了说明方便改成冲突。需将限定短语放回整句否定范围，而不是单独检查 liquid 是否合理。

6 个相交窗口全漏；2 个内部窗口。原文：[fit 第 202 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:202)。

### 10. 11997，字符 [257,310)：把贵族质量差异移到皇帝材料上

问题：印加服装。风险片段断言 **“the Inca Emperor wore clothes made of finer materials”**，句末归于资料 2。

资料 2 说普通人使用粗糙织物，并介绍皇帝衣服穿一次后烧掉；没有直接说明皇帝的材料。资料 3 则说 **“The Nobles wore far better quality than the general Indians.”** 这给出一种可能的背景推导，但从“贵族/衣服质量”到“皇帝/制作材料”仍换了主体和属性。

**原输入足够检查是否直接支持，不足以证明现实中皇帝一定不用好材料。** 本例涉及严格资料支持与常识推导边界，不能将原“无依据”标签解释成世界事实为假。14 个相交窗口全漏；8 个内部窗口。原文：[fit 第 217 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:217)。

### 11. 11997，字符 [624,642)：把戴帽子的共同主体扩展给披斗篷

同答原句为 **“Both men and women wore cloaks ... and used caps ...”**；gold 只圈 Both men and women。

资料 3 相邻两句分别是 **“Men wore a cloak over the tunic. Both men and women used cap.”** 共同主体本来只作用于戴帽子，回答把它扩展到披斗篷。**资料包含清楚的主体—动作边界**。词几乎全都出现过，但组合关系越界；这比“没有读到资料”更具体。

7 个相交窗口全漏；1 个内部窗口。原文同第 217 行。无依据不等于资料明确断言女性绝不披斗篷。

### 12. 12159，字符 [435,468)：预热是否可由热烤架和中火推出

问题：烤玉米。gold 标记 **“Preheat the grill to medium heat.”**，理由是未直接提到预热。

资料 2 要求放到 **“the hot grill”** 上，资料 3 要求 **“Grill the corn over medium heat with the lid closed.”** 因此 hot 与 medium heat 都在输入里；新增的是将它们写成操作前的 preheat 步骤。**这是推导粒度边界，不是缺少温度证据**，也没有直接相反的事实。不能把该条漏检简单当成模型识别不了中火或顺序；保留原人标，不认定已证误标。

12 个相交窗口全漏；6 个内部窗口。原文：[fit 第 232 行](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/fit.jsonl:232)。

## 共同失效点与一个具体候选

1. **有内容，不等于关系正确。** 例 2–4 的来源/步骤归属，例 11 的主体范围，例 7–8 的单次/重复事件，均可在当前完整输入中比较。当前 LR 最终收到的是两项已压缩风险分数和八项词面引用特征；即使上游看过完整原文，也不保证这些关系保留在最终分数中。
2. **否定与限定需要整句、跨句范围。** 例 1/9 要检查“资料未提供”的对象，例 5 要保留 if 条件；仅有动作词相似度不足。但本次没有逐层干预，不能证明具体哪一个模块导致漏报。
3. **部分金标界线并不等同于现实真假。** 例 6/12 的命名与自然操作推导，及例 5/10 的组合推导，应保持原标签并公开解释限制。这批定向漏检例不能代表全数据，也不能据此删难例。

若下一步只增加一项，建议考虑**关系最小对比辅助训练**：只从允许训练资料构造一对词面高度相同的陈述，保留版本严格对应原主体、条件和单次步骤；另一版本只交换主体范围、删去条件或把一次动作写成 again。带上对应原文及必要前文，使核查器学习比较“关系是否仍成立”。这种合成对须单列为派生辅助监督，不能冒充现有人工标注；尤其不从例 6/12 的争议推导自动生成负例。它比继续添加普通覆盖率更直接针对已看到的失效，但**本次仅提出候选，没有导出数据、训练或收益保证**。

## 补查：此前确实丢失了独立标题的作用范围

已只读核对两套**实际** `claims.jsonl`，不只是推测代码行为，记录见 [SCOPE_CHECK.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/fit_relation_error_review_v1/SCOPE_CHECK.json)。

- 12213 的 claim 6 `[293,303)` 是 `Passage 2:`，被识别为来源 2；风险 bullet 是 claim 8 `[378,502)`，parser 为 `none`，旧五项 claim 值为 `[0.9,0,0,0,0]`。普通词面相似度很高，却没有被比较到它所归属的来源 2。
- `cited_source_semantic_v1` 对同一 bullet 保存 `valid_ids=[]`，所以此前 6,252 对来源语义核查**没有核查这条 bullet 的三来源支持度**。被识别的独立标题不能替后续内容完成核查。
- 相反，12417 的行内 `(passage 2, step 5)` 确实识别了来源 2，缺的是更细的步骤编号；标题继承不能解决它。12405 的 `Passage 1 states ...` 是普通行内陈述，不是独立标题，不能任意扩散给后文。
- 11997 另有句尾 `(Passage N)` 被切成独立 claim，正文 claim 无引用；这是另一个切分问题，不混入标题继承修复。17319 的普通总结也不能继承前一段的行内引用。

最窄只读规则（完整独立 `Passage 1/2/3:` 标题、其后 `* / + / -` bullet、遇普通正文关闭、不覆盖已有或未知引用）在 **fit634** 中可补 24 答、241 个 claim、3,847 个词元，影响 4,952 个原窗口。覆盖统计不使用 gold，未查看校准正文；这不是预期收益或错误数。本 12 例中，12213 的风险 bullet 满足此规则；13359 配料表前已有普通总结，所以正确关闭，不继承更早的 Passage 3。

这个确定的输入丢失支持先做小范围 CPU 标题作用域对照，再考虑上面的辅助训练。后续若采用更宽规则或全793范围，应另冻方案，不改本次只读产物。
