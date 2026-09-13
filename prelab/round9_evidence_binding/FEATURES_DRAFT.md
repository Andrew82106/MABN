# Round9：主体与属性候选资料的白盒特征草案

状态：root已确认首版接口，`src/binding9.py`与11项CPU测试已完成。未加载7B/运行新GPU、未修改Round7/8。第9节记录实际交付接口；此前段落保留设计理由。

## 1. 可做什么，以及不能把它解释成什么

检验当前输出token的注意力是否偏向“问题所问主体与属性的候选句”，还是偏向其他主体、其他属性或此前生成文字。这里的候选句来自可见文本匹配，可能漏掉正确证据，也可能误收不支持答案的句子；它不是标准证据，更不是正确性证明。新分类器必须在独立训练/验证划分拟合，不能直接把比例叫幻觉概率。

Root拟定的新200题组、每组2条件适合独立执行。应保留“主体仍出现但属性缺失”和“双方主体及属性都出现但关系被错配”的困难例，避免主成绩只反映主体消失。Round8七个已看结果的案例仅作设计动机，不纳入新题的参数挑选依据。

## 2. 已审现有实现及必须规避的输入

已读：

- `round7_evidence_grounding/src/attention7.py`：Qwen2的q/k投影及RoPE/GQA重建正确可复用；每8个响应query计算因果注意力行，逐层汇总，不存全部S×S。
- `round7_evidence_grounding/src/model7.py`：Qwen2.5-7B NF4/bfloat16，28层、28个query heads、隐藏维3584，SDPA；第21层取block后原始残差，第28层取最终RMSNorm之后。
- `round8_token_localization/src/extract8.py`：精确生成token对齐、21/28状态、同形状未来扰动检查可沿用。序列长度变化会带来NF4数值差异，不能把小幅数值不同直接当未来泄漏。

**白名单只含实际展示内容：** `system, prompt, questions: list[str], passages: [{title,text}]`，以及精确生成token IDs。新matcher不接收原始row字典。

**禁止读取：** `subjects, aliases, references, reference_evidence, source_sent_ids, source_question_id, standard_answer, condition, coverage, removed_*, split, risk, labels` 等。外部0188的旧 `row.subjects` 实际含Google Cloud、EXAONE 3.0等当前partial资料未展示的信息，因此即使字段叫“主体”也不能使用。row_id仅给调用方命名缓存，不能进入匹配、数值特征或排序。

不得直接复用旧 `surface_features`，其中使用了 `row.subjects`。不得由 `generated.items` 的完整未来正文、最终parse_ok、结束位置或最终句长来选择候选资料/计算当前token特征。原items只可在特征算完后用于离线关联或评测。

## 3. CPU候选资料方案：每个问题固定一次

### 3.1 定位可见文本

用chat template重建真实prompt，并逐项验证精确input_token_ids。按展示的 `[n] title\ntext` 顺序建立来源标题、正文和正文句子/短分句的字符范围，不用数据集隐藏的证据句号。

正文分句来自 `passages[].text` 的原字符串；保留缩写、姓名首字母、括号生卒日期，不把句点机械地全部断开。不使用只存在元数据中的“sentences”作为语义线索。索引重复出现的句子时按当前位置逐段查找，不能全局find第一个。

正文token分配给最大非空字符交集的可见句子；并列时固定取较早句。由此每个正文token只属一个句子，候选/其他互斥。来源标题与编号单独留存，不能把问题中的主体名算作来源证据。整资料Lookback仍保持旧版包含标题/分隔符的定义，新的主体特征主要在正文上计算，两者名称明确区分。

### 3.2 问题主体与属性

先实现可解释的静态规则，不引入额外LLM。

1. 主体从**实际问题字符串**提取。覆盖现有的 `In what year was X born?`、`In what year did X die?`、`To which plant family does X belong?`、城市/来源国/频率、`who X collaborated with`、`year of death for the spouse of X` 等句式。比较题可由该次展示的前两个属性问题确定两个主体；这是问题文本，不是隐藏subjects。一般新题用疑问框架去除后的连续实体短语和问题关键词，失败时标明question_parse_unresolved。
2. 字符规范化只做NFKC、大小写、空白及等价连字符/引号统一，同时保留原字符映射。
3. 可从问题本身去掉明确消歧后缀如 `(musician)` 形成别名。额外别名仅接受当前展示文本明确写出的 `X, also known as Y`、`X (Y)` 等与已识别主体相连的表达；括号不是无条件别名（日期/职业/地理注释排除）。
4. 不自动把任意名字首词、姓氏、首字母缩写当同一实体；例如只有Cedric而无完整名字映射时不能确定就是Cedric Bixler-Zavala。可保留弱词面候选，但不能提升为严格匹配。
5. 属性词使用冻结的小词典及问句词形：birth/born；death/died；family；from/origin/location/based；publication frequency及daily/weekly/monthly等；collaborate/partner/work with；spouse/marry；amount/quantity等。生卒括号的双日期结构可作为时间属性线索，但它仍是候选规则，不提供标准答案。
6. 复合关系保留链条，如“X的配偶何年去世”需要spouse/marry线索以及death/date线索；单有其他家庭的某个1940不能成为X的严格候选。未知属性退回问句非停用关键词，记录解析不确定，不宣称资料一定不足。

词典/模板在开发材料上完善后冻结；不得为某个测试题加别名、改关系词、挑候选。

### 3.3 句子分组与保守角色限制

对每个问题q、每个主体a独立给正文句子标以下**匹配类型**：

- T_a：有主体明确提及或明确同指，同时有对应属性/关系线索。时间属性优先限定到该主体后接born/died/生卒括号的局部结构；不能因为一段里同时出现父亲、配偶、女儿及1940，就把所有日期都分给父亲。
- E_a：主体相关但目标属性线索未匹配到。以主体为标题的文章且正文以He/She/It/His/Her/The band等直接回指开头可作弱话题匹配；句子改谈另一个明确主体时，不将标题自动附到所有属性上。
- A：匹配所问属性线索但未匹配任何当前目标主体，即潜在“其他主体同类属性”。
- O：其他正文。

T只叫**主体－属性候选**，不叫“支持”或“正确证据”。一些关系/代词靠规则无法确定，进入弱匹配或解析不确定，不靠世界知识补全。

T_a可保留最多3句，按预先固定的匹配等级排序（明确主体+直接属性结构 > 明确别名+属性 > 保守话题回指+属性），并列取展示次序；只有满足条件才进入，空集合不强行凑top1。这个上限是算力/候选粒度选择，不看输出分数或标签。E/A/O覆盖余下正文。比较题保留T_1/T_2各自mask，另算union；不能只用union掩盖一方完全没有候选。需要同时命中两个主体的句子可出现在两者mask中，union只计一次；E/A/O与union仍互斥。

固定候选计划不使用生成答案内容。故Manus、姓名日期等token刚写出时不会因它们出现在输出中，就回头把相关资料改选为“正确候选”。

## 4. 因果路由与时点

候选计划由所有已展示问题和资料预先确定。生成过程中，只根据**到当前token为止**已读入的行首编号选择当前q。单问题始终q=0；多问题按已经完整出现的 `1.`/`2.` 等编号切换。编号尚未完整、未知编号或非预期回退时，记录route_valid=false，不能查看后续编号来修复当前状态。

可在精确响应字节流上增量解析；UTF-8字符分在两个BPE token时，只处理截至当前token已完整解码的字符，不能通过最终offset映射偷看半个字之后的内容。最终offsets仅为输出结果绑定坐标，不能参与候选语义选择。

新注意力和隐藏状态都用post-read位置P+j，仅可见输入及回答0..j。本草案将“此前回答”严格记作0..j-1，并把当前token自身注意力单列。原全局Lookback仍按旧式包含0..j，以便逐token复核一致；两种分母不混用。j=0时此前回答为空，相关比值标invalid而不是臆造测量。

## 5. 每层每头的具体信号

令a_lhj(k)为当前query对key k的因果注意力。对集合G定义：

`mass(G)=sum_{k in G} a(k)`；`density(G)=mass(G)/|G|`。

B为全部正文token，T为当前q的候选union，E为当前主体但属性未匹配的正文，A为其他主体同属性候选，Rprev为已生成0..j-1。分母epsilon仅防浮点零，集合不存在时通过valid标记区分；不把空候选与测得极低注意力混为一谈。

建议首版8个head通道，固定顺序：

| 名称 | 定义 | 意义及局限 |
| --- | --- | --- |
| body_mass | mass(B) | 当前是否关注正文，未证明支持 |
| target_share | mass(T)/mass(B) | 正文注意力中目标候选的份额 |
| target_other_density_ratio | density(T)/(density(T)+density(B\\T)) | 长度校正的候选对其他正文比较 |
| target_prev_density_ratio | density(T)/(density(T)+density(Rprev)) | 候选对之前生成文字；不含当前self |
| entity_only_share | mass(E)/mass(B) | 是否停留在该主体的其他信息 |
| attribute_distractor_share | mass(A)/mass(B) | 是否偏向其他主体的同类属性 |
| self_mass | a(P+j) | 单列当前token的自注意力 |
| pair_balance | min(mass(T_1),mass(T_2))/(max(mass(T_1),mass(T_2))+epsilon) | 比较题两个主体候选的平衡，仅两者都存在时有效 |

每个通道同时输出valid。正文存在但T为空时，T的mass确实为0；密度比较则invalid并用数值0占位，不能将其描述为已观测到“不关注候选”。pair_balance单主体不适用。未能路由的问题仅保留全局LB、hidden和body/self信号，候选通道invalid，不删除该token或样本。

低维首版：将28层分为1–7、8–14、15–21、22–28，对每段内所有head等权平均，得到4×8=32维。valid与候选存在性等单独保留。可另存全部28×28×8 head通道供**训练集内**正则化线性探针作对照；不能按测试指标挑头。首轮不追加max/std/top-k等大量汇总，避免无界调参。

额外输出原式 `token_lookback[R,784]`，用于直接对照旧globalLB。不要将新body-only分组比例冒充原Lookback。

## 6. 拟定接口与文件格式

```python
def build_candidate_plan(tokenizer, visible: dict,
                         input_token_ids: list[int]) -> dict:
    # visible仅含第2节白名单；无模型forward；逐问题静态mask。
    ...

@torch.inference_mode()
def extract_binding_features(tokenizer, model, row, generated,
                             candidate_plan=None, query_batch=8,
                             store_head_features=True) -> tuple[dict, dict]:
    # 包装层只复制可见白名单；row_id仅在返回meta用于缓存标识。
    # generated只读精确input/response IDs和原文，不读items做当前语义路由。
    ...
```

candidate_plan（CPU）：

```text
version, rendered_prompt_sha256, questions_sha256
context_token_indices                 原globalLB来源范围
body_token_indices, title_token_indices
sentence_units[]                     可见title/text原始字符范围、分配token索引
question_plans[]:
  question_index, question_text
  parsed_entities[], attribute_type, attribute_terms[]
  parser_status
  target_token_indices_by_entity[]   每主体T_a
  target_union_token_indices, entity_only_token_indices
  attribute_distractor_token_indices, other_body_token_indices
  matching_trace[]                   规则、字面命中、候选排序、弱匹配原因
```

arrays：

```text
token_ids                        [R] int64
response_token_offsets           [R,2] int32
token_question_index             [R] int32；unknown=-1
token_route_valid                [R] bool
hidden_21                        [R,3584] float32，block后原始残差
hidden_28                        [R,3584] float32，final RMSNorm后
token_lookback                    [R,784] float32，原式
binding_lowdim                   [R,32] float32，4层段×8通道
binding_valid                    [R,8] bool，mask存在性/路由控制
binding_head （可选）             [R,28,28,8] float32
binding_feature_names            [8] unicode
layer_band_bounds                [4,2] int32，闭区间[1,7]等
```

meta另存输入/精确生成hash、代码/模型配置hash、候选匹配trace、可见文本解析覆盖率、原始集合大小、路由异常、P/R/Qbatch、耗时与peak显存。集合大小/目标出现等是**表面匹配控制变量**，单独 `surface_controls` 保存，训练时可与白盒模块做同口径消融；不能将这些变量的贡献叫注意力创新。

schema还需root确认命名，但特征轴和时点可按上面直接实现。不得在提取端打开references、labels、检测分数或其他条件数据来修正mask。

## 7. 单次重放与显存

旧attention7需要先取final hidden再做ECS/PKS；R9核心不需要二者。新增独立hook：

1. 一次 `model.model(..., use_cache=False, output_hidden_states=False, output_attentions=False)`；
2. 每层的实际q_proj/k_proj输出结合真实RoPE与GQA，在每8个响应query上重建attention行；
3. 对当前query的静态候选mask及此前输出mask即时求和，写入CPU数组，然后释放attention临时量；
4. 同一次forward捕获layer21输出，返回时取layer28 final hidden；同时计算原LB；
5. 不请求全词表logits，不做PKS、不保存full attention，不切换模型的SDPA实现。

R7旧缓存只有整资料比例/ECS，不保留每句注意力，因此**不能从旧NPZ反解这些新分组比例**，至少需要这一次重放。所有候选变体在同次attention行上汇总，不能每个候选再跑模型。

在S≤3328（3072输入+256输出）、H=28、query_batch=8时，一个float32注意力块约2.84MiB；展开的bf16 K约22.75MiB。head结果最大约6.13MiB/回答（256×28×28×8×4），可直接保存在CPU。3584维21/28状态两份共约7MiB/最大256token。完整模型内存仍占主体，上述不是总显存承诺。

R8已有日志：260次hidden-only重放合计58.56秒，普通行中位0.216秒，最大peak约5.396GiB；R7含ECS/PKS的501条attention记录中位1.608秒，个别异常长耗时不能作正常估计。R9单遍提取预计位于二者之间或附近，先按短回答每条0.3–2秒做**未验证预算**：400份回答约2–14分钟，另加模型加载、文件I/O；更长上下文需重估。不得把这当已测性能。

root先选择一短一长开发样本做GPU冒烟，记录实际耗时、显存和query_batch；建议默认8，如8GB紧张降4，不自动截断资料。标注、CPU候选构建可并行；GPU只有root统一调度，ReDeEP/LUMINA另行基线重放不与核心争显存。

## 8. 验证与后续消融建议

CPU/tiny-Qwen检查足够明确后再交root：

- 同一visible输入附加/改写隐藏subjects、aliases、condition、references等，结果完全不变；包装白名单必须保证该性质。
- 只改变未来响应token，当前路由、候选mask、特征保持不变；候选mask本来就与任何输出无关。
- 只改变字面主体与属性归属的手工句，T/E/A按规则变化；含两家庭、多个日期与title话题转移不能无条件并入T。
- 原LB逐token与旧attention7一致；手工张量验证质量比/密度比、空集合、pair、GQA/RoPE、未来mask。
- tiny模型与eager全注意力对照；7B一次短/长样本按R8的同形状未来扰动法验证因果性，第21/28状态对齐。
- matcher解析失败、空候选和整项未解析都报告覆盖，不通过丢弃困难例提高成绩。

当前正式比较固定为PLAN与EVALUATION_DRAFT中的16种方法；不新增head级模型或表面特征模型。后续若要进一步隔离字面匹配的贡献，可比较：仅可见匹配/存在性/集合大小、globalLB、binding-only、同一套表面控制+globalLB/白盒binding；这些扩展不是本轮已完成的实验。训练/验证按新题组分割，同题两个条件不跨组；词典、归一化、LR正则和阈值都只能用开发/训练/验证，测试冻结后不改。

可说的初始贡献是“检验细分候选资料注意力是否比整资料注意力更适合定位关系错配”。在完成上述对照前，不预称新算法优于已有方法，不把主体未出现的简单线索算方法创新。


## 9. 已确认并实现的首版（以本节接口为准）

`extract_binding_features(tok, model, visible, generated)`返回arrays、meta。主数组键为`hidden_28`、`hidden_21`、`lookback_features[R,784]`、`binding_features[R,32]`、`token_nll`、`token_entropy`、`token_ids`、`token_start`、`token_end`；另有`binding_valid[R,8]`、`response_token_offsets`、`token_question_index`、`token_route_valid`作对齐与审计。只存固定32维汇总，不输出head级binding数组、不拟合额外head模型。global Lookback仍保留各头784维。

meta包含`binding_feature_names`、候选计划与匹配trace、输入/精确回答hash、源码hash、耗时及显存。candidate_plan使用`question_plans[].entities/aliases/attribute/attribute_terms/relation_chain/parser_status`；aliases全部从问题或展示文本产生，不读取同名输入元数据。`source_generation_sha256`、`arrays_sha256`及运行协议hash由runner侧车统一写。

候选计划同时验证questions列表与真实Questions区完全相同、passages标题正文与真实Search results区完全相同。时间属性增加最近显式主体与亲属角色检查；John Smith标题下的Jane Smith出生年不能仅因同姓而算John的候选。未知文本仍可能错匹配，因此保留候选性质和解析覆盖说明。

已运行：`prelab/.venv/Scripts/python.exe -X utf8 -m unittest discover -s prelab/round9_evidence_binding/tests -p test_binding9.py -v`。11项均通过；tiny模型为CPU、28层、隐藏16维、2个query heads/1个KV head，使用真实本地tokenizer，未加载7B权重。检查包括可见白名单不受隐藏字段影响、属性缺失与他人同属性、家庭/多主体日期归属、比较两主体独立mask、增量路由、手工质量/密度公式、prompt一致性、与eager各头注意力/21/28hidden/NLL位置一致、未来同形状扰动与截断前缀一致。实际7B耗时/显存仍待root冒烟确认。
