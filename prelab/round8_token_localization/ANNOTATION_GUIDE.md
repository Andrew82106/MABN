# 最小风险片段标注规则

只查看当前可见资料、原始输出和Round7已冻结标签/参考；不看新token分数或热图。分类标签不重写，补充定位边界。发现分类疑点时记录并把该项定位设为unresolved，由根复核。

1. 片段为该事实中实际缺乏依据或错误的最小表达，不把整个风险回答项全部涂红。保留完整回答项作为claim_scope，避免脱离主体/关系解释片段。
2. 属性值：主体已在问题中确定，而出生年、地点、数量或所属类别缺乏支持，圈属性值，例如“1960”“Ranunculaceae”。“born in”等固定框架有其他独立问题才圈。答对但当前无依据同样圈值。
3. 数值差：两年份都有依据、仅差值错，圈“two years”等数值和单位，保留正确before/after。若某年份缺失导致差值和方向均无依据，则圈“two years after”整个比较表达；只声称before/after而无数值时圈该比较词或词组。
4. 关系错配：回答新增加无依据的行为、原因或关系时，圈谓词及对象/补语，不能因其中某个词出现在其他主体的资料里就当作有依据。但问题已指定关系、只要求填属性值时，沿用规则2：问“与谁合作”，圈“Manus”；问“何时开始销售”，圈“April 2025”。这里保留完整claim_scope说明该值属于哪个主体/关系，未圈的框架不等于已证明为真。
5. 新增事实分开圈：一句给出已知年份，又作无依据比较，只圈比较；额外加入城市大小、产业地位等无依据信息另圈一个片段。多段不合并跨过正确内容。
6. 虚假资料陈述：资料明确提到主体，回答“was not mentioned”，圈“not mentioned”；实际未给出生年而正确说明缺失是拒答，沿用原排除状态。明确世界否定如“did not produce any information about…”按完整否定谓词圈，不能只圈not就丢失错误关系。
7. 边界精确到实际文本字符，全局[start,end)；不包含无关句首编号、句末标点或两端空白。命名实体的整个名称保留；近似词、否定词、比较方向等影响真值时不得丢掉。
8. 不能可靠拆分的语义保留unresolved，并说明备选读法。若较长谓词整体无依据，允许长片段，但必须解释；不能把整项广播当作片段金标。
9. 原supported项为空risk_spans；原abstained/tentative/missing或evidence_relation=unresolved项标excluded。只有原risk=1的项需要本轮人工助手逐条边界决策。
10. 非span位置在定位评价中为0，含义是不属于所标风险表达；不声称这些词全是可独立核验的真事实。评测同时给风险项内部的结果和错误span覆盖/多余报警，避免功能词或正常句比例掩盖定位质量。

JSONL基本字段：item_id,row_id,question_id,group_id,split,condition,text,start,end,source_generation_sha256,original_risk,original_stance,localization_status(resolved/excluded/unresolved),claim_scope,risk_spans,annotator,token_scores_viewed=false。

risk_spans各项含start,end,text,claim_id,rationale,evidence_refs；外层rationale和review_note可补充。辅助序列化程序只能定位标注者明确选中的原文字符串，不可自动推断风险词。初标、复核意见和最终裁决分别保存。
