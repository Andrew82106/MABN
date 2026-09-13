# Round9来源审查

目标：200个新题组，time/quantity/location/relation/action各40；来源语义不合格时保留理由，不为了凑数接受。不得按模型回答或检测成绩选样。最终数量和划分在生成前冻结。

每个正式问题正常询问一个事实/关系/行为，删除原数据确认式诱导、错前提和答案暗示；不提示资料不足、不提示可拒答。地点必须是地理地点，学校/公司名字不直接算地理属性。行为使用真实主体的行为、目标或明确归属的理由，不把剧情当现实事件、不把预测/指控改成已证实事实。类别按实际问题语义判断，不直接照搬官方类型。

每条curation JSON至少含：

```json
{
  "candidate_id": "ragognize_test_1234",
  "category": "time|quantity|location|relation|action",
  "question": "A neutral, self-contained question.",
  "subjects": ["Target name explicitly present in question"],
  "reference_answer": "Source-supported answer, preserving qualification",
  "answer_aliases": [],
  "evidence_quote": "Exact contiguous source text supplying the requested fact.",
  "common_quote": "Exact other source sentence(s) with enough context to identify the subject.",
  "partial_quote": "Exact other source sentence(s) that do not answer the question.",
  "rationale": "Why complete supports the answer and the two other quotes do not.",
  "question_rewrite_reason": "Optional reason for neutral rewrite",
  "status": "source_reviewed_candidate"
}
```

三段quote必须是source_pool中该文章原文的唯一精确子串，互不重叠；可含连续数句，不能自行编写背景。完整条件展示common+evidence；局部条件展示common+partial，按原文章次序排列。两条件始终保留同一目标来源标题。partial_quote长度尽量接近evidence_quote，差距可留给root程序挑同文额外句/复核，但不能牺牲语义。不能仅因标准答案字符串不出现就认定缺证：还须排除别名、另一表述和可推导答案。

干扰资料另在最终组装时分配：使用未被选为任何正式目标的独立真实文章和精确事实片段，同一题两条件干扰相同，优先同属性且主体不同。不得把另一划分的目标证据作为干扰输入；每个干扰来源只归属一题组，或先合并来源连接组再划分。禁止使用旧实验目标/来源和跨组复用同事件。共同来源、目标别名和实体碰撞须保留审查记录。

subjects/reference_answer/answer_aliases是来源审查与标注辅助，不进入新特征匹配器；推理候选证据只从问题与展示资料得到。源题、源类型、标题、URL/revision/time、全文哈希由candidate_id从source_pool回填，不能手填伪来源。

审查者保存选择与排除理由，初审与复核区分；正式冻结之前允许纠正出处或问题。最终数据、词元标签仍是助手审核的预实验资料，不能称为研究者人工金标准。
