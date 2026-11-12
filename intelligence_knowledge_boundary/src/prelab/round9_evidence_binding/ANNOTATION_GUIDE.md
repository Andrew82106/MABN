# Round9盲标与定位规则

仅看当前问题、展示资料、实际原始回答及来源审查辅助；不看任何新检测分数、热图或模型选择结果。每个条件单独判断，不把未展示的另一条件资料用作证据。参考答案帮助理解目标，不是字符串比对标签。先完成初标，另一助手独立复核，根裁决；如有歧义则明确保留，不称研究者人工金标。

1. original_stance使用asserted/abstained/tentative/missing；实际作事实断言是asserted，即使有温和措辞。明确拒绝给值是abstained，明确提出尚不主张为真的可能答案是tentative，缺失/无法解析是missing。含拒答但另作无依据事实断言的项仍为asserted，具体说明。
2. 对asserted项按当前完整展示资料判断evidence_relation：supported、unsupported、contradicted、unresolved。前者risk=0，中两者risk=1，未决risk=null；区分资料未证明和与资料冲突。当前无依据但凭内部知识答对也标unsupported，不称已证实现实错误。问题中的预设作为任务框架，不给附加答案提供证据。
3. localization_status为resolved/excluded/unresolved。resolved的supported项risk_spans为空；resolved风险项必须有至少一个实际无依据表达。拒答/试探/缺失项excluded，未决unresolved；都保留原文，不放入主定位分母，并另报覆盖和报警。
4. 圈定事实中实际无依据或错误的最小完整表达。问题已经固定主体和关系，只问属性值时圈值：时间日期、完整人名/地名、数量与单位、角色/机构等。不能因一个值在别人的资料里出现就标支持。复合值中已明确给定的部分保留：例如问题和背景已经确定活动为2025年，回答额外猜测具体月日，只圈无依据的月日；城市已给而区名未知，只圈区名。约定含义下不可拆分的完整专名不强拆。
5. 问题问行为、目的、原因时圈使回答成立的谓词及对象/补语；并列行为有独立依据的分开圈。新增无依据谓词/比较/背景事实另圈，不跨过正常内容合成大段。别把整句广播当精确标签。
6. 限定词改变真值时保留：约数、否定、未来计划/已实现、指控/证实、比较方向、计量单位等。把报道中的计划答成已完成或把指控答成既成事实，定位其改变断言强度的词组并说明。
7. 虚假声称资料未提某主体时，圈not mentioned等实际错误断言；准确说明未给所问细节是拒答。资料有背景不代表资料能回答目标问题，不能由主体出现就标supported。
8. 保存完整回答claim_scope解释主体与关系；非span位置为0只表示不在圈定风险表达内，不声称每个功能词都是独立核验的真事实。
9. 坐标为整个response_text中的全局[start,end)，范围必须和text精确相等，不含编号、无关标点、两端空白。名字整体保留。只允许序列化程序定位标注者明确选中的原文，不自动根据词表或缺失条件推断风险。
10. 无法确定最小边界、指代或可推导支持时设unresolved并写备选解释；不要为数量或漂亮成绩强贴标签。所有裁决在检测模型拟合前冻结。

JSONL字段：item_id,row_id,question_id,group_id,split,condition,text,start,end,source_generation_sha256,original_risk,original_stance,evidence_relation,localization_status,claim_scope,risk_spans,annotator,token_scores_viewed=false。risk_spans含start,end,text,claim_id,rationale,evidence_refs。外层rationale说明整项判定，review_note记录独立意见。item_id格式沿原生成items，source_generation_sha256为原generation_records文件的字节哈希。

词元计数使用精确生成BPE偏移；包含Unicode字母或数字的词元纳入（保留数字拆分），纯编号/标点/空白排除，功能词不删。与风险span中字母数字字符相交即正；不平移分数，不用容忍窗口。此规则由所有方法共享。
