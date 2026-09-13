这是冻结结果的事后拆解，没有训练、改阈值或挑选新方法。

| 改动 | 窗口数 | 涉及回答数 |
|---|---:|---:|
| new_FP | 424 | 73 |
| removed_FP | 283 | 82 |
| lost_TP_new_FN | 183 | 60 |
| gained_TP | 113 | 36 |

联合模型正确识别了128/151份风险回答。以下比例都以这128份回答为分母：
- 一个风险窗口也没命中：0/128。
- 至少命中一个风险窗口：128/128。
- 仍漏掉部分风险窗口：62/128。
- 仍把无风险窗口标红：63/128。
- 至少存在一个窗口误报或漏报：110/128。

新增FP与漏检的真整答标签、位置和长度分布：
- new_FP：true_answer_label={'risk_answer': 237, 'supported_answer': 187}；answer_decision={'answer_decision_correct': 248, 'answer_decision_wrong': 176}；window_position_thirds={'first_third': 104, 'last_third': 194, 'middle_third': 126}；answer_length={'17..32': 352, '<=16': 17, '>32': 55}；sentence_position_heuristic={'only_sentence': 403, 'first_sentence': 3, 'last_sentence': 18}
- lost_TP_new_FN：true_answer_label={'risk_answer': 183}；answer_decision={'answer_decision_correct': 111, 'answer_decision_wrong': 72}；window_position_thirds={'middle_third': 74, 'first_third': 24, 'last_third': 85}；answer_length={'17..32': 102, '>32': 67, '<=16': 14}；sentence_position_heuristic={'only_sentence': 178, 'last_sentence': 5}

“整答核查正确”不等于“所有位置均正确”。上面的任何窗口错误比例很严格，应和至少命中率一起读；标红4BPE与金标重叠，也不等于已完整定位到一个事实错误。位置/长度统计仅描述关联，不证明错误由这些因素造成。
