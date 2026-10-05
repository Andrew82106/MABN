# Independent confirmation cohort status

The frozen annotation material now contains 160 complete API-agent episodes:

| condition | model | episodes | topology allocation | requests | failed requests |
|---|---|---:|---|---:|---:|
| A | LANYUN Qwen 3.6 Flash | 80 selected from 120 complete | 20 each: chain/fork/join/review | 220 | 0 |
| B | LANYUN DeepSeek V4.1 Flash | 80 | 20 each: chain/fork/join/review | 220 | 0 |

The selected packets are generated at
`results/submission/development/frozen_confirmation_blind_packets_20261007_balanced_160_v2/`.
The source queue is content-bound by SHA-256
`de537a1339b92343dfb8bde07b308fe46b0ef98cb9fad5ef8040bc218730814c`.
The digest is over the sorted relative file names and bytes of the 160 source
`monitor.json` files, with NUL separators.
The two packet files contain blank annotation fields. The packet compiler removes
episode/scenario/request identity, model/provider fields, evaluator labels and
post-hoc outcomes before the packets are handed to raters.

This closes the planned data-volume and two-condition preparation gate, but not
the independent-truth gate. Both conditions use the same hand-authored policy
simulator, and no safety label is inferred from the model response. Two raters
must independently fill `risk_label`, first violation index, class, evidence IDs
and rationale; agreement and Cohen's kappa must be reported before adjudication.
Until then this is annotation material, not a human-gold test set or a new model
performance result.
