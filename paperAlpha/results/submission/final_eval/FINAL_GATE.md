# Final experiment gate

This file is the compact submission audit for the current two-line MAS risk
monitor. It supersedes exploratory result folders as the active reading order;
raw runs remain immutable for reproducibility.

## Frozen headline

On `independent_mas_v3` (4,000 traces, 1,345 positives, 186 topology
families), family-disjoint five-fold OOF evaluation gives:

| method | F1 | AUROC (95% CI) | AUPRC (95% CI) | Brier |
|---|---:|---:|---:|---:|
| runtime logistic | 0.653 | 0.805 [0.791, 0.818] | 0.653 [0.622, 0.683] | 0.167 |
| graph-features logistic | 0.578 | 0.724 [0.708, 0.739] | 0.527 [0.500, 0.558] | 0.194 |
| local-only logistic | 0.564 | 0.688 [0.672, 0.705] | 0.485 [0.460, 0.516] | 0.204 |
| AgentMonitor-style statistics + logistic (adapted target) | 0.581 | 0.712 | 0.503 | 0.197 |
| two-line BN fusion | **0.667** | **0.820 [0.804, 0.836]** | **0.675 [0.644, 0.707]** | **0.160** |

The two-line model is a white-box monitor of observable evidence, not a claim
that the protected API model has become internally transparent and not a causal
discovery method.

The repaired canonical run also changes the workflow-line Brier from the
pre-fix saturated value 0.639 to 0.236; this is why the current tracked
`metrics.json` and `REPORT.md` must be used instead of older exploratory tables.

## Checks completed

- Nested OOF fusion, family-cluster bootstrap intervals and public-field release validation.
- Field-parity ablation: removing `permission_mismatch` gives AUROC 0.811/AUPRC 0.661.
- ESWA-neighbor reliability-weighted Bayesian control: AUROC 0.763/AUPRC 0.596, below the proposed two-line fusion 0.820/0.675 on the same grouped split.
- Paired family-cluster bootstrap against that control: ΔAUROC 0.056 [0.045, 0.067] and ΔAUPRC 0.079 [0.061, 0.098], positive in all 2,000 resamples.
- Explanation-faithfulness audit: a top path is found in 71.85% of episodes; deleting it lowers the score by 0.042 on average.
- Lanyun `glm-5.3-flash` transfer: 64/80 cases complete; fusion AUROC 0.854/AUPRC 0.824. This is a separate transfer/cost cohort, not the 32-complete GLM blind-confirmation packet.
- A2ASecBench API leave-one-attack-family-out adapter: 80 episodes; fusion AUROC 0.833/AUPRC 0.899, retained only as an integration benchmark because the reference harness emits part of the labels.
- Leakage-controlled A2ASecBench role audit: 80 episodes; outcome metrics and `observed_violation` excluded, two-line AUROC/AUPRC 0.788/0.835. Labels remain benchmark-design roles, not human adjudication.
- QuadSentinel external replay: completed, but only 5.19% action coverage and a different target.
- AgentMonitor-style statistics control: completed under the same five-fold family split; AUROC 0.712/AUPRC 0.503/Brier 0.197. This is an explicit risk-target adaptation of the original task-performance monitor, not a direct original-target replication.
- MAST/MAD external text audit (boundary-clean v2): 1,642 trajectories; structural-only AUROC/AUPRC 0.295/0.675 and semantic-extension 0.468/0.763 after stripping the terminal `Evaluation` block. It is secondary transfer evidence only; released annotations are not an independent human-gold confirmation of the main security target.
- Prefix alarm audit: completed; action-before stopping gives 6.7% negative false-alarm rate (177/2655) and 15.5% positive pre-action recall, while the 20.0% any-time rate includes post-action alarms; it remains a weak diagnostic rather than reliable early warning.
- Grouped uncertainty audit: completed; runtime-BN intervals abstain on 6.1% of episodes (coverage 93.9%, mean width 0.030) and report selective coverage-risk without calling the interval a calibrated confidence interval.
- Monitor-side scaling audit: completed; p95 is 0.532 ms for 1--3 events, 1.398 ms for 7--10 events and 1.905 ms for 11+ events, excluding API/semantic extraction.
- Same-episode cost audit: 10/80 LANYUN episodes align API and semantic ledgers; serial service-time estimate p50/p95 47.386/83.429 s, local semantic+graph+BN p50/p95 0.691/0.996 ms. Coverage is partial and this is not a deployment throughput guarantee.
- LANYUN transport sanity: `LANYUN_SANITY.md` records a fresh HTTP 200 check for `glm-5.3-flash`; this is availability evidence only.
- LANYUN DAG front-end smoke: latest v5 parsed 17/18 as acyclic; exact node/edge F1 0.081/0.000 and semantic node/edge F1 0.938/0.747. This remains a front-end check, not a monitor result.
- Release-scoped regression suite: the 10 tracked test modules (36 tests) pass after
  including the dynamic hierarchical/final-feature dependencies in the package.
  The wider workspace contains historical HarnessAudit tests that require the
  optional `openai-agents` SDK and is not part of this release claim.
- Unified topology communication smoke ([report](../development/communication_mode_smoke_lanyun_20261005_r1/REPORT.md)): 12 LANYUN episodes across chain/fork/join/review and direct/summary/vote modes, 33 requests with 25 completed and 8 failed; failures are retained and monitor/evaluator artifacts are separate. Task text explicitly states policy situations, so this is transport/topology evidence only.

## Remaining gates

- An independently adjudicated/licensed MAS confirmation cohort is still missing; the current blind materials contain 152 complete episodes across two API/model conditions, but use shared simulator policies, remain below the 160-episode target, and the annotator packets are unfilled.
- The prefix alarm audit is complete but weak at the fixed 5% false-alarm operating point; it must not be advertised as reliable early warning.
- A strict same-information zero-shot LLM-judge adapter is implemented and tested. The local gateway still returns an upstream account-pool 503. After targeted retries, the LANYUN `glm-5.3-flash` run has 4,000/4,000 valid predictions; original failures and retries remain separately logged.
- On all 4,000 episodes, the two-line monitor scores AUROC/AUPRC/F1/Brier 0.820/0.675/0.633/0.160 versus the LLM judge's 0.771/0.578/0.571/0.317. This is a full-coverage same-information auxiliary baseline, not a published-method reproduction or a universal superiority claim; details are in `LLM_JUDGE_BASELINE.md`.
- A matched-information development audit is also present: a flat Logistic reaches mean AUROC 0.817 versus 0.816 for the two-line BN on five repeated family holdouts, while the BN has slightly higher mean AUPRC/F1 (0.666/0.659 versus 0.661/0.658). This prevents a broad “beats every same-information model” claim; details are in `MATCHED_INFORMATION_AUDIT.md`.
- Calibration is now reported both episode-micro and family-macro: two-line BN Brier 0.16035/0.14609 versus runtime logistic 0.16714/0.15985, with family-bootstrap intervals recorded in `GROUP_CALIBRATION_AUDIT.md`.
- The matched-information audit still lacks a preregistered untouched confirmation cohort and paired significance intervals for the flat controls; the current 4,000-episode OOF package remains development evidence, not final generalization proof.
- The release-specific judge/annotation tests pass (6/6); the broader historical `paperAlpha/tests` collection still needs the optional `openai-agents` SDK and is not claimed green in the current environment.
- Uncertainty/abstention is reported as an epistemic missing-provenance diagnostic, not a calibrated interval.
- Component-wise online cost is now closed in `development/e2e_cost_closure_20261005/REPORT.md` (API transport, semantic extraction and monitor path); a same-episode end-to-end latency and deployment-level RSS/training-cost measurement are still pending.

Until these gates are closed, the numbers above are a strong development package
and a reproducible paper candidate, not a final superiority claim for a journal.

## Journal alignment used for the protocol

- **Expert Systems with Applications**: closest methodological reference is Sadak, “A multi-agent LLM framework with Bayesian fusion and safety guardrails for ATC-pilot communication error detection,” DOI `10.1016/j.eswa.2026.132241`. We adopted its multi-agent role separation, low-positive-rate reporting, strict holdout, fixed-FPR and latency/cost expectations, but do not treat its ATC labels as a directly comparable baseline.
- **Expert Systems with Applications**: Chahine, “Separating intent from execution,” DOI `10.1016/j.eswa.2026.133781`, motivates the separate normative/operational lines and deterministic policy checks.
- **IEEE TDSC**: Yang et al., “Cracks in Collaboration,” DOI `10.1109/TDSC.2026.3670889`, supplies MAS threat/topology stress-test guidance rather than a directly comparable risk monitor.

The journal catalog records institutional CAS Q1/Q2 evidence and official publisher scope, but the current-year official CAS database entry has not been authenticated; no final partition claim is made here.
