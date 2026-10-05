# Current candidate result — two-line MAS risk monitor

This file is the single pointer for the current candidate. Detailed numbers are frozen in `../../independent_mas_journal_v1/metrics.json`; the model contract is in `../../independent_mas_journal_v1/MODEL_SPEC.md`.

The compact gate audit is in [`FINAL_GATE.md`](FINAL_GATE.md); exploratory folders are not active result pointers.

## Main grouped OOF result

Dataset: `independent_mas_v3`, 4,000 public traces, 1,345 positives, 186 topology families. Five-fold `StratifiedGroupKFold` keeps each family entirely in train or test. Labels are joined only by `episode_id`; label, mechanism, hidden fields, terminal effect and family are excluded from features.

The release validator confirms 4,000 unique trace IDs, 4,000 unique label IDs, exact set equality, and no forbidden public keys; its frozen result is `release_validator_v1.json`.

The fusion head uses inner OOF scores from both the normative workflow BN and the runtime BN. Classification thresholds are selected from inner OOF training scores; confidence intervals use family-cluster bootstrap.

| method | F1 | precision | recall | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| workflow + observed-endpoints BN | 0.516 | 0.351 | 0.978 | 0.517 | 0.340 | 0.639 | 0.645 |
| runtime BN only | 0.626 | 0.493 | 0.859 | 0.716 | 0.545 | 0.196 | 0.113 |
| runtime logistic (same public runtime features) | 0.653 | 0.562 | 0.778 | 0.805 | 0.653 | 0.167 | 0.019 |
| graph-features logistic (learned MAS graph control) | 0.578 | 0.449 | 0.813 | 0.724 | 0.527 | 0.194 | — |
| local-only logistic (learned MAS local control) | 0.564 | 0.447 | 0.763 | 0.688 | 0.485 | 0.204 | — |
| two-line BN fusion | **0.668** | **0.598** | 0.756 | **0.820** | **0.675** | **0.160** | 0.019 |

As a direct journal-neighbor control, a transparent reliability-weighted
Bayesian fusion baseline (train-only smoothed likelihoods and line reliabilities)
reaches F1/AUROC/AUPRC/Brier = 0.604/0.763/0.596/0.459 on the same grouped
split. The proposed two-line fusion therefore improves AUROC by 0.057 and
AUPRC by 0.079 over this Bayesian-fusion control; this is the relevant
comparison to the ESWA Bayesian-fusion reference, not a claim of reproducing
its ATC task. A paired topology-family bootstrap gives ΔAUROC 0.056
[0.045, 0.067] and ΔAUPRC 0.079 [0.061, 0.098], with positive differences in
all 2,000 resamples. See `reliability_bayes_baseline_v1/REPORT.md` and
`primary_vs_reliability_v1/REPORT.md`.

The graph-features and local-only rows are independent five-fold controls on the
same family split; full details are in
`../../independent_mas_learned_baselines_v1/REPORT.md`. Transparent topology,
edge-count and taint-path rule controls are reported separately in
`../../independent_mas_mas_baselines_v1/REPORT.md`; they are sanity controls,
not published-method reproductions.

To check feature parity, the same evaluator was rerun with the runtime
`permission_mismatch` component removed. The two-line fusion then gives
AUROC 0.811, AUPRC 0.661, Brier 0.165 and F1 0.650, versus 0.820/0.675/0.160/0.668
in the full model. This narrows, but does not erase, the gap to the matched
runtime-logistic control; it is an ablation, not a new headline result. See
`../../independent_mas_journal_parity_v1/REPORT.md`.

All frozen OOF methods now also have family-cluster bootstrap intervals in
`all_methods_ci_v1/REPORT.md`. The thresholded F1 in that supplementary table
is explicitly F1 at 0.5; the main table's fold-local F1 remains the primary
classification number.

At fold-local empirical 5% train-negative FPR thresholds, the two-line fusion has test FPR 0.052 and recall 0.303. This is a development threshold diagnostic, not an independently calibrated deployment guarantee. Family-cluster bootstrap 95% intervals are AUROC [0.804, 0.836], AUPRC [0.644, 0.707], F1 [0.644, 0.692].

## Scope

The monitor combines a declared workflow/knowledge line with an observable execution line. In this implementation, the so-called workflow vector also includes observed source/sink endpoints; it is not a pure workflow-DAG-only ablation. Its variables, factors, edge gates, uncertainty policy and evidence paths are inspectable. The protected API LLM remains black-box. This is risk assessment, not causal discovery or an absolute-safety guarantee. The full-trace table is an episode classification task; the separate prefix task is the only evidence used for an early-warning claim.

The prefix check reports pre-action AUROC 0.735 on the same synthetic benchmark; it is a development early-warning check, not yet a real API deployment result. Missingness strata and CPU inference cost are recorded in `../../independent_mas_journal_v1/strata.json` and `../../independent_mas_journal_v1/latency.json`.

The separate prefix alarm audit calibrates a fold-local 5% train-negative-FPR
threshold in the pre-action regime. On held-out families it gives 15.5% recall
before a privileged action among positives with an observable action, but 20.0%
negative false-alarm rate under the shift; median lead is four events for the
55 timely alarms. This supports risk ranking more than dependable early alarms.
See `prefix_alarm_v1/REPORT.md`.

The grouped uncertainty audit reports the lower/point/upper runtime-BN interval
and selective coverage-risk tradeoff in `uncertainty_coverage_v1/REPORT.md`.

The monitor-side scaling audit measures JSON-to-feature extraction plus BN
propagation at p50/p95 **0.424/0.532 ms** for 1--3 events, **0.928/1.398 ms**
for 7--10 events, and **1.483/1.905 ms** for 11+ events. API generation,
network transfer and semantic extraction are intentionally reported separately;
see `online_scaling_v1/REPORT.md`.

The DAG-construction front end was also smoke-tested against the available
LANYUN endpoint with `glm-5.3-flash` on 18 requests (six workflows and three
prompt strategies). The latest run parsed 17/18 responses as acyclic graphs;
exact node/edge F1 was 0.081/0.000, while deterministic semantic alignment
scored 0.938/0.747. This is a format/transport check, not a risk-monitoring
result; see `lanyun_dag_smoke_v5/REPORT.md`.

## External API transfer check

`../development/balanced_api_mas_transfer_v1/REPORT.md` evaluates the frozen model on eight completed, balanced two-agent episodes generated through the local API gateway, with four independent reference-policy positives. The acquisition audit verifies 8/8 runs and no monitor-label leakage. The hierarchical fusion ranking is AUROC 0.750 and AUPRC 0.750; its fixed 0.5 alarm threshold is conservative under this domain shift. Because this is a small same-gateway cohort, it is transfer evidence only, not a powered journal result. The earlier eight-case branch check remains at `../development/external_api_mas_v1/REPORT.md`.

The active knowledge-compiled binding-consistency extension is reported separately at `../development/binding_consistency_extension_v2/REPORT.md`. It uses only public workflow, authority, message and action events with fixed auditable rule weights and no refit or threshold tuning. On 11 completed cases from the expanded balanced cohort it reaches F1/AUROC/AUPRC = 0.750/0.800/0.782; the older branched cohort gives 0.889/0.969/0.950. One API failure is retained in the raw cohort and excluded from scoring. These small transfer diagnostics show where explicit authority/resource binding can complement the two BN lines; they do not change the frozen headline result or establish deployment performance.

For the same 11-case expanded cohort, the frozen two-line fusion transfer score has AUROC 0.700 and AUPRC 0.673; its fixed 0.5 threshold raises no alarm. This is consistent with the binding-template result: public authority/resource evidence helps, but the current model is not calibrated for this domain shift.

A separate second-endpoint diagnostic uses the user-authorized Lanyun service with `glm-5.3-flash`: 10 of 12 predeclared cases completed and 2 failed stage outputs were retained. The frozen two-line model ranks the 10 completed cases perfectly (AUROC/AUPRC=1.000/1.000), but its fixed 0.5 threshold raises no alarm; the sample is too small and the labels come from the same post-run simulator policy interpreter. Full endpoint and cost records are in `../development/balanced_api_mas_transfer_lanyun_v1/REPORT.md` and `../development/api_cost_report_v1/lanyun_r13.json`.

An expanded Lanyun run predeclared 40 cases and completed 26 (14 failures retained), with 9 completed positives. Without fitting on API labels, the frozen runtime BN reaches AUROC/AUPRC 0.944/0.927 and the two-line fusion 0.889/0.927; neither raises a fixed-0.5 alarm. Transport p50/p95 is 9.688/36.774 seconds over 196 requests and 218,838 tokens. This is stronger transfer evidence but remains one hand-authored simulator with post-run labels, not the independent confirmation cohort required for submission. Details are in `../development/balanced_api_mas_transfer_lanyun_r14/REPORT.md` and `../development/api_cost_report_v1/lanyun_r14.json`.

A larger Lanyun run predeclared 80 cases and completed 64 (16 failures retained), with 24 completed reference-policy violations. The frozen model ranks the completed actual-effect target at runtime-BN AUROC/AUPRC 0.872/0.827 and two-line fusion 0.854/0.824; the fixed 0.5 threshold again raises no alarms. The separate pre-execution policy-intent target gives the public knowledge rule F1/AUROC/AUPRC/Brier 0.947/0.950/0.950/0.055, while the runtime-only BN gives AUROC/AUPRC 0.545/0.690. This is a larger second-endpoint transfer diagnostic, but it still uses the same hand-authored two-agent simulator and post-run reference policy; it is not an independent confirmation cohort. Transport p50/p95 is 7.429/28.555 seconds over 416 requests and 459,902 tokens. Details are in `../development/balanced_api_mas_transfer_lanyun_r15/REPORT.md`, `../development/lanyun_knowledge_attempt_r15/REPORT.md` and `../development/api_cost_report_v1/lanyun_r15.json`.

A second API/model condition through the local gateway predeclared the same 80 cases and completed 65 (15 failures retained), with 32 completed reference-policy violations. The frozen runtime-BN and two-line fusion both reach AUROC/AUPRC 0.750/0.746 on the actual-effect target. The separate policy-intent rule reaches F1/AUROC/AUPRC/Brier 0.904/0.913/0.913/0.091. This is cross-endpoint/model transfer on the same hand-authored simulator, not an independent topology or human-adjudicated cohort. The acquisition recorded 406 requests and 310,540 tokens; the separate cost report is `../development/api_cost_report_v1/local_r20.json`. Details are in `../development/external_api_mas_transfer_local_20261004/REPORT.md` and `../development/lanyun_knowledge_attempt_local_20261004/REPORT.md`.

The unified topology transport smoke additionally exercised chain/fork/join/review with 8 low-cost LANYUN episodes: 22 requests, 15 completed and 7 failed. It validates topology scheduling, failure retention and monitor/evaluator separation only; task text explicitly states the policy situation, so it is not a risk-accuracy or independent-label result. Details are in `../development/unified_topology_api_20261005_r1/REPORT.md`.

As a leakage-controlled A2ASecBench API audit, the predeclared attack/control role was used as the target while `observed_violation` and all outcome metrics were excluded from features. Leave-one-attack-family-out over 80 episodes gives the strict two-line audit AUROC/AUPRC **0.788/0.835** (F1 0.682), versus a matched flat logistic AUROC/AUPRC 0.788/0.835. This is a stronger external API benchmark check, but its role labels are benchmark-design labels rather than human adjudication. See `../development/a2asecbench_api_final_20261002/strict_role_audit_v1/REPORT.md`.

The same 40 observable prefixes were also evaluated on a separate pre-execution policy-risk target: a case is positive when its declared request is out of scope or revoked, even if the model refuses before execution. The public knowledge rule (authority scope, revocation, and resource consistency) reaches F1/AUROC/AUPRC/Brier = 0.865/0.867/0.843/0.128; the knowledge-runtime union has 0.865/0.775/0.843/0.120. This is the intended white-box-style knowledge line, but it is target-specific and still hand-authored; it is not merged into the main operational-risk score. See `../development/lanyun_knowledge_attempt_r14/REPORT.md`.

An external boundary check uses the public AgentLeak release: 4,258 eligible MAS traces, 937 request-identity groups, five generating models and four verticals. The target is final coordinator-to-user literal disclosure, and prefixes stop before the final message. The late-prefix frozen runtime BN / two-line fusion reach AUROC 0.532 / 0.561 and AUPRC 0.333 / 0.353. This negative transfer result is retained because the release has a fixed coordinator/worker topology and the target literal can be absent from all earlier observable messages; it is not merged into the operational-risk headline metric. Details and leakage audit are in `../development/agentleak_external_journal_v1_r3/REPORT.md` and `../development/agentleak_external_projection_v1/ADAPTER_REPORT.md`.

An external QuadSentinel replay is also available on the paired HarnessAudit-derived cohort: 12 episodes, 289 native actions, 15 covered actions and 274 uncovered actions. Covered-action precision/recall are both 1.000 (TP=3, FN=0, FP=0, TN=12), but coverage is only 5.19% and the target is current native-policy detection rather than future system-risk forecasting. It is therefore reported as an external boundary/control, not combined with the main AUROC table or used for a superiority claim. See `../development/quadsentinel_replication_20261001/REPORT.md`.

The public MAST/MAD release provides a separate text-only external audit: 1,642
trajectories across 11 MAS-by-benchmark groups, with 1,237 positives under its
released MAST failure annotations. The boundary-clean v2 adapter strips the
terminal `Evaluation` block before feature construction. Structural-only gives
AUROC/AUPRC/F1/Brier **0.295/0.675/0.859/0.197**; adding the fixed semantic
runtime extension gives **0.468/0.763/0.859/0.189**. This target is any released
MAS failure, not the main authorization/taint risk target, so it is secondary
transfer evidence. The separate human subset has only 19 traces (18 positive)
and is descriptive only. See
`../development/mast_two_layer_v2_20261005/REPORT.md`.

An explanation-faithfulness audit on the same 4,000 out-of-fold episodes found an inspectable highest-scoring path in 71.85% of episodes. Removing that path edge lowered the monitor score by 0.042 on average (0.066 on positive episodes versus 0.030 on negative episodes); the positive-episode drop rate was 0.348 versus 0.154. This is a counterfactual faithfulness diagnostic, not causal attribution or proof that the black-box model used that path. Details are in `explanation_counterfactual_v1/REPORT.md`.

## Submission gate

This is not yet a journal-ready final claim. The expanded Lanyun and local-gateway cohorts now exercise 80 predeclared cases each, an external QuadSentinel replay is complete, family-cluster intervals are unified, field-parity and explanation-faithfulness audits are included, and the prefix/uncertainty diagnostics are now explicit. QuadSentinel has only 5.19% action coverage and a different target; the prefix audit also exposes a held-out false-alarm shift. The package still needs an independently adjudicated or licensed MAS cohort and an online latency/cost report that includes semantic extraction and monitor scaling. The API and QuadSentinel results are therefore reported as transfer/boundary evidence, not as a claim of statistical superiority. The A2ASecBench API queue remains a pipeline sanity check because its labels are tied to the reference runner.
