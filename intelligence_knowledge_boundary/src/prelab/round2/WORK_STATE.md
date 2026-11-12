# Completed round 2 — 2026-09-10

The expanded experiment is complete: five planned runs, 110 newly reserved human-news sources, six supervised alarm-control checkpoints, final metric and feature audits, generation alignment checks and results/REPORT.md. No experiments are still running. Conference baselines are local adaptations, not full paper reproductions. No subagents were used. The notes below this completion snapshot are historical progress records, not pending instructions.

Final evidence: 7B Trivia after-state bag AUROC .868 vs NLL .806; news HaMI-before within-answer localization .578 vs fully position-supervised after-state LR .688. Low-FP alerts remain poor (that LR original test 3.1% clean-answer false alarm, 4% error-answer recall). Limited position labels improve ranking more consistently than model scaling: 24/60/120/240 labels gives large attention LR .580/.590/.632/.660, small .582/.592/.626/.668.

New source-disjoint confirmation: 80 clean + 30 error news, frozen before alarm-control fitting, no parameter or threshold selection on it. Same attention features: bag-supervised MLP local AUC .570, token-supervised MLP .709 (paired source bootstrap improvement CI [.060,.216]); vs local-supervised LR .688, MLP difference CI [-.009,.050], so no proven architectural gain. Added clean-max-risk loss gives .730 local AUC but recall 17.8% at 9.2% clean-answer false alarm vs plain MLP 7.8% at 5.4%; not reliable improvement. Checkpoint hashes and full predictions are under results/confirmation.

RAG auto labels are NOT a factual-error benchmark: evidence review of all 14 candidate-positive test answers found 4 clear conflicts, 3 supported, 7 ambiguous/nonanswers/question defects. Review by execution assistant only, not independent human annotators. Preserve primary results as reference-matching diagnostics. RAG all 1050 original token traces saved; 11 re-tokenization mismatches repaired before fitting. Both fixed-sample seeded regeneration audits pass; earlier Trivia is only sampled, not fully live-traced. Final feature and independent global/alarm metric audits pass. See final_audit.json and alert_metric_audit.json. Suggested next research direction is limited position annotation for evidence-conflict localization/human review assistance, not a claim of solved live OSINT hallucination monitoring.

## Historical progress below (superseded)

Interpreter: `D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe`; shared CA torch inherited with user authorization. GPU RTX3070 8GB. 7B NF4 Qwen weights verified, GPU model allocation 5.20 GiB. Serialize substantial GPU processes: overlapping probe fitting and 7B extraction caused Windows shared-memory paging. CPU-only supervised sklearn fits can overlap 7B generation. Evaluator supports `--cpu`.

Completed this round:

- Qwen2.5-7B-Instruct-bnb-4bit download, pinned Unsloth revision and full SHA validated.
- Frozen TriviaQA 2700 questions (2000/300/400), SQuAD-derived three-fact RAG 1050 contexts (700/150/200), source/title isolation recorded.
- News small and large: all 394 original human-annotated RAGTruth responses re-extracted, before/after token, attention ratios, probability/evidence contrast. Both model feature audits passed, and selective-hook extraction equals standard hidden states.
- News small: all weak, fully local-supervised and 10/25/50/100% local training-label budget probes evaluated. Limited-label validation uses only bag labels. Token AUROC means: few24=.582, few60=.592, few120=.626, full240=.668. Full local-label validation attention baseline=.671. HaMI before=.556, after=.566. Bag TF-IDF=.649; naive source copy local=.510.
- News large: weak fitting and preliminary CPU evaluation complete. HaMI before bag=.585/local=.578; after .576/.580; mean-linear bag=.634; attention weak bag=.615/local=.550; source+attention .603/.533. Validation-selected uncertainty fails test bag=.499/local=.509. Full supervised and limited-label fits still running, so current large-news metrics are incomplete.
- Small Trivia: all 2700 own answers generated and fitted/evaluated. Usable train1761 (337 correct1424 mismatch), val264, test345 (75 correct270 mismatch). Bag HaMI before .767, after .789, uncertainty .791, mean linear .777, NLL topk .752, TF-IDF .676, shuffled .521. This is benchmark-target correctness, not verified factuality of every claim or localization.
- Small RAG pilot stopped after ~215 training outputs due format/usable-label problems; preserve outputs; no completed small RAG comparison claimed. Decision JSON recorded.

LIVE tool sessions (poll/log before deciding):

LATEST UPDATE (takes precedence over older session descriptions below):

- Large Trivia generation finished all2700. Session23869 is now generating RAG large (about190/1050, ~1.4sec/response at update).
- Old large-news CPU supervisor finally saved supervised_before.pkl (val token AUC .6173). Session21437 verifies this pickle and stops the old original sequential supervised process to avoid duplicate work; it may have completed already.
- Session78755 separately runs CPU parallel-C fits for `--supervised-kinds after lookback --workers 3`; same sklearn algorithm/seed/grid, independent processes. Log `supervised_news_large_parallel.log`; first after-layer0 candidates complete.
- Session2531 waits for before and lookback checkpoints, then runs large `limited_labels.py`, then complete CPU evaluation. Original63886 limited-label continuation will normally not run because its child was intentionally stopped after preserving before.
- CPU worker parallelism is ordinary numerical fitting, not subagent delegation.
- Actual generation tracing found2/189 initial RAG outputs whose decoded text retokenizes differently. Added `original_token_offsets` and exact-ID extraction to engine, and `fix_generation_tokens.py`. Existing live RAG process has old loaded extraction code; AFTER generation, repair mismatches before any RAG fitting. `analyze_run.py` now automatically inserts this repair before labeling RAG large. All189 UTF-8 byte offset checks passed. The repair records canonical-vs-original differences and keeps output text identical.
- `audit_generation.py` added for sampled old Trivia/RAG exact seeded regeneration. Run after large inference and before/after fitting with exclusive GPU, as convenient. New RAG output rows already save emitted token IDs.
- `reviewed_sample_metrics.py` added and automatically runs after full large-Trivia analysis. Large 30-item audit confirmed2 correct variants mislabeled by matching and2 ambiguous/questionable items; secondary sample metrics exclude2 and correct2, without changing the prespecified primary labels.
- Small and large feature audits both passed; selective-layer hook test passed. Large-news weak CPU evaluation exists but remains incomplete until local/limited checkpoints evaluated. Small-news full limited-budget curve now evaluated:24=.582,60=.592,120=.626,240=.668.
- Report builder now marks missing supervision methods as pending, not only missing datasets. `verify_round2.py` requires expected method families for every run and independent metric checks.

- Session 63886: after weak large-news completion, runs `probes.py --task news --model large --supervised` (CPU sklearn; log `logs/supervised_news_large.log`), then `limited_labels.py --model large` (log `logs/limited_labels_news_large.log`). No evaluator follows automatically.
- Session 23869: completed small feature audit and small-news re-evaluation, now running `run_inference.py --model large --task trivia` (log `logs/inference_trivia_large.log`), then automatically `run_inference.py --model large --task rag` if Trivia succeeds. At checkpoint ~1050/2700 Trivia done, ~0.4 sec/response. RAG has not started. Parent PowerShell waits normally.
- Earlier sessions 20737 preliminary large-news CPU eval, 76295 large feature audit + weak fitting, 38357 small selective-hook audit + limited-label fits, 83092 large news generation are complete or finishing; confirm if needed. Main old sessions no longer relevant.

Remaining work:

1. Monitor large own Trivia then RAG; no GPU fitting while these run. CPU fits/evaluation okay. Label generation runs already have partial label files from audit; always relabel full completed outputs before fitting.
2. When large-news supervised+limited fits complete, re-evaluate news large with `--cpu`. Current preliminary summary lacks those methods.
3. After own generation is done, run `analyze_run.py --task trivia --model large` and `--task rag --model large` sequentially. This does full labels, weak fits, local-supervised diagnostics (RAG only), evaluation. Watch failures/class counts; do not silently treat no data as success.
4. RAG weak fits include a new prespecified exploratory fact-level MIL: raw JSON/answer-object syntax determines 3 answer spans without reading aliases or local labels; average features within fact then bag max. `fact_after` (original HaMI loss) and `fact_support_lookback` (BCE) train bag-only; project fact scores to answer tokens. Not within-fact pinpointing or claimed novelty. Implemented but not run yet. Check this new path on completed RAG data before claiming success.
5. Audit full RAG automatic labels. `review_bundle.py --task rag --model large` makes fixed15correct15error sample, no predictions. Review evidence and answers and save caveats/decisions. Automatic mismatches are candidate errors; human news is primary localization evidence.
6. `audit_generation.py --model large` and `--model small` after GPU generation finishes: sampled seeded regeneration checks same text and token IDs against stored replay features; takes ~8 examples per completed own task. New RAG generation will save actual generation token IDs and `token_replay_exact`. Old Trivia runs predate tracing and require sampled audit. Do not call unverified replay a complete live-state trace.
7. Re-run `audit_features.py --model small` at the end to cover all files (CPU loop but loads0.5B GPU; serialize). Large audit already passed initial cache/hook/causality, see `feature_audit_large.json`.
8. Implement/report reviewed Trivia sample secondary evaluation if useful: large `results/trivia_large_label_review.json` has30 reviewed items, 2 semantically correct mismatches (Lily-of-the-Valley, Romans), 2 ambiguous/questionable-reference exclusions. Review bundle persisted; no probe predictions had been seen. Main matching rules are frozen; do not silently change only test labels. Small initial label-rule audit is separately documented.
9. `build_report.py` produces results/REPORT.md, comparison.png and all test risk HTML. Current report stale/incomplete (only small news initially); rebuild after full results and inspect plot. Add a concise evidence-based interpretation and continuation direction, without claiming weak token localization automatically works or that probes beat simpler baselines unless observed.
10. `verify_round2.py` independently recalculates AUROC/F1 and news local AUROC, verifies all five required runs and all expected methods/features. It requires feature_audit_small/large and selective_capture audit. Run, investigate failures, finish provenance/link checks, update README status. Verify actual generation trace results and dataset label limitations too.
11. Goal complete only after all required expanded work and final interpretation delivered; do not mark complete while large runs or evaluations remain.

Label caveats and frozen revisions:

- News uses released Evident Conflict spans; relative to given evidence, not separately established world truth.
- Trivia: aliases that merely repeat question subjects excluded, multiple sentences and obvious nonanswers excluded. Still imperfect reference matching and historical question validity. Small-test label audit before any Trivia fits caused these shared revisions, transparently documented in `results/label_rule_audit.json`.
- Large Trivia audit (30 test answers, prediction blind) found2 correct variants among15 mismatches,1 underspecified Fly answer,1 questionable Europe-reference question. No global matching revision from this audit; record secondary reviewed sample. Do not claim automatic mismatches are all factual hallucinations.
- RAG: parse exact array3strings, or 3 answer fields with exact accompanying question order; normalize Unicode dashes; partial/inflection overlap ambiguous; refuses/truncations excluded. Position labels only within answer content, excluding JSON punctuation/ignored answers. Training weak branch never reads local_annotations.

Artifacts/source:

- `BASELINES.md` accurately distinguishes direct official HaMI network/loss import, adapted attention features/local supervision, naive controls, limited-label diagnostics and unimplemented literature.
- HaMI official commit3d277605534999d2756a6eeef759d0a91199ad58; NeurIPS2025 final paper. Lookback-Lens commit e0a1fa3a898fbf6512af7be5567dea8ffe7a6620; EMNLP2024. Both papers local prelab/references. ICR ACL2025 and Simple Factuality Probes FindingsEMNLP2025 checked as related work, not implemented. Max-pooling Network Revisited arXiv2605.08863v2 is a2026 preprint, not verified conference acceptance; max-pooling itself is not novel.
- New `prelab/.gitignore` excludes round2 data/logs/checkpoints/predictions and cloned Lookback-Lens. Existing round1 results preserved.
- Root README prelab links round2 as in progress. No doc/ long-term route changes this round; avoid editing doc without its required routing/impact workflow.
