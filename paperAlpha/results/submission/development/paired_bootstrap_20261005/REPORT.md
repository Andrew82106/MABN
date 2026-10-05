# Paired family-cluster bootstrap

This audit compares the frozen primary OOF scores with the strongest
same-information MAS proxy and the learned graph/local controls. It resamples
the 186 scenario families together, so paired episodes stay within their
family. It is not an external confirmation cohort.

| comparison (ours minus baseline) | ΔAUROC | 95% CI | ΔAUPRC | 95% CI |
|---|---:|---:|---:|---:|
| trust/reputation risk | 0.090 | [0.077, 0.103] | 0.176 | [0.153, 0.200] |
| graph-features logistic | 0.096 | [0.084, 0.108] | 0.147 | [0.127, 0.167] |
| local-only logistic | 0.132 | [0.117, 0.149] | 0.190 | [0.166, 0.216] |

All 2,000 valid family-cluster draws were positive for both deltas in every
comparison. The full JSON outputs are local generated artifacts from
`scripts/paired_group_bootstrap_v1.py` and are not required for a clean
checkout.
