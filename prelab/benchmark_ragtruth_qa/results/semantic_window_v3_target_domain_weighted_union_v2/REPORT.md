# Target-domain weighted evidence-union v2

The frozen 70-column v3 `backbone_union` design (base26 + evidence-union44) and C=0.001 were used unchanged. Exact response/claim/microclaim/hypothesis joins mapped all 34,919 claims to all 653,979 shared four-BPE windows; every claim has window coverage. The first 168,123 design rows reproduce native-only v3 bit-for-bit.

The expanded evidence status is consistent: the 158,929-row `missing_requests` file is the immutable request manifest. Extraction is complete for all 39/39 shards, and the later 25,864-row aggregate is hash-bound and complete.

| auxiliary alpha | native window F1 | window AP | window AUROC | native answer F1 | answer AP | answer AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.609944 | 0.588455 | 0.875333 | 0.759531 | 0.786105 | 0.796320 |
| 0.25 | 0.610149 | 0.587412 | 0.877859 | 0.753623 | 0.777808 | 0.793081 |
| 0.5 | 0.610247 | 0.588272 | 0.878757 | 0.752941 | 0.779623 | 0.792832 |
| 1 | 0.607921 | 0.586865 | 0.879269 | 0.754438 | 0.784658 | 0.793171 |

Selected alpha: **0.5**, using only held native OOF. The alpha=0 native control vector is the already audited native-only v3 OOF vector, so it is bit-exact. All five zero-weight models were also freshly refit and their raw scores were preserved; their maximum cross-process floating-point deviation from the older v3 run was 2.605e-10 and did not change any threshold decision or confusion count.

| evaluation | window F1 | window AP | window AUROC | answer F1 | answer AP | answer AUROC |
|---|---:|---:|---:|---:|---:|---:|
| selected native OOF | 0.610247 | 0.588272 | 0.878757 | 0.752941 | 0.779623 | 0.792832 |
| selected all-domain OOF / native thresholds | 0.586679 | 0.567216 | 0.902683 | 0.683653 | 0.723417 | 0.846456 |
| calibration strict / native OOF thresholds | 0.649420 | 0.714135 | 0.908322 | 0.806630 | 0.927145 | 0.887288 |
| calibration F1Opt diagnostic | 0.665121 | 0.714135 | 0.908322 | 0.875000 | 0.927145 | 0.887288 |
| native-only v3 strict | 0.644971 | 0.719675 | 0.906335 | 0.806630 | 0.932968 | 0.890678 |
| semantic-window-v2 strict | 0.653995 | 0.700032 | 0.907753 | 0.854545 | 0.923673 | 0.881525 |

| evaluation | window TP/FP/FN/TN | answer TP/FP/FN/TN |
|---|---:|---:|
| selected native OOF | 13,924 / 10,233 / 7,553 / 136,413 | 256 / 96 / 72 / 210 |
| selected all-domain OOF / native thresholds | 40,255 / 38,542 / 18,178 / 557,004 | 872 / 552 / 255 / 2,001 |
| calibration strict / native OOF thresholds | 3,753 / 1,821 / 2,231 / 34,436 | 73 / 8 / 27 / 51 |
| calibration F1Opt diagnostic | 4,304 / 2,654 / 1,680 / 33,603 | 91 / 17 / 9 / 42 |

Formal common-calibration F1Opt references: Lookback 0.600882/0.845455, LUMINA 0.331299/0.785047, and GHOST 0.320361/0.772201 (window/answer). The historical repeatedly calibration-selected incumbent is 0.690281/0.891089.

Generator identity has zero inference columns. It affects only training sample weights. Held source-connected groups are removed across all generators in every fold; all-domain OOF never selects alpha or thresholds. Calibration ran once after freeze, F1Opt is diagnostic only, formal baseline hashes stayed unchanged, and official test data remained unopened.

The independent verifier rebuilt every fit window, reconstructed all 25,864 expanded hypothesis identities, replayed all 20 serialized fold models and every OOF metric/selection, and replayed calibration scores from label-free features without reopening calibration labels. All checks passed; the sole calibration replay rounding difference was 5.55e-17, within one float64 ULP.
