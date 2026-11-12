# Expanded semantic window v2 — fit-only domain diagnostic

All rows below use the already frozen five-fold OOF scores and global fit thresholds. No model or operational threshold was changed; subgroup F1-opt values in JSON are descriptive only.

| generator | answers | ans +% | ans AUROC | ans AP | ans F1 | windows | win +% | win AUROC | win AP | win F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-3.5-turbo-0613 | 578 | 9.17 | 0.751 | 0.318 | 0.375 | 63416 | 2.37 | 0.835 | 0.336 | 0.462 |
| gpt-4-0613 | 614 | 4.72 | 0.697 | 0.160 | 0.118 | 84171 | 0.59 | 0.807 | 0.071 | 0.089 |
| llama-2-13b-chat | 631 | 41.84 | 0.813 | 0.762 | 0.684 | 130655 | 10.33 | 0.902 | 0.589 | 0.584 |
| llama-2-70b-chat | 629 | 32.91 | 0.864 | 0.769 | 0.708 | 120790 | 9.48 | 0.933 | 0.636 | 0.615 |
| llama-2-7b-chat | 634 | 51.74 | 0.789 | 0.793 | 0.734 | 168123 | 12.77 | 0.876 | 0.577 | 0.612 |
| mistral-7B-instruct | 594 | 41.41 | 0.856 | 0.809 | 0.737 | 86824 | 11.53 | 0.905 | 0.605 | 0.613 |

## Native versus added

- Native 634: window positive 12.77%, answer positive 51.74%; F1 0.612/0.734.
- Added 3,046: window positive 7.61%, answer positive 26.23%; F1 0.592/0.670.
- Added-subset F1-opt changes F1 by only +0.002/+0.003; threshold choice is not the main deficit.
- The current model on native rows differs from the old native-only OOF by +0.001/-0.008. The native 26-D feature replay has max error 0.

Strict saved calibration at frozen fit thresholds remains 0.640 window F1 and 0.789 answer F1.

The evidence rules out a feature-construction mismatch on the native rows and points to generator/domain heterogeneity and changed label priors. This is a diagnostic association, not a causal proof.
