# Expanded semantic window v2

`whitebox_geometry`, C=0.1 was copied from the native-634 fit-only winner before this run. No expanded-fit feature or C search was performed.

All 3,680 fit answers (615 source-connected groups) supplied 653,979 directly supervised four-BPE windows, including 58,433 positive windows. Lookback and large inputs were reconstructed only from each source group's frozen upstream held fold; generation NLL was label-blind. The first 168,123 native feature rows replayed exactly.

| partition / threshold | window F1 | answer F1 |
|---|---:|---:|
| expanded fit 5-fold OOF / fit-F1Opt | 0.599064 | 0.688277 |
| calibration / frozen fit thresholds | 0.640311 | 0.788571 |

Calibration was scored once after the full model and thresholds were frozen. No calibration threshold was fitted, the official test remained unopened, GPU was unused, and formal baseline files were unchanged.
