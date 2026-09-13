# Exact evidence-union NLI v2: native CPU audit

The manifest contains the exact sentence-instance union of attention top 3 and
BM25 top 2 independently in each passage.  All 61,779
fit and 15,556 calibration candidate instances hit
their occurrence-bound frozen NLI caches; neither cohort has a missing request.

| Cohort | Claims | Candidate instances | Candidate count histogram |
|---|---:|---:|---|
| fit | 9,055 | 61,779 | {'3': 100, '4': 109, '5': 430, '6': 2395, '7': 3902, '8': 1862, '9': 257} |
| calibration | 2,267 | 15,556 | {'4': 27, '5': 68, '6': 656, '7': 1030, '8': 412, '9': 74} |

| Coverage of union best | fit attention | fit BM25 | cal attention | cal BM25 |
|---|---:|---:|---:|---:|
| entailment | 74.61% | 92.69% | 71.33% | 92.59% |
| contradiction | 37.23% | 84.64% | 39.57% | 83.46% |

The reusable label-free claim interface has 44 float32 columns:
union, attention and BM25 subset max/mean E/N/C, class margins, lack-E,
OR-risk, union gains and best-score coverage flags.  Exact candidate E/N/C is
retained separately in each cohort's `candidate_arrays_*.npz`.

The portable cache catalog contains 71,273 bit-identical reusable requests and 808 conflicting requests; the latter are deliberately marked for recomputation when attaching new answers.
Calibration used the rule frozen by the fit score command and did not select or
change a rule.  No labels, pretrained model, GPU, official test data, detector
training, or baseline write was used by this audit.
