# Expanded-fit semantic source attribution v1

This runner maps only the 3,046 additional fit answers. It imports the exact frozen native v1 Q/K/V attribution implementation and writes to a separate cache.

- Inputs: frozen replay plans plus label-free expanded microclaims.
- Excluded: 16 punctuation-only microclaims with no lexical BPE.
- Geometry: 485,856 new eligible 4-BPE windows; with the retained 168,123 native-fit windows this is 653,979.
- Data isolation: no answer labels, token/window labels, calibration, or official test data are opened.
- GPU policy: `prepare`, `cpu-check`, and `status` are CPU-only; `extract` is explicit and resumable.
- Method: unchanged post-token attention-times-value-norm source attribution from native v1.
