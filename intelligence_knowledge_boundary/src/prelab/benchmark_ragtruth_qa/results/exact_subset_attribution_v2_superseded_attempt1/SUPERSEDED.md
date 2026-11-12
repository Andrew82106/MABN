# Superseded CPU-only attempt

This directory was frozen and audited before GPU use, then superseded because
the first v2 runtime signature rehashed the 179 MB prepared-input file inside
every per-answer cache signature.  No GPU/model, labels, test data, fitting, or
baseline was used.  The final `exact_subset_attribution_v2` keeps the identical
WDDM admission policy and binds the already verified prepared-input digest as a
constant after `check_ready` performs the full hash check once.
