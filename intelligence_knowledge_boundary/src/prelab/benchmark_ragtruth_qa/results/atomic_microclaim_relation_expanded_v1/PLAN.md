# Expanded atomic-microclaim relation data v1

Build one supervised example per factual atomic microclaim. Each paired input contains the question, the claim-selected top-2 evidence from every released passage, and the contextualized claim. Gold comes only from the unchanged human answer spans projected through original BPE coordinates.

Fit uses 3,680 answers. Calibration reuses exactly the existing 2,267 scored microclaims. Group folds and weights keep source-connected groups indivisible. This stage performs CPU preparation and audit only.
