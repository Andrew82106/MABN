# Combined semantic-attribution fit interface

This is the frozen data interface for a later `semantic_window_v2` experiment. It combines the original 634 fit feature shards and 3,046 expanded fit shards without changing either extractor.

- Gold: 3,680 fit answers, 665,708 raw BPEs, 34,919 scored microclaims, and 653,979 eligible 4-BPE windows.
- Positive gold: 47,398 BPEs, 3,474 microclaims, and 58,433 windows.
- Exclusions: 22 punctuation-only microclaims and 692 no-lexical-BPE windows.
- Grouping: the locked 615 source-connected groups remain indivisible.
- Isolation: no calibration/test path is opened.
- This module only loads, verifies, and projects data; it has no model, loss, feature choice, threshold, or GPU command.
