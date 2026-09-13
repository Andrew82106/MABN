# Failure history

- `prepare` reached the one-token answer `14641`. The candidate-window rule had been corrected, but the window-to-claim edge loop still indexed four positions rather than clipping at the short answer end. Partial artifacts, failure JSON and exact sources are retained. No GPU/model/test/baseline was touched.
