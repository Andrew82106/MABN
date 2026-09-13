# Failure history

- `prepare` completed all 3,839 answers, then the count invariant found one missing eligible window. The one-token answer `14641` must produce one short window under the frozen evaluator; the draft used the ordinary `n-k+1` formula and produced zero. Full partial artifacts and exact failed sources are retained. No GPU/model/test/baseline was touched.
