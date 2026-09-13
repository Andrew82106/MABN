# Failure history

- `prepare` failed before completing the first answer. Root cause: antecedent subject `I` was tested as a raw substring, so the `i` inside `taking` suppressed required subject insertion. The partial `.pending` files, immutable failure JSON, and exact failed source snapshots are retained. No GPU/model/test/baseline was touched.
