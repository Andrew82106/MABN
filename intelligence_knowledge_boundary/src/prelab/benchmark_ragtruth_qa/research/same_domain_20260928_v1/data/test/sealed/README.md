# Sealed internal test targets

Only the final scorer may read labels.jsonl after model, threshold, and feature selection are frozen. It contains copied automatic span/token/window labels. No test label statistics were computed by this preparation. The underlying official train data has prior research exposure; do not call this an unseen external test.
