# Independent audit repair

The first independent audit used a different word tokenizer for its BM25 recomputation: it joined apostrophe-linked words, while the frozen retriever splits them. That auditor correctly stopped and its failure record is retained. `audit_v2.py` changes only the independent recomputation to the exact published Unicode-word rule; no dataset artifact or manifest is changed.
