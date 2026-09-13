# PsiloQA auxiliary-data candidate

Source: [EMNLP 2025 Findings paper](https://aclanthology.org/2025.findings-emnlp.626/), [author dataset](https://huggingface.co/datasets/s-nlp/PsiloQA), [author code](https://github.com/s-nlp/PsiloQA).

The dataset supplies Wikipedia passages, questions, model answers and automatically annotated error spans. Answers were generated without the passages. GPT-4o supplied annotations; refusals were filtered upstream. This differs from native RAG generation and human gold. The paper reports character IoU/AP, not our four-BPE F1; its transfer experiments do not establish a gain on our fixed QA calibration set.

Local decision: inventory only the official English training subset, preserve alignment failures, group repeated Wikipedia material, and check source overlap before considering auxiliary training. Keep current QA labels, windows, calibration and sealed test unchanged. Do not download released detector weights with uncertain training provenance.

The original paper and its download/hash record are stored in `doc/ref_paper/intelligence_knowledge_boundary/hallucination_detection/PsiloQA_EMNLP2025_Findings.pdf` and the adjacent `.source.json`. This note records a data candidate, not a completed training experiment.
