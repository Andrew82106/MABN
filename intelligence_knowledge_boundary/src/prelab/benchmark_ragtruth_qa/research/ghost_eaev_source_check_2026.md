# Additional source checks, 2026-09-12

## GHOST — ACL 2026 main conference

The [paper](https://aclanthology.org/2026.acl-long.993.pdf), Sections 3 and A, averages token-wise geometry into answer features: adjacent-layer cosine change, similarity to the final state, normalized top-10 entropy, and unweighted input-embedding dispersion. Its classifier is a tuned random forest. The reported RAGTruth setting has 2,500 examples, an 80/20 split, and automated answer-level judging. Its F1 is therefore not our human-labelled four-BPE localization F1. No author implementation link was found in the inspected PDF.

Local decision: retain the four token features instead of immediately averaging the answer. `src/ghost_geometry.py` implements streaming layer accumulation and a low-memory equivalent of pairwise dispersion. A random tiny CPU Llama matches dense formulas within 2.39e-7 and causal-prefix calculations within 5.97e-8; repeated output is exact. All four features use the position before reading the target answer token, an explicit local adaptation. No real sample, pretrained weights, GPU or classifier training has been used. Extraction and downstream comparison remain pending; this is not a completed GHOST reproduction or an accuracy result.

## EAEV — Findings of ACL 2026

The [paper](https://aclanthology.org/2026.findings-acl.1477.pdf), Sections 3.3–3.6, combines entity identity, semantic similarity, numerical consistency and perturbation stability. Main results use a fine-tuned verifier trained from generated entity annotations, rather than only these raw scores. Consequently, copying its arithmetic would not reproduce its reported system.

Local inference: removing the strongest evidence window can also remove the only valid support for a correct statement. Therefore its minimum-under-removal score is not automatically suitable for our partial-evidence setting. Entity correspondence remains useful motivation, but we should verify support relationships rather than assume robustness to evidence deletion establishes correctness. No EAEV labels, model or classifier were created.

Both official PDFs and URL/SHA records are saved under `doc/ref_paper/intelligence_knowledge_boundary/hallucination_detection/`. Existing sources and experimental inputs remain unchanged.
