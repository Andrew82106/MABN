# MABN

Multi-Agent Graph Analysis: studying unsafe behavior propagation and containment in LLM multi-agent systems.

## Structure

- `doc/` — project design documents, work guides, and task briefs
- `shared_lab/` — shared experiment platform (kernel, scenario configs, experiment hooks)
- `paperAlpha/` — current MAS safety paper workspace: BayesTrace extension (research direction under discussion), with legacy pre_exp1 code and data preserved
- `intelligence_knowledge_boundary/` — independent Journal of Intelligence project: event knowledge boundaries and verification support
- `prelab/` — weakly supervised factual-error monitoring pilot: data, local model, probes, and heldout localization tests

## Quick Start

See [doc/README.md](doc/README.md) for navigation and [paperAlpha/README.md](paperAlpha/README.md) for the implementation details.

## Paper handoff

The current factual-hallucination monitoring experiment, its fixed evaluation
protocol, baseline table, limitations, and suggested paper framing are in
[the collaborator handoff](prelab/benchmark_ragtruth_qa/research/collaborator_paper_brief_v1/EXPERIMENT_OVERVIEW_CN.md).
Large model weights, virtual environments, raw/generated arrays, and local
logs are intentionally excluded; download locations and hashes are recorded in
the experiment manifests.
