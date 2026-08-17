# P1 local-model qualification

This namespace is a prerequisite check for two already-installed, local
Ollama models. It is not a P1 scientific batch and it never changes:

```text
P1 scientific gate: NOT_STARTED
eligible_for_scientific_analysis: false
```

The fixed candidates are `qwen3:8b` and `ministral-3:8b`. Each candidate may
run exactly one serial three-episode qualification batch. The three fictional
vendor tasks, eight frozen role prompts, graph, permissions, decoding,
timeouts, and budgets are identical for both candidates. No prompt is tuned
after observing a run.

All generated qualification evidence is contained in
`paperAlpha/data/pre_exp1/p1_model_qualification/`; static fictional fixtures
are in `paperAlpha/data/shared/p1_model_qualification/`. Neither contains real
vendors, credentials, or research-analysis data.

## Commands

Run from `paperAlpha` with the required environment:

```powershell
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_model_qualification\tests -p no:cacheprovider
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_model_qualification\scripts\run_qualification_dry.py
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_model_qualification\scripts\check_ollama_inventory.py
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_model_qualification\scripts\run_qualification_local.py --model qwen3:8b --allow-local-qualification
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_model_qualification\scripts\run_qualification_local.py --model ministral-3:8b --allow-local-qualification
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_model_qualification\scripts\validate_qualification_run.py --run-id <RUN_ID>
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_model_qualification\scripts\replay_qualification_run.py --run-id <RUN_ID>
```

`check_ollama_inventory.py` uses only `ollama --version`, `ollama list`, and
`ollama show <model> --verbose`. The code has no pull, create, copy, download,
remote-provider, credential, `.env`, or retry path. Local inference is limited
to loopback `http://127.0.0.1:11434`, concurrency `1`, retry `0`, 256 output
tokens, 90 seconds per call, and 54 calls across both candidates.
