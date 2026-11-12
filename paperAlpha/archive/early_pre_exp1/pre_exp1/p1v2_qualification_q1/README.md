# P1v2 Q1 local qualification

This directory is an independent, one-shot engineering qualification harness for
the fixed local Ollama tag `qwen3:8b`. It is not a P1 or P2 experiment and does
not establish a scientific, model-capability, or causal conclusion.

The only public live entry point is:

```powershell
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification_q1/scripts/run_q1_live.py --allow-live-q1
```

It has no configurable endpoint, model, prompt, fixture, root, retry, or
concurrency arguments. It may contact only `127.0.0.1:11434`, using `GET
/api/tags` for start/end identity evidence and `POST /api/chat` for the fixed
task cards. The implementation uses only the Python standard library.

Every run is append-only under
`paperAlpha/data/pre_exp1/p1v2_qualification_q1/runs/<run-id>/`. Validators and
replay are offline and have no transport code. A result of `qualified`,
`not_qualified`, or `rework` always leaves every P1/P2 eligibility flag false
and requires independent acceptance before any later work.
