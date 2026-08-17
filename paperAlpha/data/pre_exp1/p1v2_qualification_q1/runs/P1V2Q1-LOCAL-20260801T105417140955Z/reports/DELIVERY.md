# P1v2 Q1 local qualification delivery

- Run ID: `P1V2Q1-LOCAL-20260801T105417140955Z`
- Protocol: `P1V2_Q1_LIVE_LOCAL`; local model: `qwen3:8b`.
- Q0 current run: `P1V2Q-READINESS-DRY-20260801T010106000000Z`.
- Single decision: `not_qualified`.
- Screen/Confirmation executed: 32/0; unstarted: 0/96.
- Calls: inference=32, metadata=2, loopback=34, remote=0, replay-model=0, replay-network=0.
- Start/end fingerprint equal: `True`.
- Interpreter: `D:\anaconda\envs\multi_agent_graph\python.exe` / Python `3.11.15`; dependency changes: none.
- Live command: `conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification_q1/scripts/run_q1_live.py --allow-live-q1`.
- Q1 validation/replay are offline ledger checks. Q0 public pre/post validation and replay are recorded by the external acceptance command log.
- Manifest source-tree hash: `d14521afe8d85e762f50bd46445ccf2dc9817bb9d293b4f699e6cef8ad82cc90`.
- No `.env`, remote API, retry, model switch, P1 run, or P2 run is authorized by this artifact.
