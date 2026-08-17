# P1v2-Q0 离线资格框架交付报告

- Historical Q0 run ID: `P1V2Q-READINESS-DRY-20260801T010101000000Z`
- Record status: `historical_pre_rework`
- Historical pre-rework record; not for downstream decisions.
- Artifact kind: `p1v2_model_output_qualification`
- Phase: `P1V2_Q0_READINESS_DRY`
- Data role: `p1v2_qualification`
- Execution mode: `readiness_dry`

## 结果摘要

- request_count: 142
- model_output: 128
- model_output_rejected: 12
- model_call_failed: 2
- fake_provider_calls: 142
- real_model_calls: 0
- local_loopback_http_calls: 0
- remote_network_calls: 0
- replay_model_calls: 0
- replay_network_calls: 0

## 离线复核

- Validation: pass
- Replay: pass
- Python executable: `D:\anaconda\envs\multi_agent_graph\python.exe`
- Python version: `3.11.15`
- New dependencies: none
- Protected trees before/after: unchanged

## P1v2-A 契约来源身份

- `paperAlpha/pre_exp1/p1v2_benign/schemas/response_contract.schema.json`: `0d8ff62da193ededb93016fe33dd3c48dda39854b7e766285736e2b68aa43fec`
- `paperAlpha/pre_exp1/p1v2_benign/prompts/public_response_contract.txt`: `90920fd5368a9723b04986ced1c5df7cfe508fdd3e113e2894d3810d3287e64c`
- `paperAlpha/pre_exp1/p1v2_benign/configs/readiness.json`: `cbdd66f51cf9e9a56ddf1ae4b6c301187a32d109567509381fc85c25501a0965`
- `paperAlpha/data/pre_exp1/p1v2_benign/readiness_dry/manifests/manifest_P1V2-READINESS-DRY-20260801T000002000000Z.json`: `f239ba3f5dbf85f250a3d5868922914c8ca1f902b45e8ae6386ef61314a4f726`

## 明确边界

P1v2-Q0 readiness status: pass

P1v2-Q1 live model calls: not authorized

P1/P2 status: locked

本报告只证明离线资格框架的工程准入状态；它不是模型能力结论，也不构成 P1 或 P2 结果。
