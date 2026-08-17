# P1v2-Q0 离线资格框架交付报告

- Historical Q0 run ID: `P1V2Q-READINESS-DRY-20260801T010102000000Z`
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

## 测试证据

- Command: `conda run --no-capture-output -n multi_agent_graph python -B -m pytest -p no:cacheprovider paperAlpha/pre_exp1/p1v2_qualification/tests -q`
- Implementation verification summary: `9 passed`.
- No new dependencies were installed.

## 冻结的未来 Q1 配置（Q0 未执行）

- model_output_stack_id: `QWEN3_STRICT_JSON_V1`
- model_tag: `qwen3:8b`
- provider: `local_ollama_loopback`
- future_endpoint: `http://127.0.0.1:11434/api/chat`
- format / temperature / seed / think: `json` / `0` / `20260801` / `False`
- concurrency / retries / timeout: `1` / `0` / `60`

## Q0 身份与下游边界

- is_new_p1_run: `False`
- eligible_for_p1_gate_analysis: `False`
- eligible_for_p2: `False`
- eligible_for_confirmatory_analysis: `False`
- eligible_for_causal_effect_analysis: `False`
- p1_go: `False`
- p2_allowed: `False`
- qualification_candidate_status: `'pending'`
- qualification_batch_decision: `'pending'`
- real_model_calls: `0`
- local_loopback_http_calls: `0`
- remote_network_calls: `0`
- replay_model_calls: `0`
- replay_network_calls: `0`

## P1v2-A 契约来源身份

- `paperAlpha/pre_exp1/p1v2_benign/schemas/response_contract.schema.json`: `0d8ff62da193ededb93016fe33dd3c48dda39854b7e766285736e2b68aa43fec`
- `paperAlpha/pre_exp1/p1v2_benign/prompts/public_response_contract.txt`: `90920fd5368a9723b04986ced1c5df7cfe508fdd3e113e2894d3810d3287e64c`
- `paperAlpha/pre_exp1/p1v2_benign/configs/readiness.json`: `cbdd66f51cf9e9a56ddf1ae4b6c301187a32d109567509381fc85c25501a0965`
- `paperAlpha/data/pre_exp1/p1v2_benign/readiness_dry/manifests/manifest_P1V2-READINESS-DRY-20260801T000002000000Z.json`: `f239ba3f5dbf85f250a3d5868922914c8ca1f902b45e8ae6386ef61314a4f726`

## 受保护历史树哈希

- `paperAlpha/data/pre_exp1/p1_benign`: before `46c025af42be65f17b04b4372a35ba38eb39fa6861935845bb2ebe9460658ee4`; after `46c025af42be65f17b04b4372a35ba38eb39fa6861935845bb2ebe9460658ee4`
- `paperAlpha/data/pre_exp1/p1_model_qualification`: before `1bc7784f33358547218926750c216394d56bed7ad9a7d4ff27c344ba038fcf34`; after `1bc7784f33358547218926750c216394d56bed7ad9a7d4ff27c344ba038fcf34`
- `paperAlpha/data/pre_exp1/p1_scientific_benign`: before `1f27f805fb0258af47f7de9c1963194123d84c2c5478326075d2819ba1ab9b32`; after `1f27f805fb0258af47f7de9c1963194123d84c2c5478326075d2819ba1ab9b32`
- `paperAlpha/data/pre_exp1/p1v2_benign`: before `ccb7e8ea38b2231411dda8bde0237447bd7644c17c3269276d835051b2334f57`; after `ccb7e8ea38b2231411dda8bde0237447bd7644c17c3269276d835051b2334f57`
- `paperAlpha/pre_exp1/p1_benign`: before `a99a34ba6f8cac94201e505ea00699a0b70dd5147a9bf1fded9ca330f97de67c`; after `a99a34ba6f8cac94201e505ea00699a0b70dd5147a9bf1fded9ca330f97de67c`
- `paperAlpha/pre_exp1/p1_model_qualification`: before `bef9e43ad7e9e825cca262fb457bcd2f51e6db83d1a17126c40781b424c606d0`; after `bef9e43ad7e9e825cca262fb457bcd2f51e6db83d1a17126c40781b424c606d0`
- `paperAlpha/pre_exp1/p1v2_benign`: before `832a0bcbdcfb4cf9a58dc676c7578bec4e2556b89bd191a077261e98ef59c9e7`; after `832a0bcbdcfb4cf9a58dc676c7578bec4e2556b89bd191a077261e98ef59c9e7`

## 明确边界

P1v2-Q0 readiness status: pass

P1v2-Q1 live model calls: not authorized

P1/P2 status: locked

本报告只证明离线资格框架的工程准入状态；它不是模型能力结论，也不构成 P1 或 P2 结果。
