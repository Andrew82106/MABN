# Forced-evidence quote probe GPU runner — independent review V2

**Status: PASS for the fixed eight-claim GPU smoke only. Full extraction remains blocked.**

Reviewed locks:

- Runner SHA256: `7989c1c9f5e8c835df8ccf4797a242b50138513e6a9c255374d386787c574139`
- CPU selfcheck SHA256: `f2581eaecd295f05003204e2f3e541c0dca190470d6550b30f4d2eb8402afe1d`
- Protocol SHA256: `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa`
- Plan SHA256: `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`

The V2 review found no remaining static blocker. The runner reads the frozen 3,776-row label-free sample manifest as its sole sample data input, uses `claim_prompt_text`, performs whole-prefix greedy stopping, keeps cached decoding only for generated IDs/statistics, and takes P3 hidden/attention from a full `use_cache=False` replay. P2 is a whole-string no-cache teacher-forced trajectory. Feature assembly is fixed at P1=21, P2=549, P3=549.

The earlier malformed-tag blocker is closed: nested quote tags are removed from the isolated relation context, while text after the first close and terminal EOS remain excluded. Independent CPU checks covered this branch and Unicode decode/re-encode offsets.

The Q/K hook uses Transformers' official `apply_rotary_pos_emb` and `repeat_kv` and matched the official CPU dense-attention oracle with maximum absolute error `0.0`. The runner delegates GPU exclusivity to the audited WDDM gate. Fixed smoke indices and prompt lengths reproduced exactly: `[2826, 994, 1533, 1695, 1299, 1143, 3352, 3752]` and `[255, 354, 385, 435, 508, 568, 634, 799]`; the shortest and longest are repeated. Atomic record commit/resume roundtrip and review gates passed.

No GPU or pretrained model was started during this review. No gold, calibration, or official-test file was opened; no scoring ran and no baseline was modified.

The canonical review JSON now permits `gpu-smoke`. `full_extract_allowed` is `false`; full extraction requires a separate independent review of the resulting GPU smoke artifact.
