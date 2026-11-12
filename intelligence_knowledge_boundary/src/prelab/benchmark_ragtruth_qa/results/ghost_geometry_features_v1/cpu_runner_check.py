"""Small CPU integration test of the new runner's independent dense oracle."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_ghost_feature_extraction_v1 as run
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    config = LlamaConfig(vocab_size=47, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 47, (1, 35))
    positions = torch.arange(5, 35)
    with torch.no_grad():
        actual = run.ghost.extract(model, ids, positions, run.FIRST, run.LAST, run.LOGIT_BATCH)
        dense = run.dense_oracle(model, ids, positions)
        maximum = float((actual-dense).abs().max())
        assert torch.allclose(actual, dense, atol=8e-7, rtol=0)
        assert torch.equal(actual, run.ghost.extract(model, ids, positions, run.FIRST, run.LAST, run.LOGIT_BATCH))
    with tempfile.TemporaryDirectory(prefix='missing_smoke_test_', dir=OUT) as directory:
        with patch.object(run, 'OUT', Path(directory)), patch.object(run.loader, 'load_nf4') as gpu_loader:
            try:
                run.extract([], {})
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing smoke must block extract')
            gpu_loader.assert_not_called()
    assert not torch.cuda.is_initialized()
    result = {'status': 'passed', 'CPU_model': 'random32-layer Llama, hidden32, SDPA',
              'full_input_tokens': 35, 'raw_answer_tokens': 30, 'lm_head_chunks': [16, 14],
              'runner_dense_oracle_max_abs_error': maximum, 'repeat_exact': True,
              'missing_smoke_blocks_before_GPU_loader': True, 'GPU_used': False,
              'pretrained_weights_loaded': False, 'test_opened': False, 'trained': False,
              'code_sha256': {str(Path(__file__).resolve()): run.q.sha(Path(__file__)),
                              str(Path(run.__file__).resolve()): run.q.sha(Path(run.__file__))}}
    run.frozen_json(OUT / 'CPU_RUNNER_SELFCHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
