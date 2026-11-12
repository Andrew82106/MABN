"""Config-only meta-device parameter count. No weights, data, training or GPU."""
from pathlib import Path
import hashlib
import json
import requests
import torch
import transformers
from transformers import ModernBertConfig, ModernBertForTokenClassification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert not (HERE / 'RESOURCE_COUNTS.json').exists()
assert not torch.cuda.is_initialized()
repo = 'answerdotai/ModernBERT-large'
api_url = f'https://huggingface.co/api/models/{repo}'
r = requests.get(api_url, timeout=30); r.raise_for_status(); api = r.json()
revision = api['sha']
url = f'https://huggingface.co/{repo}/resolve/{revision}/config.json'
r = requests.get(url, timeout=30); r.raise_for_status()
large_bytes = r.content
large = json.loads(large_bytes)
(HERE / 'large_config.json').write_bytes(large_bytes)
base_path = ROOT.parent / 'models/ModernBERT-base/config.json'
base = json.loads(base_path.read_bytes())
counts = {}
for name, raw in [('base', base), ('large', large)]:
    cfg = ModernBertConfig.from_dict(raw)
    cfg.num_labels = 2
    cfg.reference_compile = False
    cfg._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = ModernBertForTokenClassification(cfg)
    assert all(p.device.type == 'meta' for p in model.parameters())
    n = sum(p.numel() for p in model.parameters())
    components = {}
    for pname, p in model.named_parameters():
        group = 'embeddings' if pname.startswith('model.embeddings') else ('layers' if pname.startswith('model.layers') else 'other_norm_and_classification_head')
        components[group] = components.get(group, 0) + p.numel()
    counts[name] = {'token_classifier_parameters': n, 'components': components,
                    'layers': cfg.num_hidden_layers, 'hidden': cfg.hidden_size,
                    'heads': cfg.num_attention_heads, 'intermediate': cfg.intermediate_size,
                    'one_fp32_copy_bytes': 4*n, 'params_grads_adam_m_v_bytes': 16*n,
                    'persistent_GiB': 16*n/2**30,
                    'persistent_plus_one_fp32_sized_temporary_GiB': 20*n/2**30}
    del model
assert counts['base']['token_classifier_parameters'] == 149606402
assert not torch.cuda.is_initialized()
report = {'scope': 'Config/meta-device only; no model weights downloaded or allocated',
          'official_repository': repo, 'revision': revision, 'config_url': url,
          'official_api_safetensors_metadata': api.get('safetensors'),
          'config_sha256': hashlib.sha256(large_bytes).hexdigest(),
          'base_config_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
          'transformers': transformers.__version__, 'torch': torch.__version__,
          'models': counts, 'GPU_initialized': torch.cuda.is_initialized(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE / 'RESOURCE_COUNTS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False), flush=True)
