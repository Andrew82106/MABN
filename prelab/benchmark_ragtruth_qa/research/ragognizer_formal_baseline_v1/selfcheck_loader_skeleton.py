"""Instantiate the exact headed class on the meta device without model weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from accelerate import init_empty_weights
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = HERE.parents[3]
HEAD_REPO = PROJECT / "third_party" / "transformer-heads"
BASE = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf-base-f5db-runtime"
FT = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf-offline-runtime" / "ft_llm"
OUTPUT = HERE / "LOADER_SKELETON_AUDIT.json"


def main() -> None:
    sys.path.insert(0, str(HEAD_REPO))
    from transformer_heads.config import HeadConfig, create_headed_model_config
    from transformer_heads.model.model import get_multi_head_transformer
    from transformer_heads.util.helpers import get_model_params

    params = get_model_params(str(BASE))
    base_class = params["model_class"]
    base_config = base_class.config_class.from_pretrained(str(BASE), local_files_only=True)
    configs = [
        HeadConfig(**value)
        for value in json.loads((FT / "head_configs.json").read_text(encoding="utf-8")).values()
    ]
    headed_config = create_headed_model_config(base_class.config_class).from_base_class(
        base_config, configs
    )
    with init_empty_weights():
        model = get_multi_head_transformer(base_class)(headed_config)
    head = model.heads["hallu_head_neg_16"]
    actual_layers = [
        [int(layer.in_features), int(layer.out_features), layer.bias is not None]
        for layer in head.lins
    ]
    expected_layers = [[4096, 1024, True], [1024, 1024, True], [1024, 1, False]]
    if actual_layers != expected_layers:
        raise RuntimeError(f"headed class architecture drift: {actual_layers}")
    saved_shapes = {}
    with safe_open(FT / "hallu_head_neg_16.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            saved_shapes[key] = list(map(int, handle.get_slice(key).get_shape()))
    expected_shapes = {
        "lins.0.bias": [1024], "lins.0.weight": [1024, 4096],
        "lins.1.bias": [1024], "lins.1.weight": [1024, 1024],
        "lins.2.weight": [1, 1024],
    }
    if saved_shapes != expected_shapes:
        raise RuntimeError("saved integrated-head shapes drifted")
    result = {
        "version": "ragognizer-loader-skeleton-cpu-audit-v1",
        "status": "pass",
        "base_class": base_class.__name__,
        "model_type": model.config.model_type,
        "hidden_layers": int(model.config.num_hidden_layers),
        "integrated_heads": list(model.heads),
        "detection_head_layers": actual_layers,
        "saved_detection_head_shapes": saved_shapes,
        "lm_head": [int(model.lm_head.in_features), int(model.lm_head.out_features)],
        "model_weights_loaded": False,
        "labels_read": False,
        "gpu_workload_started": False,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
