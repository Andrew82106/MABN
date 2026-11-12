"""Pinned, monitor-only BlindGuard PI/SCL inference; not a privacy-risk predictor.

The author path calls GATSCL.encode + inference_new, NOT GATSCL.forward.
It embeds the initial response of each agent, constructs first-round recipient
features, and ranks negative all-node cosine-similarity sums. Later rounds keep
the initial ranking. We preserve those choices, including zero features for
nodes without incoming edges. The upstream Apache-2.0 source stays external.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from importlib.metadata import version


UPSTREAM_REPOSITORY = "https://github.com/MR9812/BlindGuard"
UPSTREAM_COMMIT = "ecbd8e9b02c9ebc63b8bf2d7074a6d32a4a2c8b4"
PROFILE = "blindguard-author-pi-scl-type0-csqa-first-round-monitor-only"
EMBEDDING_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
CHECKPOINT = (
    "PI/checkpoint_un2/csqa/20250701_181329-defend_type_SCL-hiddim_1024-"
    "latent_512-heads_8-layers_2-epochs_20-lr_0.001.pth"
)
CHECKPOINT_SHA256 = "51bf76aced8bc7708d1d535f63062967971ab4268705a30e01ce0b5da225f65f"
SOURCE_BLOBS = {
    "LICENSE": "75a647b14670d3a1f409ba57afc0a24b45a5aa6d",
    "PI/TAM.py": "ee8bfc7cd0777b3db6f6f082f39ab2b6064b8d7d",
    "PI/utils.py": "adfd8030bed6be125a48b2536e5cd915e84a84d5",
    "PI/main_defense_for_different_topology1.py": "89d20b59c9a2d747ecaaaf9ac23e473e9831b219",
}
MODEL_CONFIG = {
    "in_channels": 384, "hidden_channels": 1024, "out_channels": 512,
    "heads": 8, "dropout": 0.0, "type": 0,
}


def default_repository() -> Path:
    return Path(__file__).resolve().parents[3] / 'code/external/BlindGuard'


def default_embedding_snapshot() -> Path:
    return (Path.home() / ".cache" / "huggingface" / "hub"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots" / EMBEDDING_REVISION)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkout(repository: str | Path) -> dict:
    """Check only selected files in a deliberately partial author checkout."""
    repository = Path(repository).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("BlindGuard source commit does not match the audited revision")
    files = {}
    for relative, expected in SOURCE_BLOBS.items():
        payload = (repository / relative).read_bytes()
        # Git's Windows checkout may convert LF to CRLF. Accept that transport
        # conversion only, while still rejecting any source-content change.
        canonical = payload.replace(b"\r\n", b"\n")
        blob = hashlib.sha1(f"blob {len(canonical)}\0".encode() + canonical).hexdigest()
        if blob != expected:
            raise ValueError(f"Author source was changed: {relative}")
        files[relative] = {"git_blob": blob, "checkout_sha256": hashlib.sha256(payload).hexdigest(),
                           "canonical_LF_sha256": hashlib.sha256(canonical).hexdigest()}
    checkpoint = repository / CHECKPOINT
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("BlindGuard checkpoint does not match the selected author weights")
    return {
        "repository": UPSTREAM_REPOSITORY, "commit": commit, "license": "Apache-2.0",
        "selected_source_files": files,
        "checkpoint": {"path": CHECKPOINT, "sha256": checkpoint_sha,
                       "bytes": checkpoint.stat().st_size},
    }


def inspect_embedding_snapshot(snapshot: str | Path) -> dict:
    """Require the cached original embedding; never resolve a remote model ID."""
    snapshot = Path(snapshot).resolve()
    required = ["config.json", "modules.json", "tokenizer.json", "tokenizer_config.json",
                "vocab.txt", "special_tokens_map.json", "sentence_bert_config.json",
                "config_sentence_transformers.json", "1_Pooling/config.json", "model.safetensors"]
    for name in required:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"Original embedding cache missing: {name}")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    modules = json.loads((snapshot / "modules.json").read_text(encoding="utf-8"))
    expected_types = ["sentence_transformers.models.Transformer",
                      "sentence_transformers.models.Pooling",
                      "sentence_transformers.models.Normalize"]
    if config.get("hidden_size") != 384 or [m["type"] for m in modules] != expected_types:
        raise ValueError("Embedding cache architecture differs from the original MiniLM profile")
    if _sha256(snapshot / "model.safetensors") != EMBEDDING_SHA256:
        raise ValueError("Embedding weights differ from the pinned cached MiniLM revision")
    return {
        "model": EMBEDDING_ID, "revision": EMBEDDING_REVISION,
        "weights_sha256": EMBEDDING_SHA256, "dimension": 384,
        "local_files_only": True, "trust_remote_code": False,
        "extra_encode_normalization": False,
        "built_in_normalize_module": True,
        "config_sha256": {name: _sha256(snapshot / name) for name in required if name != "model.safetensors"},
    }


def _load_author_code(repository: Path):
    """Import the reviewed model module, not author training/API runners."""
    import numpy as np
    import torch
    from torch_scatter import scatter_mean

    spec = importlib.util.spec_from_file_location(
        "_paperalpha_blindguard_pinned_pi_tam", repository / "PI" / "TAM.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Reuse the exact reviewed function AST. Do not import the whole runner:
    # that would import API agents and unrelated training/defense modules.
    helper_path = repository / "PI" / "main_defense_for_different_topology1.py"
    parsed = ast.parse(helper_path.read_text(encoding="utf-8"), filename=str(helper_path))
    functions = [node for node in parsed.body
                 if isinstance(node, ast.FunctionDef) and node.name == "embeddings2graph"]
    if len(functions) != 1:
        raise ValueError("Audited embeddings2graph function was not found exactly once")
    namespace = {"np": np, "torch": torch, "scatter_mean": scatter_mean}
    helper_module = ast.Module(body=functions, type_ignores=[])
    exec(compile(helper_module, str(helper_path), "exec"), namespace)
    return module.GATSCL, namespace["embeddings2graph"]


def validate_graph(adjacency, node_count: int):
    import numpy as np

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 2:
        raise ValueError("At least two visible agents are required by this adapter")
    matrix = np.asarray(adjacency)
    if matrix.shape != (node_count, node_count):
        raise ValueError("Adjacency must match the visible agent count")
    if not np.issubdtype(matrix.dtype, np.number) and matrix.dtype != np.bool_:
        raise ValueError("Adjacency must be binary numeric values")
    if not np.isfinite(matrix).all() or not np.isin(matrix, [0, 1]).all():
        raise ValueError("Adjacency must contain finite binary values")
    if not matrix.any():
        raise ValueError("The author projection requires at least one communication edge")
    return matrix.astype(np.int64, copy=False)


class BlindGuardMonitor:
    """Frozen author model plus offline original sentence encoder.

    Continuous scores are anomaly rankings, NOT probabilities. This adapter
    accepts no labels, attacker identities, policy outcomes or future records.
    It does not modify any protected agent or perform model/API generation.
    """

    def __init__(self, repository: str | Path | None = None, *, device="cpu"):
        import torch

        self.repository = Path(repository or default_repository()).resolve()
        self.provenance = inspect_checkout(self.repository)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA devices are supported")
        model_class, self._graph_projection = _load_author_code(self.repository)
        self.model = model_class(**MODEL_CONFIG)
        state = torch.load(self.repository / CHECKPOINT, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected the author's plain tensor state dictionary")
        if not all(torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError("Non-finite author checkpoint tensor")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.provenance.update({
            "profile": PROFILE, "model_config": MODEL_CONFIG.copy(),
            "checkpoint_loading": "torch.load(weights_only=True), strict=True",
            "checkpoint_tensor_shapes": {k: list(v.shape) for k, v in state.items()},
            "active_path": "GATSCL.encode(type=0) -> GATSCL.inference_new -> negative score -> torch.topk",
            "inactive_author_members": "GAT forward and training projection fc1 are loaded for strict state compatibility but are not called by the author SCL inference path.",
            "device": str(self.device),
            "packages": {name: version(name) for name in
                         ["torch", "torch-geometric", "torch-scatter", "torch-sparse",
                          "numpy", "sentence-transformers", "transformers"]},
        })
        self._sentence_encoder = None

    def load_embedding(self, snapshot: str | Path | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        snapshot = Path(snapshot or default_embedding_snapshot()).resolve()
        self.provenance["embedding"] = inspect_embedding_snapshot(snapshot)
        self._sentence_encoder = SentenceTransformer(
            str(snapshot), device=str(self.device), local_files_only=True,
            trust_remote_code=False,
        )
        self._sentence_encoder.eval()
        dimension_method = getattr(self._sentence_encoder, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = self._sentence_encoder.get_sentence_embedding_dimension
        if dimension_method() != 384:
            raise ValueError("Original text encoder must produce 384-dimensional embeddings")

    def encode_messages(self, messages: list[str]):
        if not isinstance(messages, (list, tuple)) or len(messages) < 2:
            raise ValueError("Provide one visible initial response per agent")
        if not all(isinstance(message, str) for message in messages):
            raise TypeError("Messages must be text only, never event dictionaries or labels")
        if self._sentence_encoder is None:
            self.load_embedding()
        # Same model.encode semantics as PI/utils.py; caching and batching do
        # not replace the author's model or add an external normalization.
        return self._sentence_encoder.encode(
            list(messages), convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=False,
        )

    def project_embeddings(self, initial_embeddings, adjacency):
        import numpy as np

        embeddings = np.asarray(initial_embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != 384:
            raise ValueError("Expected [agent_count, 384] original sentence embeddings")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding input must be finite")
        matrix = validate_graph(adjacency, len(embeddings))
        # Preserve the complete author function, including its unused padded
        # edge attributes. SCL uses x and edge_index only.
        x, edge_index, edge_attr = self._graph_projection([embeddings], matrix)
        return x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device)

    def score_embeddings(self, initial_embeddings, adjacency, *, topk: int = 3) -> dict:
        import torch

        started = time.perf_counter()
        x, edge_index, _ = self.project_embeddings(initial_embeddings, adjacency)
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= len(x):
            raise ValueError("topk must be explicitly within [1, agent_count]; no silent clipping")
        with torch.inference_mode():
            z = self.model.encode(x, edge_index)
            adj = torch.eye(len(x), device=self.device)
            adj[edge_index[0], edge_index[1]] = 1.0
            similarities = self.model.inference_new(z, adj)
            scores = -similarities
            if not torch.isfinite(scores).all():
                raise ValueError("Author inference produced non-finite scores")
            indices = torch.topk(scores, topk).indices
            score_values = scores.detach().cpu().tolist()
            topk_values = indices.detach().cpu().tolist()
        return {
            "profile": PROFILE, "score_kind": "negative_all_node_cosine_similarity_sum",
            "is_probability": False, "node_anomaly_scores": score_values,
            "topk_node_indices": topk_values, "topk": topk,
            "agent_count": len(x), "directed_edge_count": edge_index.shape[1],
            "used_round_index": 0, "monitor_only": True,
            "detector_seconds": time.perf_counter() - started,
        }

    def score_initial_round(self, messages, adjacency, *, topk: int = 3) -> dict:
        started = time.perf_counter()
        embeddings = self.encode_messages(messages)
        encoding_seconds = time.perf_counter() - started
        result = self.score_embeddings(embeddings, adjacency, topk=topk)
        result["encoding_seconds"] = encoding_seconds
        return result

    def score_visible_prefix(self, response_rounds, adjacency, *, topk: int = 3) -> dict:
        """Author profile: later responses never change the initial ranking.

        This is a stateless replay convenience, not incremental retraining.
        The author defense runner caches the first top-k; re-evaluating the
        frozen deterministic first-round model preserves that behavior.
        """
        if not isinstance(response_rounds, (list, tuple)) or not response_rounds:
            raise ValueError("At least one complete visible response round is required")
        first = response_rounds[0]
        if not isinstance(first, (list, tuple)) or not all(isinstance(x, str) for x in first):
            raise TypeError("First-round responses must contain only agent text")
        result = self.score_initial_round(first, adjacency, topk=topk)
        result.update({"visible_round_count": len(response_rounds),
                       "later_rounds_used": False,
                       "temporal_scope": "author_first_round_ranking_retained"})
        return result
