"""CPU-only Atomic Evidence Router v1 development candidate.

The runner joins frozen, label-free per-token HARP/NLL/GHOST/LUMINA caches
to the deterministic atomic microclaims.  It deliberately waits for the
separately frozen atomic_microclaim_nli_v1 evidence matrix before any fit.
Only fit labels are used for the five-fold group OOF gate.  Calibration is
opened once, and only if the fixed K=4 router beats both fixed controls on
fit OOF.  The official test and formal baseline implementations are absent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/atomic_evidence_router_v1"
UPSTREAM = ROOT / "results/atomic_microclaim_nli_v1"
ATOMIC_INPUT = UPSTREAM / "inputs.jsonl"
ATOMIC_PREP = UPSTREAM / "preparation_complete.json"

NLL_DIR = ROOT / "data/features"
NLL_MANIFEST = ROOT / "data/feature_manifest.json"
HARP_DIR = ROOT / "data/harp_features"
HARP_MANIFEST = ROOT / "data/harp_manifest.json"
HARP_BASIS = ROOT / "data/harp_basis.npz"
HARP_BASIS_META = ROOT / "data/harp_basis.json"
GHOST_DIR = ROOT / "results/ghost_geometry_features_v1"
LUMINA_DIR = ROOT / "results/lumina_qa_features_v1"

PARTITIONS = ("fit", "calibration")
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_CLAIMS = {"fit": 9055, "calibration": 2267}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
EXPECTED_RELATION_WIDTH = 315
HARP_RANK = 205
HARP_WIDTH = 2 * HARP_RANK
FOLDS = 5
SEED = 20261102
THREADS = 4
EPOCHS = 120
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-2
RANK_WEIGHT = 0.25
RANK_MARGIN = 1.0
ROUTE_TEMPERATURE = 0.1
ROBUST_CLIP = 12.0
NLL_QUANTILE = 0.9

GHOST_NAMES = (
    "adjacent_layer_cosine_change",
    "layer_to_final_cosine",
    "top10_normalized_entropy",
    "top10_unweighted_embedding_divergence",
)
LUMINA_NAMES = (
    "ipr",
    "mmd",
    "lumina",
    "original_answer_probability",
    "original_max_probability",
    "random_answer_probability",
    "random_max_probability",
)
POOLINGS = ("mean", "max", "std")
TRACE_BASE_NAMES = (
    "nll_mean",
    "nll_max",
    "nll_min",
    "nll_std",
    *(f"ghost_{name}_{pool}" for name in GHOST_NAMES for pool in POOLINGS),
    *(f"lumina_{name}_{pool}" for name in LUMINA_NAMES for pool in POOLINGS),
    "microclaim_bpe_count",
    "microclaim_relative_midpoint",
)
TRACE_NAMES = TRACE_BASE_NAMES[:4] + ("nll_above_outer_train_q90_fraction",) + TRACE_BASE_NAMES[4:]
assert len(TRACE_BASE_NAMES) == 39 and len(TRACE_NAMES) == 40


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        assert read_json(path) == value, ("Frozen JSON changed", str(path))
        return
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def frozen_text(path: Path, text: str):
    path = Path(path)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, ("Frozen text changed", str(path))
    else:
        path.write_text(text, encoding="utf-8")


def atomic_npz(path: Path, **arrays):
    path = Path(path)
    assert not path.exists(), ("Refuse overwrite", str(path))
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    pending.replace(path)


def atomic_torch(path: Path, value):
    path = Path(path)
    assert not path.exists(), ("Refuse overwrite", str(path))
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    pending.replace(path)


def rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def protocol():
    return {
        "version": "atomic-evidence-router-v1",
        "role": "ours development candidate; formal baseline implementations and scores are read-only and untouched",
        "scope": {
            "answers": "Original RAGTruth QA development fit634 + calibration159 only.",
            "claims": "The 9,055 fit and 2,267 calibration lexical microclaims frozen by atomic_microclaim_nli_v1.",
            "windows": "Unchanged eligible four raw-BPE, stride-one windows; punctuation remains in width.",
            "official_test": "No official-test path or loader exists in this runner.",
        },
        "upstream_gate": {
            "required": "atomic_microclaim_nli_v1 complete.json plus evidence_features.npy, feature_names.json, microclaim_labels.npy and summary.json",
            "missing_behavior": "Write readiness.json with status WAIT_UPSTREAM, print WAIT, fit nothing, and emit no metric.",
            "relation_width": EXPECTED_RELATION_WIDTH,
            "supervised_upstream_prediction_used": False,
        },
        "views": {
            "relation": "The upstream frozen raw 315-dimensional evidence/relation feature row for the same microclaim.",
            "harp": "First 205 coordinates of the existing ascending bottom-256 output-head basis; per claim concatenate mean and signed maximum-absolute-value pooling (410 dims).",
            "trace": {
                "nll": "mean, max, min, std, and fraction above the current outer-training-fold 90th percentile",
                "ghost": "mean, max, std for each of four frozen token columns",
                "lumina": "mean, max, std for each of seven frozen token columns, including diagnostic probability columns as raw candidate inputs",
                "geometry": "raw BPE count and midpoint divided by max(answer_BPE_count-1,1)",
                "width": 40,
            },
            "labels_used_in_pooling": False,
        },
        "model": {
            "view_projections": {"relation": [315, 24], "harp": [410, 24], "trace": [40, 16]},
            "fusion": "GELU view projections -> concat64 -> Linear64x32 -> GELU -> Linear32x16 -> GELU",
            "routes": [4, 1],
            "routing": "softmax(cosine(r, prototype)/0.1); risk=logsumexp(log(route_probability)+route_linear_logit)",
            "loss": "weighted BCE + 0.25*mean softplus(1-risk_positive+risk_negative)",
            "pairs": "One deterministic opposite-label mate per training claim where possible; same answer first, then same source-connected group; deduplicate; never cross groups.",
            "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            "epochs_fixed_before_results": EPOCHS,
            "gradient_norm_clip": 5.0,
            "seed": SEED,
        },
        "crossfit": {
            "folds": FOLDS,
            "group": "source-connected group_id",
            "standardization": "Per view, outer-training-fold median and IQR; zero IQR becomes 1; transformed values clipped to [-12,12].",
            "nll_threshold": "Recomputed from only the outer training claims for each fold; full-fit value used only for a gated calibration pass.",
            "weights": "Equal group mass -> answer mass -> microclaim mass -> binary-class balance -> equal group loss mass.",
            "thresholds": "Window and answer thresholds chosen once from fit OOF separately for K=4 and K=1.",
        },
        "stage_gate": {
            "relation_control": "Best evidence__ candidate's fit-OOF window and answer F1 from the frozen upstream summary; calibration fields are never used by the gate.",
            "rule": "Open calibration only if K=4 fit-OOF window F1 and answer F1 are each strictly greater than both K=1 and the relation-only control.",
            "failure": "Save fit-OOF artifacts and a gate-failed report; do not access calibration labels or score calibration.",
            "calibration_selection": False,
        },
        "projection": "A lexical BPE inherits all overlapping microclaim risks; every four-BPE window and whole answer take the maximum overlapping risk.",
        "stages": ["initialize", "self-test", "prepare", "check", "status", "train", "verify"],
        "GPU": "Forbidden; every entry asserts CUDA was not initialized.",
    }


def plan_text():
    return """# Atomic Evidence Router v1\n\nThis is a CPU-only development candidate. It pools the existing HARP/NLL/GHOST/LUMINA token caches into the frozen atomic microclaims, then waits for the 315-dimensional raw relation/evidence output from `atomic_microclaim_nli_v1`. Missing upstream output produces `WAIT_UPSTREAM`; no surrogate feature or score is fabricated.\n\nFit uses five source-group folds. The fixed K=4 router and K=1 ablation share architecture, loss, epochs and preprocessing. The NLL tail cutoff and robust scalers are fitted inside each outer fold. Four-BPE windows and answer scores are max projections from microclaim risk. Calibration is read only after the frozen fit-OOF gate passes; it never selects structure, parameters, epoch or threshold. Official test and formal baseline code are outside this runner.\n\nExact commands from repository root:\n\n```powershell\nprelab\\.venv\\Scripts\\python.exe prelab\\benchmark_ragtruth_qa\\src\\run_atomic_evidence_router_v1.py status\nprelab\\.venv\\Scripts\\python.exe prelab\\benchmark_ragtruth_qa\\src\\run_atomic_evidence_router_v1.py train\nprelab\\.venv\\Scripts\\python.exe prelab\\benchmark_ragtruth_qa\\src\\run_atomic_evidence_router_v1.py verify\n```\n"""


class AtomicEvidenceRouter(torch.nn.Module):
    def __init__(self, routes: int):
        super().__init__()
        assert routes in (1, 4)
        self.routes = routes
        self.relation = torch.nn.Linear(EXPECTED_RELATION_WIDTH, 24)
        self.harp = torch.nn.Linear(HARP_WIDTH, 24)
        self.trace = torch.nn.Linear(len(TRACE_NAMES), 16)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.GELU(),
            torch.nn.Linear(32, 16), torch.nn.GELU(),
        )
        self.prototypes = torch.nn.Parameter(torch.empty(routes, 16))
        self.route_weight = torch.nn.Parameter(torch.empty(routes, 16))
        self.route_bias = torch.nn.Parameter(torch.zeros(routes))
        torch.nn.init.normal_(self.prototypes, std=0.25)
        torch.nn.init.xavier_uniform_(self.route_weight)

    def forward(self, relation, harp, trace):
        views = (
            torch.nn.functional.gelu(self.relation(relation)),
            torch.nn.functional.gelu(self.harp(harp)),
            torch.nn.functional.gelu(self.trace(trace)),
        )
        representation = self.fusion(torch.cat(views, dim=1))
        normalized_r = torch.nn.functional.normalize(representation, dim=1, eps=1e-8)
        normalized_c = torch.nn.functional.normalize(self.prototypes, dim=1, eps=1e-8)
        log_route = torch.log_softmax(normalized_r @ normalized_c.T / ROUTE_TEMPERATURE, dim=1)
        route_logits = representation @ self.route_weight.T + self.route_bias
        return torch.logsumexp(log_route + route_logits, dim=1)


def robust_fit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    center = np.median(matrix, axis=0).astype(np.float32)
    q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0).astype(np.float32)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0
    assert np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale > 0)
    return center, scale


def robust_apply(matrix, scaler):
    center, scale = scaler
    value = (np.asarray(matrix, dtype=np.float32) - center) / scale
    return np.clip(value, -ROBUST_CLIP, ROBUST_CLIP).astype(np.float32, copy=False)


def deterministic_pairs(response_ids, groups, labels):
    labels = np.asarray(labels, dtype=np.int8)
    response_ids = np.asarray(response_ids)
    groups = np.asarray(groups)
    by_response = defaultdict(lambda: {0: [], 1: []})
    by_group = defaultdict(lambda: {0: [], 1: []})
    for index, (rid, group, label) in enumerate(zip(response_ids, groups, labels)):
        by_response[str(rid)][int(label)].append(index)
        by_group[str(group)][int(label)].append(index)
    pairs = set()
    for index, (rid, group, label) in enumerate(zip(response_ids, groups, labels)):
        opposite = by_response[str(rid)][1 - int(label)]
        if not opposite:
            opposite = by_group[str(group)][1 - int(label)]
        if not opposite:
            continue
        mate = opposite[index % len(opposite)]
        positive, negative = (index, mate) if label == 1 else (mate, index)
        assert groups[positive] == groups[negative]
        pairs.add((int(positive), int(negative)))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def group_claim_weights(response_ids, groups, labels):
    response_ids, groups, labels = np.asarray(response_ids), np.asarray(groups), np.asarray(labels, dtype=np.int8)
    tree = defaultdict(lambda: defaultdict(list))
    for index, (rid, group) in enumerate(zip(response_ids, groups)):
        tree[str(group)][str(rid)].append(index)
    weight = np.zeros(len(labels), dtype=np.float64)
    for answers in tree.values():
        for indices in answers.values():
            weight[indices] = 1.0 / (len(answers) * len(indices))
    weight /= weight.mean()
    mass = np.bincount(labels, weights=weight, minlength=2)
    assert np.all(mass > 0)
    weight *= (mass.sum() / (2.0 * mass))[labels]
    for answers in tree.values():
        indices = [index for values in answers.values() for index in values]
        weight[indices] *= (len(labels) / len(tree)) / weight[indices].sum()
    weight *= len(labels) / weight.sum()
    assert np.isfinite(weight).all() and np.all(weight > 0)
    return weight.astype(np.float32)


def train_router(relation, harp, trace, labels, response_ids, groups, routes, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    assert not torch.cuda.is_initialized()
    model = AtomicEvidenceRouter(routes).cpu().train()
    tensors = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) for x in (relation, harp, trace)]
    target = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    weight = torch.from_numpy(group_claim_weights(response_ids, groups, labels))
    pairs = deterministic_pairs(response_ids, groups, labels)
    pair_tensor = torch.from_numpy(pairs) if len(pairs) else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    first_loss = final_loss = None
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*tensors)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (bce * weight).sum() / weight.sum()
        if pair_tensor is not None:
            positive, negative = pair_tensor[:, 0], pair_tensor[:, 1]
            ranking = torch.nn.functional.softplus(RANK_MARGIN - logits[positive] + logits[negative]).mean()
            loss = loss + RANK_WEIGHT * ranking
        if first_loss is None:
            first_loss = float(loss.detach())
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    assert not torch.cuda.is_initialized()
    return model, {"first_loss": first_loss, "final_loss": final_loss, "pairs": len(pairs)}


def predict(model, views):
    with torch.inference_mode():
        tensors = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) for x in views]
        result = torch.sigmoid(model(*tensors)).numpy().astype(np.float64)
    assert result.ndim == 1 and np.isfinite(result).all() and ((0 <= result) & (result <= 1)).all()
    return result


def synthetic_selfcheck():
    assert not torch.cuda.is_initialized()
    rng = np.random.default_rng(17)
    n = 24
    labels = np.tile(np.asarray([0, 1], np.int8), n // 2)
    groups = np.repeat(np.asarray([f"g{i}" for i in range(6)]), 4)
    responses = np.asarray([f"r{i // 2}" for i in range(n)])
    relation = rng.normal(size=(n, EXPECTED_RELATION_WIDTH)).astype(np.float32)
    harp = rng.normal(size=(n, HARP_WIDTH)).astype(np.float32)
    trace = rng.normal(size=(n, len(TRACE_NAMES))).astype(np.float32)
    relation[:, 0] += labels * 2
    scalers = [robust_fit(x[:16]) for x in (relation, harp, trace)]
    views = [robust_apply(x, scaler) for x, scaler in zip((relation, harp, trace), scalers)]
    pairs = deterministic_pairs(responses, groups, labels)
    assert len(pairs) and all(groups[p] == groups[n_] and labels[p] == 1 and labels[n_] == 0 for p, n_ in pairs)
    outputs = {}
    for routes in (1, 4):
        model, training = train_router(*views, labels, responses, groups, routes, 3 + routes)
        value = predict(model, views)
        assert value.shape == (n,) and np.isfinite(value).all()
        outputs[str(routes)] = {"parameters": sum(p.numel() for p in model.parameters()),
                                "finite": True, "training": training}
    owners = [[[0]], [[0]], [[1]], [[1]], [[1]]]
    windows = [[0, 1, 2, 3], [1, 2, 3, 4]]
    scores = np.asarray([0.2, 0.8])
    projected = [max(scores[c] for token in window for c in owners[token]) for window in windows]
    assert projected == [0.8, 0.8]
    folds = list(GroupKFold(3).split(np.arange(n), labels, groups))
    assert all(set(groups[train]).isdisjoint(set(groups[held])) for train, held in folds)
    return {
        "status": "passed",
        "model_forward": outputs,
        "same_group_rank_pairs": len(pairs),
        "group_fold_isolation": True,
        "max_projection": True,
        "trace_base_width": len(TRACE_BASE_NAMES),
        "trace_runtime_width": len(TRACE_NAMES),
        "GPU_initialized": torch.cuda.is_initialized(),
        "labels_from_real_data_accessed": False,
    }


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), ("Preserve existing run", str(OUT))
    OUT.mkdir(parents=True)
    atomic_json(OUT / "protocol.json", protocol(), frozen=True)
    frozen_text(OUT / "PROTOCOL.md", plan_text())
    atomic_json(OUT / "PREFLIGHT.json", synthetic_selfcheck(), frozen=True)
    print("ATOMIC_EVIDENCE_ROUTER_PROTOCOL_FROZEN", flush=True)


def manifest_maps():
    nll = read_json(NLL_MANIFEST)
    harp = read_json(HARP_MANIFEST)
    ghost = read_json(GHOST_DIR / "feature_manifest.json")
    lumina = read_json(LUMINA_DIR / "feature_manifest.json")
    assert nll["complete"] and nll["completed_records"] == 793
    assert harp["complete"] and harp["record_count"] == 793 and not harp["labels_or_test_read"]
    assert ghost["status"] == "complete" and ghost["records"] == 3839 and ghost["all_records_validated"]
    assert lumina["status"] == "complete" and lumina["records"] == 3839 and lumina["all_records_validated"]
    assert not ghost["test_opened"] and not ghost["labels_used"] and not ghost["trained"]
    assert not lumina["test_opened"] and not lumina["labels_used"] and not lumina["trained"]
    assert ghost["feature_names"] == list(GHOST_NAMES)
    assert lumina["feature_names"] == list(LUMINA_NAMES)
    with np.load(HARP_BASIS, allow_pickle=False) as saved:
        assert saved["components"].shape == (256, 4096)
        eigenvalues = saved["eigenvalues"]
        assert eigenvalues.shape == (256,) and np.all(np.diff(eigenvalues) >= 0)
    return (
        {str(x["response_id"]): x for x in nll["records"]},
        {str(x["response_id"]): x for x in harp["records"]},
        {str(x["response_id"]): x for x in ghost["entries"]},
        {str(x["response_id"]): x for x in lumina["entries"]},
    )


def load_token_sources(row, maps, verify_hash=True):
    rid = str(row["response_id"])
    nll_map, harp_map, ghost_map, lumina_map = maps
    assert all(rid in mapping for mapping in maps), rid
    paths = (
        NLL_DIR / f"{rid}.npz",
        HARP_DIR / f"{rid}.npz",
        GHOST_DIR / "features" / ghost_map[rid]["file"],
        LUMINA_DIR / "features" / lumina_map[rid]["file"],
    )
    expected_hashes = (
        nll_map[rid]["npz_sha256"], harp_map[rid]["npz_sha256"],
        ghost_map[rid]["npz_sha256"], lumina_map[rid]["npz_sha256"],
    )
    if verify_hash:
        for path, expected in zip(paths, expected_hashes):
            assert sha(path) == expected, (rid, str(path))
    with np.load(paths[0], allow_pickle=False) as saved:
        nll = saved["nll"].copy()
        token_ids = saved["token_ids"].astype(np.int64, copy=True)
        offsets = saved["response_token_offsets"].astype(np.int64, copy=True)
        raw_offsets = saved["response_token_offsets_raw"].astype(np.int64, copy=True)
    with np.load(paths[1], allow_pickle=False) as saved:
        harp = saved["harp"][:, :HARP_RANK].copy()
        assert np.array_equal(saved["token_ids"], token_ids)
        assert np.array_equal(saved["response_token_offsets"], offsets)
    with np.load(paths[2], allow_pickle=False) as saved:
        ghost = saved["ghost_features"].copy()
        assert str(saved["response_id"].item()) == rid
        assert np.array_equal(saved["token_ids"], token_ids)
        assert np.array_equal(np.column_stack((saved["token_start"], saved["token_end"])), offsets)
        assert np.array_equal(np.column_stack((saved["token_start_raw"], saved["token_end_raw"])), raw_offsets)
    with np.load(paths[3], allow_pickle=False) as saved:
        lumina = saved["lumina_features"].copy()
        assert str(saved["response_id"].item()) == rid
        assert np.array_equal(saved["token_ids"], token_ids)
        assert np.array_equal(np.column_stack((saved["token_start"], saved["token_end"])), offsets)
        assert np.array_equal(np.column_stack((saved["token_start_raw"], saved["token_end_raw"])), raw_offsets)
    n = len(token_ids)
    assert nll.shape == (n,) and harp.shape == (n, HARP_RANK)
    assert ghost.shape == (n, len(GHOST_NAMES)) and lumina.shape == (n, len(LUMINA_NAMES))
    assert all(x.dtype == np.float32 and np.isfinite(x).all() for x in (nll, harp, ghost, lumina))
    assert np.all(nll >= 0)
    return {"nll": nll, "harp": harp, "ghost": ghost, "lumina": lumina,
            "token_ids": token_ids, "offsets": offsets, "raw_offsets": raw_offsets}, paths


def pool_columns(matrix):
    values = []
    for column in range(matrix.shape[1]):
        one = matrix[:, column]
        values.extend((float(one.mean()), float(one.max()), float(one.std())))
    return values


def pool_claim(source, token_indices):
    indices = np.asarray(token_indices, dtype=np.int64)
    assert len(indices) and np.array_equal(indices, np.unique(indices))
    assert indices.min() >= 0 and indices.max() < len(source["nll"])
    harp = source["harp"][indices]
    maximum_rows = np.argmax(np.abs(harp), axis=0)
    signed_maxabs = harp[maximum_rows, np.arange(HARP_RANK)]
    harp_pooled = np.concatenate((harp.mean(0), signed_maxabs)).astype(np.float32)
    nll = source["nll"][indices].astype(np.float32, copy=True)
    trace = [float(nll.mean()), float(nll.max()), float(nll.min()), float(nll.std())]
    trace += pool_columns(source["ghost"][indices])
    trace += pool_columns(source["lumina"][indices])
    trace += [float(len(indices)), float(indices.mean() / max(1, len(source["nll"]) - 1))]
    trace = np.asarray(trace, dtype=np.float32)
    assert harp_pooled.shape == (HARP_WIDTH,) and trace.shape == (len(TRACE_BASE_NAMES),)
    assert np.isfinite(harp_pooled).all() and np.isfinite(trace).all() and np.isfinite(nll).all()
    return harp_pooled, trace, nll


def source_binding(maps, rows, selected):
    manifests = [NLL_MANIFEST, HARP_MANIFEST, HARP_BASIS, HARP_BASIS_META,
                 GHOST_DIR / "feature_manifest.json", GHOST_DIR / "features_complete.json",
                 LUMINA_DIR / "feature_manifest.json", LUMINA_DIR / "features_complete.json",
                 ATOMIC_INPUT, ATOMIC_PREP, Path(__file__)]
    identities = []
    for row in rows:
        rid = str(row["response_id"])
        identities.append({
            "response_id": rid, "partition": row["partition"],
            "nll": maps[0][rid]["npz_sha256"], "harp": maps[1][rid]["npz_sha256"],
            "ghost": maps[2][rid]["npz_sha256"], "lumina": maps[3][rid]["npz_sha256"],
        })
    return {
        "status": "label_free_token_sources_validated_and_pooled",
        "source_sha256": {str(path.resolve()): sha(path) for path in manifests},
        "selected_record_identity_sha256": digest(identities),
        "selected_record_count": len(identities),
        "validated_source_file_count": selected,
        "actual_reusable_fields": {
            "nll": ["nll", "token_ids", "response_token_offsets", "response_token_offsets_raw"],
            "harp": ["harp[:,0:205]", "token_ids", "response_token_offsets"],
            "ghost": ["ghost_features[:,0:4]", "token_ids", "token_start/end", "token_start/end_raw"],
            "lumina": ["lumina_features[:,0:7]", "token_ids", "token_start/end", "token_start/end_raw"],
            "atomic": ["response_id", "group_id", "partition", "claim_id", "microclaim_id", "lexical_token_indices", "lexical_token_microclaims"],
        },
        "labels_accessed": False, "GPU_used": False, "official_test_opened": False,
    }


def prepare():
    assert not torch.cuda.is_initialized()
    assert read_json(OUT / "protocol.json") == protocol()
    assert read_json(OUT / "PREFLIGHT.json")["status"] == "passed"
    assert not (OUT / "pooled_token_views.npz").exists()
    atomic_complete = read_json(ATOMIC_PREP)
    assert atomic_complete["status"] == "CPU_prepared_not_model_extracted"
    for name, expected in atomic_complete["files_sha256"].items():
        assert sha(UPSTREAM / name) == expected, name
    rows = read_jsonl(ATOMIC_INPUT)
    assert len(rows) == sum(EXPECTED_ANSWERS.values())
    assert Counter(row["partition"] for row in rows) == EXPECTED_ANSWERS
    assert Counter(claim["partition"] for row in rows for claim in row["claims"]) == EXPECTED_CLAIMS
    maps = manifest_maps()
    started = time.perf_counter()
    peak = rss_bytes()
    harp_parts, trace_parts, nll_parts = [], [], []
    response_ids, group_ids, partitions, claim_ids, microclaim_ids = [], [], [], [], []
    claim_ptr = [0]
    response_ptr = [0]
    answer_order = []
    selected_files = 0
    for answer_index, row in enumerate(rows):
        source, paths = load_token_sources(row, maps, verify_hash=True)
        selected_files += len(paths)
        rid = str(row["response_id"])
        assert [claim["claim_id"] for claim in row["claims"]] == list(range(len(row["claims"])))
        assert len(row["lexical_token_microclaims"]) == len(source["nll"])
        for claim in row["claims"]:
            h, trace, nll = pool_claim(source, claim["lexical_token_indices"])
            harp_parts.append(h); trace_parts.append(trace); nll_parts.append(nll)
            response_ids.append(rid); group_ids.append(str(row["group_id"])); partitions.append(row["partition"])
            claim_ids.append(int(claim["claim_id"])); microclaim_ids.append(str(claim["microclaim_id"]))
            claim_ptr.append(claim_ptr[-1] + len(nll))
        response_ptr.append(response_ptr[-1] + len(row["claims"])); answer_order.append(rid)
        current = rss_bytes()
        if current is not None:
            peak = current if peak is None else max(peak, current)
        if (answer_index + 1) % 100 == 0:
            print("ATOMIC_ROUTER_POOL", answer_index + 1, len(rows), flush=True)
    harp_matrix = np.asarray(harp_parts, dtype=np.float32)
    trace_matrix = np.asarray(trace_parts, dtype=np.float32)
    flat_nll = np.concatenate(nll_parts).astype(np.float32, copy=False)
    assert harp_matrix.shape == (sum(EXPECTED_CLAIMS.values()), HARP_WIDTH)
    assert trace_matrix.shape == (sum(EXPECTED_CLAIMS.values()), len(TRACE_BASE_NAMES))
    assert len(claim_ptr) == len(harp_matrix) + 1 and claim_ptr[-1] == len(flat_nll)
    atomic_npz(
        OUT / "pooled_token_views.npz",
        harp=harp_matrix, trace_base=trace_matrix, nll_values=flat_nll,
        nll_claim_ptr=np.asarray(claim_ptr, dtype=np.int64),
        response_ids=np.asarray(response_ids), group_ids=np.asarray(group_ids),
        partitions=np.asarray(partitions), claim_ids=np.asarray(claim_ids, dtype=np.int32),
        microclaim_ids=np.asarray(microclaim_ids), response_claim_ptr=np.asarray(response_ptr, dtype=np.int64),
        answer_order=np.asarray(answer_order),
    )
    binding = source_binding(maps, rows, selected_files)
    atomic_json(OUT / "source_binding.json", binding, frozen=True)
    elapsed = time.perf_counter() - started
    stats = {
        "status": "CPU_token_views_ready_waiting_for_atomic_relation_features",
        "answers": len(rows), "claims": len(harp_matrix), "nll_claim_token_occurrences": len(flat_nll),
        "pooled_widths": {"harp": HARP_WIDTH, "trace_base": len(TRACE_BASE_NAMES), "trace_with_fold_nll_tail": len(TRACE_NAMES)},
        "seconds": elapsed, "peak_process_rss_bytes_observed": peak,
        "pooled_npz_bytes": (OUT / "pooled_token_views.npz").stat().st_size,
        "labels_accessed": False, "GPU_used": False, "official_test_opened": False,
    }
    atomic_json(OUT / "preparation_statistics.json", stats, frozen=True)
    files = ("protocol.json", "PROTOCOL.md", "PREFLIGHT.json", "pooled_token_views.npz",
             "source_binding.json", "preparation_statistics.json")
    atomic_json(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_WAIT_UPSTREAM", "files_sha256": {name: sha(OUT / name) for name in files},
        "labels_accessed": False, "GPU_used": False, "official_test_opened": False,
    }, frozen=True)
    print("ATOMIC_EVIDENCE_ROUTER_CPU_PREPARED_WAIT_UPSTREAM", flush=True)


def load_prepared():
    assert not torch.cuda.is_initialized()
    complete = read_json(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_WAIT_UPSTREAM"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    assert read_json(OUT / "protocol.json") == protocol()
    binding = read_json(OUT / "source_binding.json")
    for name, expected in binding["source_sha256"].items():
        assert sha(Path(name)) == expected, name
    with np.load(OUT / "pooled_token_views.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    n = sum(EXPECTED_CLAIMS.values())
    assert arrays["harp"].shape == (n, HARP_WIDTH) and arrays["harp"].dtype == np.float32
    assert arrays["trace_base"].shape == (n, len(TRACE_BASE_NAMES)) and arrays["trace_base"].dtype == np.float32
    assert arrays["nll_claim_ptr"].shape == (n + 1,) and arrays["nll_claim_ptr"][0] == 0
    assert arrays["nll_claim_ptr"][-1] == len(arrays["nll_values"])
    assert np.all(np.diff(arrays["nll_claim_ptr"]) > 0)
    assert len(arrays["response_ids"]) == len(arrays["group_ids"]) == len(arrays["partitions"]) == n
    assert Counter(arrays["partitions"].tolist()) == EXPECTED_CLAIMS
    assert np.isfinite(arrays["harp"]).all() and np.isfinite(arrays["trace_base"]).all()
    assert np.isfinite(arrays["nll_values"]).all() and np.all(arrays["nll_values"] >= 0)
    return arrays


def check():
    arrays = load_prepared()
    rows = read_jsonl(ATOMIC_INPUT)
    maps = manifest_maps()
    response_ptr = arrays["response_claim_ptr"]
    samples = []
    for answer_index in (0, len(rows) // 2, len(rows) - 1):
        row = rows[answer_index]
        source, _ = load_token_sources(row, maps, verify_hash=True)
        local_claim = len(row["claims"]) // 2
        global_claim = int(response_ptr[answer_index]) + local_claim
        h, trace, nll = pool_claim(source, row["claims"][local_claim]["lexical_token_indices"])
        left, right = arrays["nll_claim_ptr"][global_claim:global_claim + 2]
        assert np.array_equal(h, arrays["harp"][global_claim])
        assert np.array_equal(trace, arrays["trace_base"][global_claim])
        assert np.array_equal(nll, arrays["nll_values"][left:right])
        samples.append({"response_id": str(row["response_id"]), "claim_id": local_claim,
                        "BPEs": len(nll), "exact_repool": True})
    result = {
        "status": "passed_WAIT_UPSTREAM", "sample_exact_repool": samples,
        "claim_identity_unique": len(set(zip(arrays["response_ids"].tolist(), arrays["claim_ids"].tolist()))) == len(arrays["claim_ids"]),
        "all_finite": True, "source_axes_cross_cache_equal": True,
        "real_labels_accessed": False, "GPU_used": False, "official_test_opened": False,
    }
    assert result["claim_identity_unique"]
    atomic_json(OUT / "CPU_CHECK.json", result, frozen=True)
    print("ATOMIC_EVIDENCE_ROUTER_CPU_CHECK_PASSED_WAIT_UPSTREAM", flush=True)


def upstream_readiness(write=True):
    required = ("complete.json", "evidence_features.npy", "feature_names.json",
                "microclaim_labels.npy", "summary.json")
    missing = [name for name in required if not (UPSTREAM / name).exists()]
    if missing:
        result = {
            "status": "WAIT_UPSTREAM", "upstream": "atomic_microclaim_nli_v1",
            "missing": missing, "fit_started": False, "metrics_emitted": False,
            "GPU_used": False, "official_test_opened": False,
        }
    else:
        complete = read_json(UPSTREAM / "complete.json")
        assert complete["status"] == "complete_development_only"
        for name, expected in complete["files_sha256"].items():
            assert sha(UPSTREAM / name) == expected, name
        names = read_json(UPSTREAM / "feature_names.json")
        assert len(names["evidence"]) == EXPECTED_RELATION_WIDTH
        evidence = np.load(UPSTREAM / "evidence_features.npy", mmap_mode="r")
        labels = np.load(UPSTREAM / "microclaim_labels.npy", mmap_mode="r")
        assert evidence.shape == (sum(EXPECTED_CLAIMS.values()), EXPECTED_RELATION_WIDTH)
        assert evidence.dtype == np.float32 and labels.shape == (sum(EXPECTED_CLAIMS.values()),)
        assert labels.dtype.kind in "biu"
        summary = read_json(UPSTREAM / "summary.json")
        assert summary["microclaims"] == sum(EXPECTED_CLAIMS.values())
        assert summary["fit_only_model_and_threshold_selection"] and not summary["official_test_opened"]
        result = {
            "status": "READY", "upstream": "atomic_microclaim_nli_v1",
            "complete_sha256": sha(UPSTREAM / "complete.json"),
            "evidence_sha256": sha(UPSTREAM / "evidence_features.npy"),
            "labels_sha256": sha(UPSTREAM / "microclaim_labels.npy"),
            "relation_width": EXPECTED_RELATION_WIDTH, "fit_started": False,
            "GPU_used": False, "official_test_opened": False,
        }
    if write:
        atomic_json(OUT / "readiness.json", result, frozen=False)
    return result


def status():
    load_prepared()
    state = upstream_readiness(True)
    print("ATOMIC_EVIDENCE_ROUTER_" + state["status"], json.dumps(state, ensure_ascii=False), flush=True)


def nll_tail_fraction(values, pointers, threshold):
    output = np.empty(len(pointers) - 1, dtype=np.float32)
    for index, (left, right) in enumerate(zip(pointers[:-1], pointers[1:])):
        output[index] = np.mean(values[left:right] > threshold)
    assert np.isfinite(output).all()
    return output


def fold_trace(trace_base, values, pointers, training_claims):
    pieces = [values[pointers[index]:pointers[index + 1]] for index in training_claims]
    threshold = float(np.quantile(np.concatenate(pieces), NLL_QUANTILE))
    tail = nll_tail_fraction(values, pointers, threshold)
    trace = np.column_stack((trace_base[:, :4], tail, trace_base[:, 4:])).astype(np.float32)
    assert trace.shape == (len(trace_base), len(TRACE_NAMES))
    return trace, threshold


def fit_partition_metadata():
    answers = read_jsonl(ROOT / "data/answers_fit.jsonl")
    tokens = read_jsonl(ROOT / "data/tokens_fit.jsonl")
    windows = read_jsonl(ROOT / "data/windows_k4_fit.jsonl")
    assert len(answers) == len(tokens) == EXPECTED_ANSWERS["fit"] and len(windows) == EXPECTED_WINDOWS["fit"]
    assert [str(x["response_id"]) for x in answers] == [str(x["response_id"]) for x in tokens]
    assert all(row["partition"] == "fit" for row in answers + tokens + windows)
    by_response = {str(answer["response_id"]): {"answer": answer, "tokens": token}
                   for answer, token in zip(answers, tokens)}
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[str(window["response_id"])].append(index)
    assert all(answer_windows[str(answer["response_id"])] for answer in answers)
    return {"partition": "fit", "answers": answers, "tokens": tokens, "windows": windows,
            "by_response": by_response, "answer_windows": dict(answer_windows)}


def calibration_partition_metadata():
    answers = read_jsonl(ROOT / "data/answers_calibration.jsonl")
    tokens = read_jsonl(ROOT / "data/tokens_calibration.jsonl")
    windows = read_jsonl(ROOT / "data/windows_k4_calibration.jsonl")
    assert len(answers) == len(tokens) == EXPECTED_ANSWERS["calibration"] and len(windows) == EXPECTED_WINDOWS["calibration"]
    assert [str(x["response_id"]) for x in answers] == [str(x["response_id"]) for x in tokens]
    assert all(row["partition"] == "calibration" for row in answers + tokens + windows)
    by_response = {str(answer["response_id"]): {"answer": answer, "tokens": token}
                   for answer, token in zip(answers, tokens)}
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[str(window["response_id"])].append(index)
    assert all(answer_windows[str(answer["response_id"])] for answer in answers)
    return {"partition": "calibration", "answers": answers, "tokens": tokens, "windows": windows,
            "by_response": by_response, "answer_windows": dict(answer_windows)}


def row_geometry():
    rows = read_jsonl(ATOMIC_INPUT)
    lookup = {str(row["response_id"]): row for row in rows}
    offsets = {}
    cursor = 0
    for row in rows:
        offsets[str(row["response_id"])] = cursor
        cursor += len(row["claims"])
    assert cursor == sum(EXPECTED_CLAIMS.values())
    return lookup, offsets


def project_partition(metadata, row_lookup, claim_offsets, claim_scores):
    window_scores = np.empty(len(metadata["windows"]), dtype=np.float64)
    for window_index, window in enumerate(metadata["windows"]):
        rid = str(window["response_id"])
        row = row_lookup[rid]
        owners = row["lexical_token_microclaims"]
        claim_ids = {int(cid) for token in window["token_indices"] for cid in owners[token]}
        assert claim_ids
        window_scores[window_index] = max(claim_scores[claim_offsets[rid] + cid] for cid in claim_ids)
    answer_scores = np.asarray([
        max(window_scores[metadata["answer_windows"][str(answer["response_id"])]])
        for answer in metadata["answers"]
    ], dtype=np.float64)
    assert np.isfinite(window_scores).all() and np.isfinite(answer_scores).all()
    return window_scores, answer_scores


def choose_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    assert set(labels.tolist()) == {0, 1} and np.isfinite(scores).all()
    order = np.argsort(-scores, kind="stable"); sorted_scores = scores[order]; sorted_labels = labels[order]
    last = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    tp = np.r_[0, np.cumsum(sorted_labels)[last]]; predicted = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[last]]
    f1 = 2 * tp / (predicted + labels.sum())
    precision = np.divide(tp, predicted, out=np.zeros(len(predicted), dtype=float), where=predicted > 0)
    best = max(range(len(thresholds)), key=lambda i: (f1[i], precision[i], thresholds[i]))
    return {"threshold": float(thresholds[best]), "fit_f1": float(f1[best]),
            "fit_precision": float(precision[best])}


def count_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.count_nonzero((labels == 1) & predicted)); fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted)); tn = int(np.count_nonzero((labels == 0) & ~predicted))
    return {
        "n": len(labels), "positive": int(labels.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def score_partition(metadata, window_scores, answer_scores, thresholds):
    return {
        "windows": count_metrics([row["label"] for row in metadata["windows"]], window_scores, thresholds["window"]["threshold"]),
        "answers": count_metrics([row["label"] for row in metadata["answers"]], answer_scores, thresholds["answer"]["threshold"]),
    }


def thresholds_from_fit(metadata, window_scores, answer_scores):
    return {
        "window": choose_threshold([row["label"] for row in metadata["windows"]], window_scores),
        "answer": choose_threshold([row["label"] for row in metadata["answers"]], answer_scores),
    }


def scale_views(relation, harp, trace, training_indices, prediction_indices):
    scalers = [robust_fit(matrix[training_indices]) for matrix in (relation, harp, trace)]
    train = [robust_apply(matrix[training_indices], scaler) for matrix, scaler in zip((relation, harp, trace), scalers)]
    predict_values = [robust_apply(matrix[prediction_indices], scaler) for matrix, scaler in zip((relation, harp, trace), scalers)]
    return train, predict_values, scalers


def state_bundle(model, scalers, nll_threshold, train_info):
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "scalers": [{"center": center, "scale": scale} for center, scale in scalers],
        "nll_threshold": nll_threshold, "training": train_info,
    }


def relation_control_from_upstream():
    summary = read_json(UPSTREAM / "summary.json")
    candidates = [(name, value) for name, value in summary["candidates"].items() if value["family"] == "evidence"]
    assert candidates
    name, value = max(candidates, key=lambda item: tuple(item[1]["selection_key"]))
    # Deliberately extract fit OOF only. No calibration value participates.
    return {"candidate": name, "fit_OOF_metrics": value["fit_OOF_metrics"],
            "selection_source": "upstream frozen fit OOF only"}


def run_crossfit(arrays, relation, fit_labels, fit_meta):
    partitions = arrays["partitions"]
    fit = np.flatnonzero(partitions == "fit")
    assert len(fit) == EXPECTED_CLAIMS["fit"] and len(fit_labels) == len(fit)
    groups = arrays["group_ids"]; responses = arrays["response_ids"]
    splits = list(GroupKFold(FOLDS).split(fit, fit_labels, groups[fit]))
    predictions = {1: np.full(len(arrays["harp"]), np.nan), 4: np.full(len(arrays["harp"]), np.nan)}
    bundles = {1: [], 4: []}; fold_reports = {1: [], 4: []}
    for routes in (1, 4):
        for fold, (train_local, held_local) in enumerate(splits):
            train_indices, held_indices = fit[train_local], fit[held_local]
            trace, nll_threshold = fold_trace(arrays["trace_base"], arrays["nll_values"], arrays["nll_claim_ptr"], train_indices)
            train_views, held_views, scalers = scale_views(relation, arrays["harp"], trace, train_indices, held_indices)
            model, train_info = train_router(
                *train_views, fit_labels[train_local], responses[train_indices], groups[train_indices],
                routes=routes, seed=SEED + routes * 100 + fold,
            )
            predictions[routes][held_indices] = predict(model, held_views)
            bundles[routes].append(state_bundle(model, scalers, nll_threshold, train_info))
            fold_reports[routes].append({
                "fold": fold, "train_claims": len(train_indices), "held_claims": len(held_indices),
                "train_groups": len(set(groups[train_indices])), "held_groups": len(set(groups[held_indices])),
                "nll_outer_train_q90": nll_threshold, **train_info,
            })
            print("ATOMIC_ROUTER_OOF", "K=" + str(routes), "fold", fold + 1, FOLDS, flush=True)
        assert np.isfinite(predictions[routes][fit]).all()
    row_lookup, claim_offsets = row_geometry()
    results = {}
    for routes in (1, 4):
        windows, answers = project_partition(fit_meta, row_lookup, claim_offsets, predictions[routes])
        thresholds = thresholds_from_fit(fit_meta, windows, answers)
        results[routes] = {
            "thresholds_from_fit_OOF": thresholds,
            "fit_OOF_metrics": score_partition(fit_meta, windows, answers, thresholds),
            "folds": fold_reports[routes],
            "window_scores": windows, "answer_scores": answers,
        }
    return predictions, bundles, results


def fit_full(arrays, relation, labels, routes):
    fit = np.flatnonzero(arrays["partitions"] == "fit")
    all_indices = np.arange(len(arrays["partitions"]))
    trace, nll_threshold = fold_trace(arrays["trace_base"], arrays["nll_values"], arrays["nll_claim_ptr"], fit)
    train_views, all_views, scalers = scale_views(relation, arrays["harp"], trace, fit, all_indices)
    model, train_info = train_router(
        *train_views, labels[fit], arrays["response_ids"][fit], arrays["group_ids"][fit],
        routes=routes, seed=SEED + routes * 1000,
    )
    scores = predict(model, all_views)
    return scores, state_bundle(model, scalers, nll_threshold, train_info)


def save_training_result(summary, fit_predictions, fit_results, bundles, full=None):
    assert not (OUT / "summary.json").exists()
    score_arrays = {}
    finite_fit = np.flatnonzero(np.isfinite(fit_predictions[1]))
    assert np.array_equal(finite_fit, np.flatnonzero(np.isfinite(fit_predictions[4])))
    score_arrays["fit_claim_indices"] = finite_fit
    for routes in (1, 4):
        score_arrays[f"claim_scores_K{routes}_fit_oof"] = fit_predictions[routes][finite_fit]
        score_arrays[f"window_scores_K{routes}_fit_oof"] = fit_results[routes]["window_scores"]
        score_arrays[f"answer_scores_K{routes}_fit_oof"] = fit_results[routes]["answer_scores"]
    if full is not None:
        for routes, value in full.items():
            score_arrays[f"claim_scores_K{routes}_fullfit"] = value["claim_scores"]
            score_arrays[f"window_scores_K{routes}_calibration"] = value["window_scores"]
            score_arrays[f"answer_scores_K{routes}_calibration"] = value["answer_scores"]
    atomic_npz(OUT / "scores.npz", **score_arrays)
    atomic_torch(OUT / "models.pt", {"crossfit": bundles,
                 "full_fit": None if full is None else {k: v["bundle"] for k, v in full.items()}})
    atomic_json(OUT / "summary.json", summary, frozen=True)
    lines = [
        "# Atomic Evidence Router v1", "",
        "CPU-only；四类 token 缓存先按原子微主张聚合。关系特征来自冻结上游原始 315 维矩阵；没有使用其监督预测。", "",
        "| 模型 | fit OOF 窗口F1 | fit OOF 整答F1 | calibration窗口F1 | calibration整答F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for routes in (1, 4):
        fit_metric = fit_results[routes]["fit_OOF_metrics"]
        cal_metric = summary.get("calibration", {}).get(f"K{routes}")
        cal_window = "未打开" if cal_metric is None else f"{cal_metric['windows']['f1']:.6f}"
        cal_answer = "未打开" if cal_metric is None else f"{cal_metric['answers']['f1']:.6f}"
        lines.append(f"| K={routes} | {fit_metric['windows']['f1']:.6f} | {fit_metric['answers']['f1']:.6f} | {cal_window} | {cal_answer} |")
    lines += ["", f"阶段门：**{summary['gate']['status']}**。", "", "正式 baseline 未修改；official test 未打开。"]
    frozen_text(OUT / "REPORT.md", "\n".join(lines) + "\n")
    names = ("summary.json", "REPORT.md", "scores.npz", "models.pt")
    atomic_json(OUT / "complete.json", {
        "status": summary["status"], "files_sha256": {name: sha(OUT / name) for name in names},
        "formal_baselines_modified": False, "official_test_opened": False,
    }, frozen=True)


def train():
    assert not torch.cuda.is_initialized()
    arrays = load_prepared()
    readiness = upstream_readiness(True)
    if readiness["status"] != "READY":
        print("ATOMIC_EVIDENCE_ROUTER_WAIT_UPSTREAM", json.dumps(readiness, ensure_ascii=False), flush=True)
        return
    assert not (OUT / "summary.json").exists(), "Preserve completed candidate"
    started = time.perf_counter()
    relation = np.load(UPSTREAM / "evidence_features.npy", mmap_mode="r")
    labels_mmap = np.load(UPSTREAM / "microclaim_labels.npy", mmap_mode="r")
    fit_indices = np.flatnonzero(arrays["partitions"] == "fit")
    # Access only fit labels until the OOF gate has been evaluated.
    fit_labels = np.asarray(labels_mmap[fit_indices], dtype=np.int8).copy()
    assert set(fit_labels.tolist()) == {0, 1}
    fit_meta = fit_partition_metadata()
    fit_predictions, bundles, fit_results = run_crossfit(arrays, relation, fit_labels, fit_meta)
    relation_control = relation_control_from_upstream()
    k1 = fit_results[1]["fit_OOF_metrics"]; k4 = fit_results[4]["fit_OOF_metrics"]
    rc = relation_control["fit_OOF_metrics"]
    passed = (
        k4["windows"]["f1"] > k1["windows"]["f1"] and
        k4["answers"]["f1"] > k1["answers"]["f1"] and
        k4["windows"]["f1"] > rc["windows"]["f1"] and
        k4["answers"]["f1"] > rc["answers"]["f1"]
    )
    gate = {"status": "PASSED" if passed else "FAILED_CALIBRATION_UNOPENED",
            "rule": protocol()["stage_gate"]["rule"], "relation_control": relation_control,
            "K1_fit_OOF": k1, "K4_fit_OOF": k4}
    common = {
        "method": "atomic_evidence_router_v1", "upstream_binding": readiness,
        "fit_OOF": {f"K{routes}": {key: value for key, value in fit_results[routes].items()
                                      if key not in ("window_scores", "answer_scores")} for routes in (1, 4)},
        "gate": gate, "calibration_used_for_structure_or_threshold": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    }
    if not passed:
        summary = {"status": "fit_oof_complete_gate_failed_calibration_unopened", **common,
                   "calibration_labels_accessed": False, "seconds": time.perf_counter() - started}
        save_training_result(summary, fit_predictions, fit_results, bundles)
        print("ATOMIC_EVIDENCE_ROUTER_GATE_FAILED_CALIBRATION_UNOPENED", flush=True)
        return

    # Only this branch accesses calibration labels and files.
    all_labels = np.empty(len(labels_mmap), dtype=np.int8)
    all_labels[fit_indices] = fit_labels
    calibration_indices = np.flatnonzero(arrays["partitions"] == "calibration")
    all_labels[calibration_indices] = np.asarray(labels_mmap[calibration_indices], dtype=np.int8)
    cal_meta = calibration_partition_metadata()
    row_lookup, claim_offsets = row_geometry()
    full = {}
    calibration = {}
    for routes in (1, 4):
        claim_scores, bundle = fit_full(arrays, relation, all_labels, routes)
        windows, answers = project_partition(cal_meta, row_lookup, claim_offsets, claim_scores)
        thresholds = fit_results[routes]["thresholds_from_fit_OOF"]
        calibration[f"K{routes}"] = score_partition(cal_meta, windows, answers, thresholds)
        full[routes] = {"claim_scores": claim_scores, "window_scores": windows,
                        "answer_scores": answers, "bundle": bundle}
    summary = {"status": "development_complete_after_fit_oof_gate", **common,
               "calibration": calibration, "calibration_labels_accessed": True,
               "seconds": time.perf_counter() - started}
    save_training_result(summary, fit_predictions, fit_results, bundles, full)
    print("ATOMIC_EVIDENCE_ROUTER_DEVELOPMENT_COMPLETE", flush=True)


def verify():
    load_prepared()
    if not (OUT / "complete.json").exists():
        state = upstream_readiness(True)
        print("ATOMIC_EVIDENCE_ROUTER_VERIFY_" + state["status"], flush=True)
        return
    complete = read_json(OUT / "complete.json")
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    summary = read_json(OUT / "summary.json")
    assert summary["status"] == complete["status"]
    assert not summary["formal_baselines_modified"] and not summary["official_test_opened"]
    with np.load(OUT / "scores.npz", allow_pickle=False) as saved:
        assert all(np.isfinite(saved[name]).all() for name in saved.files)
    print("ATOMIC_EVIDENCE_ROUTER_ALL_AVAILABLE_ARTIFACTS_VERIFIED", complete["status"], flush=True)


def record_failure(stage, error):
    try:
        folder = OUT / "failures"; folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.time_ns()}_{stage}.json"
        atomic_json(path, {
            "status": "FAILED_PRESERVED", "stage": stage, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "GPU_initialized": torch.cuda.is_initialized(), "official_test_opened": False,
        })
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "status", "train", "verify"))
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=THREADS):
            if args.stage == "initialize": initialize()
            elif args.stage == "self-test": print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
            elif args.stage == "prepare": prepare()
            elif args.stage == "check": check()
            elif args.stage == "status": status()
            elif args.stage == "train": train()
            else: verify()
        assert not torch.cuda.is_initialized()
    except Exception as error:
        record_failure(args.stage, error)
        raise


if __name__ == "__main__":
    main()
