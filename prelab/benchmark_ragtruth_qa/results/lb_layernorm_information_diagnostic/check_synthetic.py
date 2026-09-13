"""Synthetic CPU information-loss check. Does not fit or alter any old source."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
SOURCE = QA / "src/generation_residual_fusion.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    torch.set_num_threads(4)
    original_source_hash = sha(SOURCE)
    spec = importlib.util.spec_from_file_location("fusion_information_check", SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(20261011)
        head = module.GenerationResidualFusion().cpu().eval()
    base = torch.linspace(.125, .625, 1024)
    variants = {
        "base": base,
        "global_shift_plus_025": base + .25,
        "positive_scale_times_14": base * 1.4,
        "same_mean_dispersion_times_14": .375 + 1.4*(base-.375),
        "constant_025": torch.full((1024,), .25),
        "constant_075": torch.full((1024,), .75),
    }
    keys = list(variants)
    x = torch.stack([variants[k] for k in keys])
    assert bool(torch.isfinite(x).all() and (x >= 0).all() and (x <= 1).all())
    # All other inputs identical. Zero-initialized final residual is reported but
    # is NOT used to infer LN invariance; compare LN and the 24/32-d branch states.
    semantic = torch.linspace(-1., 1., 768).repeat(len(keys), 1)
    nll = torch.full((len(keys), 1), 2.)
    baseline = torch.full((len(keys),), .2)
    with torch.no_grad():
        norm = head.lookback[0](x)
        lb24 = head.lookback(x)
        components = head.components(semantic, baseline, x, nll)
    assert head.lookback[0].normalized_shape == (1024,)
    assert head.lookback[0].eps == 1e-5
    assert torch.equal(components["logit"], baseline)

    # This existing source-only post-read/no-header 1024-d model matches the LB
    # definition consumed by the proposed fusion. Read weights only, no real rows.
    model_dir = QA / "results/lookback_regularization_v2"
    summary_path = model_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = summary["selected"]["lb_source_post_legacy"]
    model_path = model_dir / (entry["candidate"] + ".pkl")
    assert sha(model_path) == entry["model_sha256"]
    obj = pickle.loads(model_path.read_bytes())
    model, scaler = obj["model"], obj["scaler"]
    assert model.n_features_in_ == 1024
    raw = x.numpy().astype(np.float64)
    # Fixed fit-only per-feature standardization retains inter-example shifts.
    standardized = scaler.transform(raw)
    logits = model.decision_function(standardized)
    probabilities = model.predict_proba(standardized)[:, 1]
    effective_weights = model.coef_[0]/scaler.scale_
    shift_sensitivity = float(effective_weights.sum())
    values = {}
    for i, name in enumerate(keys):
        values[name] = {
            "minimum": float(x[i].min()), "maximum": float(x[i].max()),
            "raw_mean": float(x[i].mean()), "raw_population_std": float(x[i].std(unbiased=False)),
            "analytic_unfitted_mean_linear_readout": float(raw[i].mean()),
            "frozen_lb_LR_logit": float(logits[i]), "frozen_lb_LR_probability": float(probabilities[i]),
            "zero_initial_residual_final_logit_not_evidence": float(components["logit"][i]),
        }
    comparisons = {}
    pairs = [("base", k) for k in keys[1:4]] + [("constant_025", "constant_075")]
    for left, right in pairs:
        i, j = keys.index(left), keys.index(right)
        ln_error = float((norm[i]-norm[j]).abs().max())
        branch_error = float((lb24[i]-lb24[j]).abs().max())
        comparisons[left + "__" + right] = {
            "layernorm_max_abs_delta": ln_error,
            "lookback24_max_abs_delta": branch_error,
            "generation32_max_abs_delta": float((components["generation"][i]-components["generation"][j]).abs().max()),
            "gate_abs_delta": float((components["gate"][i]-components["gate"][j]).abs()),
            "raw_mean_delta": values[right]["raw_mean"]-values[left]["raw_mean"],
            "raw_std_delta": values[right]["raw_population_std"]-values[left]["raw_population_std"],
            "frozen_lb_LR_logit_delta": float(logits[j]-logits[i]),
            "frozen_lb_LR_probability_delta": float(probabilities[j]-probabilities[i]),
        }
    assert comparisons["base__global_shift_plus_025"]["layernorm_max_abs_delta"] < 2e-6
    assert comparisons["base__same_mean_dispersion_times_14"]["layernorm_max_abs_delta"] < .001
    assert comparisons["constant_025__constant_075"]["lookback24_max_abs_delta"] == 0.
    assert abs(shift_sensitivity) > 1e-6
    shift_delta = comparisons["base__global_shift_plus_025"]["frozen_lb_LR_logit_delta"]
    assert np.isclose(shift_delta, .25*shift_sensitivity, rtol=1e-5, atol=1e-5)
    assert sha(SOURCE) == original_source_hash and sha(model_path) == entry["model_sha256"]
    np.savez_compressed(OUT / "synthetic_arrays.npz", names=np.array(keys), legal_lb=raw,
                        layernorm=norm.numpy(), lookback24=lb24.numpy(),
                        generation32=components["generation"].numpy())
    result = {
        "status": "passed_information_loss_confirmed", "source_sha256": original_source_hash,
        "synthetic_only": True, "seed": 20261011, "device": "cpu", "dtype": "float32",
        "layernorm_normalized_shape": [1024], "layernorm_eps": 1e-5,
        "variants": values, "comparisons": comparisons,
        "frozen_linear_reference": {"candidate": entry["candidate"], "model_sha256": entry["model_sha256"],
                                    "summary_sha256": sha(summary_path), "effective_weight_sum_shift_sensitivity": shift_sensitivity,
                                    "same_legacy_LB_definition": True, "no_new_model_fit": True},
        "reasoning": {
            "shift": "LN(x+b*1)=LN(x) in exact arithmetic for any learned elementwise gamma/beta: the per-token head mean is removed.",
            "scale": "For a>0, LN(a*x+b)=gamma*(x-mean(x))/sqrt(var(x)+eps/a^2)+beta. With variance much larger than eps, scale differences are strongly compressed, not mathematically erased when eps>0.",
            "constant_counterexample": "All heads .25 vs all heads .75 have exactly equal LN and branch outputs here; no later head using only those branch states can reconstruct the lost mean.",
            "zero_residual": "Final fusion logits are all identical by deliberate zero residual initialization, so final-output equality alone would be invalid evidence. Intermediate LN/Lookback24/generation32 are the relevant checks.",
            "raw_LR": "A linear map mean(x) can distinguish global levels without any training. The actual saved 1024-d LR also has nonzero shift sensitivity after fixed train-only feature standardization.",
        },
        "smallest_proposal_not_implemented": "Keep LN->24 relative-head representation and append raw per-token LB mean and population std after fit-only fixed scaling; do not LayerNorm these two scalar summaries away. Fusion input becomes67 instead of65. If preserving65 is essential, use LN->22 plus2 summaries as the24-d LB branch in a separate version/control.",
        "limits": ["Synthetic [0,1] vectors demonstrate representational loss, not a measured real-data F1 effect.",
                   "Semantic and NLL channels might indirectly correlate with mean/dispersion; this does not restore guaranteed access to the discarded LB summaries.",
                   "LayerNorm still retains relative head patterns. This result does not imply the branch is useless.",
                   "No changes to frozen or running source, weights, protocol, real training data or current v2 baseline."],
        "trained": False, "GPU_used": False, "real_fit_calibration_test_rows_opened": False,
        "script_sha256": sha(__file__), "arrays_sha256": sha(OUT / "synthetic_arrays.npz"),
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = ["# LB逐词元LayerNorm的信息损失", "",
              "确认：现有模块在每个词元的1024个头之间做LayerNorm，会消除所有头共同升降的平均水平；正比例缩放的影响也被大幅压缩。它保留相对头形状，不能完整保留绝对资料依赖程度。", "",
              "只使用合法[0,1]合成向量，其他语义状态/NLL/基线logit全部相同；未拟合、未GPU、未读取真实训练或测试样本。", "",
              "| 对比 | 原LB均值变化 | LN输出最大差 | LB24支路最大差 | 已冻结LB线性logit差 |",
              "|---|---:|---:|---:|---:|"]
    for name, v in comparisons.items():
        report.append(f'| {name} | {v["raw_mean_delta"]:+.6f} | {v["layernorm_max_abs_delta"]:.3g} | {v["lookback24_max_abs_delta"]:.3g} | {v["frozen_lb_LR_logit_delta"]:+.6f} |')
    report += ["", "全0.25与全0.75的向量被映射成完全相同的LB支路输出。平移不变性对后续学习到的LayerNorm缩放/偏置也成立。缩放仅近似不变：eps=1e-5仍留下微弱差异，不能声称方差信息在数学上完全消失。", "",
               "注意最终残差头初始化为0，所以所有最终logit本来就相同；本检查使用归一化与中间支路输出作为证据，未用这个零头现象误判。冻结LR的合成概率只说明能区分输入，不是这些合成向量的真实幻觉概率。", "",
               "最小建议：保留现有LN相对形状支路，额外旁路输入该词元的LB均值和标准差；两个标量只用训练集固定统计量做尺度统一，不再做逐词元LN。只增加2维（65→67）；若必须同总宽，可另版用22维形状+2维统计量替代原24维。现有冻结版本和v2基线均不改。", "",
               "这证明信息路径存在损失，不能单凭合成例子断言它解释了既有F1差距，或保证补回后提高成绩。"]
    (OUT / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparisons": comparisons,
                      "shift_sensitivity": shift_sensitivity}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
