"""CPU-only R25 formula compatibility checks; no data/cache/weights are read.

Uses random tiny Qwen2, synthetic token IDs and the pinned R7 reference.
R19 stats/mmd are AST-isolated: importing/preparing R19 is deliberately avoided.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM


ROOT = Path(__file__).resolve().parents[1]
R7 = ROOT.parent / "round7_evidence_grounding"
R19 = ROOT.parent / "round19_three_signal_probe/src/extract19.py"
R25 = ROOT / "src/extract25.py"
MEASUREMENTS = {}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


port = load_file("r25_formula_port7", R7 / "src/lumina7.py")
official_path = R7 / "references/lumina_official.py"
assert hashlib.sha256(official_path.read_bytes()).hexdigest() == port.OFFICIAL_SHA256
official = load_file("r25_formula_official", official_path)
tree = ast.parse(R19.read_text(encoding="utf-8"))
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"stats", "mmd"}]
assert {n.name for n in functions} == {"stats", "mmd"}
r19 = {"lumina7": port}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(R19), "exec"), r19)


def r25_pure_functions():
    tree = ast.parse(R25.read_text(encoding="utf-8"))
    names = {"joint_prefix", "mix_scores"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    constants = {target.id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name) and target.id == "LAMBDA"}
    assert set(constants) == {"LAMBDA"}
    scope = {"np": np, **constants}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(R25), "exec"), scope)
    return scope


class Formula25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(2501)
        cfg = Qwen2Config(vocab_size=128, hidden_size=24, intermediate_size=48,
                          num_hidden_layers=3, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=256,
                          attention_dropout=0.0, sliding_window=None,
                          use_sliding_window=False)
        cfg._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(cfg).cpu().eval()
        # A non-identity norm makes accidentally omitting the second final norm
        # observable; default unit weights can conceal that convention error.
        with torch.no_grad():
            cls.model.model.norm.weight.copy_(torch.linspace(0.4, 1.8, cfg.hidden_size))
        assert next(cls.model.parameters()).device.type == "cpu"
        cls.reference = official.LUMINA(cls.model, None, device="cpu")
        cls.prefix = [3, 5, 7, 11, 13, 17, 19, 23, 29]
        cls.answer = [31, 37, 41, 43, 47]

    @torch.inference_mode()
    def test_p_minus_one_and_all_output_layers(self):
        ids = torch.tensor([self.prefix + self.answer])
        full = self.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        p, targets, expected = self.reference._LUMINA__get_ans(
            full.logits, ids, torch.tensor([self.prefix]), full.hidden_states[1:])
        final, layers = port.collect_response_hidden(self.model, self.prefix, self.answer)
        start = len(self.prefix) - 1
        self.assertEqual(len(layers), self.model.config.num_hidden_layers)
        self.assertEqual(targets.tolist(), self.answer)
        self.assertEqual(final.shape, (len(self.answer), self.model.config.hidden_size))
        for got, ref in zip(layers, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        torch.testing.assert_close(final[0], full.hidden_states[-1][0, start], rtol=0, atol=0)
        torch.testing.assert_close(final[-1], full.hidden_states[-1][0, start + len(self.answer) - 1], rtol=0, atol=0)
        # Wrong P+j slice must fail for the nonconstant synthetic response.
        self.assertGreater(float((final-full.hidden_states[-1][0, start+1:start+1+len(self.answer)]).abs().max()), 0.01)
        stats = port.final_statistics(self.model, final, self.answer, chunk_size=2)
        torch.testing.assert_close(stats["answer_probs"], p[torch.arange(len(self.answer)), targets])

    @torch.inference_mode()
    def test_streaming_ipr_pinned_official_and_double_final_norm(self):
        final, layers = port.collect_response_hidden(self.model, self.prefix, self.answer)
        p = self.model.lm_head(final).float().softmax(-1)
        lens = [self.model.lm_head(self.model.model.norm(h)).float().softmax(-1) for h in layers]
        targets = torch.tensor(self.answer)
        expected = self.reference._LUMINA__compute_ipr(lens, p, targets)
        results = []
        for chunk in (1, 2, 16):
            stats = port.final_statistics(self.model, final, self.answer, chunk_size=chunk)
            actual = port.ipr_from_hidden(self.model, layers, stats, chunk_size=chunk)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
            self.assertEqual(actual.dtype, torch.float32)
            results.append(actual)
        wrong_once = self.reference._LUMINA__compute_ipr(lens[:-1] + [p], p, targets)
        wrong_omit = self.reference._LUMINA__compute_ipr(lens[:-1], p, targets)
        self.assertGreater(float((expected-wrong_once).abs().max()), 1e-5)
        self.assertGreater(float((expected-wrong_omit).abs().max()), 1e-5)
        MEASUREMENTS["ipr_max_abs_vs_pinned"] = max(float((x-expected).abs().max()) for x in results)
        MEASUREMENTS["omitting_second_final_norm_changes_ipr"] = float((expected-wrong_once).abs().max())

    @torch.inference_mode()
    def test_same_length_future_tokens_do_not_change_pre_token_statistics(self):
        changed = list(self.answer)
        changed[2:] = [71, 73, 79]
        final, layers = port.collect_response_hidden(self.model, self.prefix, self.answer)
        other_final, other_layers = port.collect_response_hidden(self.model, self.prefix, changed)
        # token index 2 is also unchanged: its prediction precedes its input ID.
        for x, y in zip(layers, other_layers):
            torch.testing.assert_close(x[:3], y[:3], rtol=0, atol=0)
        a = port.final_statistics(self.model, final, self.answer)
        b = port.final_statistics(self.model, other_final, changed)
        for key in ("top_probs", "top_ids", "max_probs", "max_ids"):
            torch.testing.assert_close(a[key][:3], b[key][:3], rtol=0, atol=0)
        # Actual-target correction at changed token index 2 must not be compared.
        torch.testing.assert_close(a["answer_probs"][:2], b["answer_probs"][:2], rtol=0, atol=0)

    @torch.inference_mode()
    def test_mmd_top100_preserves_mass_and_matches_dense_kernel(self):
        generator = torch.Generator().manual_seed(2502)
        embedding = torch.nn.Embedding(128, 24).cpu()
        embedding.weight.copy_(torch.randn(128, 24, generator=generator))
        p = torch.randn(5, 128, generator=generator).softmax(-1)
        q = (2.5*torch.randn(5, 128, generator=generator)).softmax(-1)
        pp, pi = p.topk(100, -1)
        qp, qi = q.topk(100, -1)
        expected = self.reference._LUMINA__compute_mmd(p, q, embedding, k=100)
        actual = port.cosine_mmd_from_topk(pp, pi, qp, qi, embedding, chunk_size=2)
        pe, qe = F.normalize(embedding(pi), dim=-1), F.normalize(embedding(qi), dim=-1)
        displacement = (pp[..., None]*pe).sum(1)-(qp[..., None]*qe).sum(1)
        manual = 0.5*((pp.sum(-1)-qp.sum(-1)).square()+displacement.square().sum(-1))
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-7)
        torch.testing.assert_close(actual, manual, rtol=1e-6, atol=1e-8)
        self.assertGreater(float((pp.sum(-1)-qp.sum(-1)).abs().max()), 0.01)
        wrong = port.cosine_mmd_from_topk(pp/pp.sum(-1, keepdim=True), pi,
                                         qp/qp.sum(-1, keepdim=True), qi, embedding)
        self.assertGreater(float((actual-wrong).abs().max()), 1e-4)
        identical = port.cosine_mmd_from_topk(pp, pi, pp, pi, embedding)
        torch.testing.assert_close(identical, torch.zeros_like(identical), rtol=0, atol=0)
        MEASUREMENTS["mmd_max_abs_vs_pinned"] = float((actual-expected).abs().max())

    @torch.inference_mode()
    def test_r19_pure_stats_and_original_slot_columns(self):
        prefix = np.asarray(self.prefix)
        masks = [[2, 3], [6, 7]]
        donors = [[53, 59], [61, 67]]
        changes, columns = [], []
        p = r19["stats"](self.model, prefix.tolist(), self.answer)
        for mask, replacements in zip(masks, donors):
            one = prefix.copy()
            one[mask] = replacements
            changes.append(one)
            q = r19["stats"](self.model, one.tolist(), self.answer)
            columns.append(r19["mmd"](self.model, p, q))
        reused = np.stack(columns, axis=1)
        self.assertEqual(reused.shape, (len(self.answer), 2))
        self.assertEqual(reused.dtype, np.float32)
        for slot, one in enumerate(changes):
            full = self.model(input_ids=torch.tensor([one.tolist()+self.answer]), use_cache=False)
            q, _ = self.reference._LUMINA__get_ans(full.logits,
                torch.tensor([one.tolist()+self.answer]), torch.tensor([one.tolist()]))
            full_p = self.model(input_ids=torch.tensor([prefix.tolist()+self.answer]), use_cache=False)
            original, _ = self.reference._LUMINA__get_ans(full_p.logits,
                torch.tensor([prefix.tolist()+self.answer]), torch.tensor([prefix.tolist()]))
            expected = self.reference._LUMINA__compute_mmd(original, q, self.model.get_input_embeddings(), k=100)
            np.testing.assert_allclose(reused[:, slot], expected.numpy(), rtol=1e-3, atol=2e-7)
        # Both bodies require a joint forward. Individual columns remain intact.
        joint = prefix.copy()
        for mask, replacements in zip(masks, donors):
            joint[mask] = replacements
        changed_positions = np.flatnonzero(joint != prefix).tolist()
        self.assertEqual(changed_positions, sorted(masks[0]+masks[1]))
        joint_q = r19["stats"](self.model, joint.tolist(), self.answer)
        joint_mmd = r19["mmd"](self.model, p, joint_q)
        self.assertGreater(float(np.max(np.abs(joint_mmd-reused.sum(axis=1)))), 1e-8)
        MEASUREMENTS["joint_vs_sum_of_single_mmd_max_abs"] = float(np.max(np.abs(joint_mmd-reused.sum(axis=1))))

    def test_r25_joint_uses_both_disjoint_masks_and_rejects_malformed_plans(self):
        compose = r25_pure_functions()["joint_prefix"]
        original = list(range(11, 21))

        def entry(slot, mask, used):
            prefix = original.copy()
            for i, value in zip(mask, used):
                prefix[i] = value
            return {"passage_index": slot, "eligible_body_token_positions": mask,
                    "replacement_body_token_ids": used,
                    "replacement_prefix_token_ids": prefix,
                    "changed_token_positions": [i for i, (a, b) in enumerate(zip(original, prefix)) if a != b]}

        # One eligible position deliberately retains the same token ID.
        interventions = [entry(0, [1, 2], [71, original[2]]), entry(1, [5, 6], [73, 79])]
        untouched = copy.deepcopy(interventions)
        joint, mask, changes = compose(original, interventions)
        expected = original.copy()
        expected[1], expected[5], expected[6] = 71, 73, 79
        self.assertEqual(list(joint), expected)
        self.assertEqual(list(mask), [1, 2, 5, 6])
        self.assertEqual(list(changes), [1, 5, 6])
        self.assertEqual(original, list(range(11, 21)))
        self.assertEqual(interventions, untouched)
        malformed = [interventions[:1], [interventions[0], entry(1, [2, 3], [83, 89])]]
        bad = copy.deepcopy(interventions)
        bad[0]["eligible_body_token_positions"] = [1, len(original)]
        malformed.append(bad)
        bad = copy.deepcopy(interventions)
        bad[0]["replacement_body_token_ids"] = [71]
        malformed.append(bad)
        bad = copy.deepcopy(interventions)
        bad[0]["replacement_prefix_token_ids"][0] = 97
        bad[0]["changed_token_positions"] = [0, 1]
        malformed.append(bad)
        bad = copy.deepcopy(interventions)
        bad[0]["replacement_prefix_token_ids"].append(101)
        malformed.append(bad)
        bad = copy.deepcopy(interventions)
        bad[0]["changed_token_positions"] = [1, 2]
        malformed.append(bad)
        for index, bad in enumerate(malformed):
            with self.subTest(malformed=index), self.assertRaises((AssertionError, ValueError, IndexError)):
                compose(original, bad)

    def test_r25_mix_is_signed_unclipped_float32(self):
        mix = r25_pure_functions()["mix_scores"]
        ipr = np.asarray([0.25, 1.0, 0.0, 0.5], dtype=np.float32)
        mmd = np.asarray([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        expected = np.asarray([-0.375, 0.5, 0.0, 0.0], dtype=np.float32)
        actual = mix(ipr, mmd)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(mix(ipr, mmd, lam=0.0), -mmd)
        np.testing.assert_array_equal(mix(ipr, mmd, lam=1.0), ipr)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Formula25Tests))
    source_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(port.__file__), official_path, R19, R25)}
    report = {"status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun, "device": "CPU", "data_or_gold_read": False,
        "r19_prepare_called": False, "measurements": MEASUREMENTS,
        "extract25_sha256": source_hashes[str(R25.resolve())],
        "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "limits": "Synthetic formula and AST-isolated pure API checks only; full R25 extraction and actual NF4 GPU behavior not validated here."}
    destination = ROOT / "data/cpu_selfcheck.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
