"""CPU-only mathematical and causal checks against pinned official LUMINA.

Run with the existing prelab Python; no weights are downloaded or GPU used.
"""
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumina7 as port


def load_official():
    path = ROOT / "references/lumina_official.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != port.OFFICIAL_SHA256:
        raise AssertionError("Official reference changed from pinned commit")
    spec = importlib.util.spec_from_file_location("lumina_official_pinned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LUMINA


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
        return [ord(c) for c in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(i)) for i in ids)


def make_fixture():
    tok = CharacterTokenizer()
    row = {"row_id": "target__partial", "question_id": "target", "system": "Helpful.",
           "questions": ["Where is A?", "Where is B?"],
           "passages": [{"title": "A", "text": "A is in Peru."},
                        {"title": "B", "text": "B is a city."}]}
    row["prompt"] = "Questions:\n1. Where is A?\n2. Where is B?" + port.MARKER + port._render_passages(row["passages"])
    random_row = {"row_id": "random__complete", "question_id": "random", "questions": ["IGNORE OTHER QUESTION"],
                  "passages": [{"title": "C", "text": "C is a river."},
                               {"title": "D", "text": "D is a ship."}]}
    response = "1. Peru.\n2. Iran."
    generated = {"row_id": row["row_id"], "response": response,
                 "input_token_ids": port._chat_ids(tok, row["prompt"], row["system"]),
                 "response_token_ids": [ord(c) for c in response],
                 "response_token_offsets": [[i, i+1] for i in range(len(response))],
                 "items": [{"item_id": "target__partial__1", "item_index": 1, "text": "Peru.",
                            "start": 3, "end": 8, "last_content_character": 6, "parse_ok": True},
                           {"item_id": "target__partial__2", "item_index": 2, "text": "Iran.",
                            "start": 12, "end": 17, "last_content_character": 15, "parse_ok": True}]}
    return tok, row, generated, random_row


class LuminaFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(982)
        config = Qwen2Config(vocab_size=128, hidden_size=24, intermediate_size=48,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=2048,
                             attention_dropout=0.0, bos_token_id=1, eos_token_id=2)
        config._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(config).eval().cpu()
        cls.official_class = load_official()

    def test_cosine_mmd_is_official_kernel_sum_without_renormalization(self):
        generator = torch.Generator().manual_seed(71)
        embedding = torch.nn.Embedding(128, 24)
        with torch.no_grad():
            embedding.weight.copy_(torch.randn(128, 24, generator=generator))
        p = torch.randn(5, 128, generator=generator).softmax(-1)
        q = (torch.randn(5, 128, generator=generator)*2.0).softmax(-1)
        official = self.official_class(self.model, None, device="cpu")
        expected = official._LUMINA__compute_mmd(p, q, embedding, k=100)
        pp, pi = p.topk(100, -1)
        qp, qi = q.topk(100, -1)
        actual = port.cosine_mmd_from_topk(pp, pi, qp, qi, embedding, chunk_size=2)
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-7)
        self.assertGreater(float((pp.sum(-1)-qp.sum(-1)).abs().max()), 0.01)
        self.assertFalse(torch.allclose(actual, port.cosine_mmd_from_topk(
            pp/pp.sum(-1, keepdim=True), pi, qp/qp.sum(-1, keepdim=True), qi, embedding)))

    @torch.inference_mode()
    def test_capture_matches_hugging_face_all_output_layers_and_prefix(self):
        prefix, answer = [5, 8, 12, 90], [11, 21, 31, 41, 51]
        full = self.model.model(input_ids=torch.tensor([prefix+answer]),
                                output_hidden_states=True, use_cache=False)
        final, layers = port.collect_response_hidden(self.model, prefix, answer)
        first = len(prefix)-1
        for index, expected in enumerate(full.hidden_states[1:]):
            torch.testing.assert_close(layers[index], expected[0, first:first+len(answer)])
        torch.testing.assert_close(final, full.last_hidden_state[0, first:first+len(answer)])
        # The last capture is post-final-normalization, not raw final block.
        self.assertEqual(len(layers), self.model.config.num_hidden_layers)
        short_final, short_layers = port.collect_response_hidden(self.model, prefix, answer[:3])
        for a, b in zip(layers, short_layers):
            torch.testing.assert_close(a[:3], b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(final[:3], short_final, rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_streamed_ipr_matches_official_reference(self):
        prefix, answer = [4, 15, 55], [12, 33, 34, 35, 100]
        final, layers = port.collect_response_hidden(self.model, prefix, answer)
        stats = port.final_statistics(self.model, final, answer, chunk_size=2)
        p = F.softmax(self.model.lm_head(final), dim=-1)
        hid_prob = [F.softmax(self.model.lm_head(self.model.model.norm(h)), dim=-1) for h in layers]
        official = self.official_class(self.model, None, device="cpu")
        expected = official._LUMINA__compute_ipr(hid_prob, p, torch.tensor(answer))
        actual = port.ipr_from_hidden(self.model, layers, stats, chunk_size=2)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)

    @torch.inference_mode()
    def test_end_to_end_exact_ids_official_formula_and_chunk_invariance(self):
        tok, row, generated, random_row = make_fixture()
        values, metadata = port.extract_lumina(tok, self.model, row, generated, random_row, chunk_size=3)
        again, _ = port.extract_lumina(tok, self.model, row, generated, random_row, chunk_size=7)
        for key in ("token_lumina_mmd", "token_lumina_ipr", "token_lumina_score"):
            np.testing.assert_allclose(values[key], again[key], rtol=2e-5, atol=3e-7)
        answer = generated["response_token_ids"]
        original_prefix = generated["input_token_ids"]
        random_prefix = port._chat_ids(tok, port.build_random_prompt(row, random_row), row["system"])
        original = self.model(input_ids=torch.tensor([original_prefix+answer]), output_hidden_states=True, use_cache=False)
        other = self.model(input_ids=torch.tensor([random_prefix+answer]), use_cache=False)
        official = self.official_class(self.model, tok, device="cpu")
        p, targets, layers = official._LUMINA__get_ans(original.logits, torch.tensor([original_prefix+answer]),
                                                        torch.tensor([original_prefix]), original.hidden_states[1:])
        q, _ = official._LUMINA__get_ans(other.logits, torch.tensor([random_prefix+answer]), torch.tensor([random_prefix]))
        expected_mmd = official._LUMINA__compute_mmd(p, q, self.model.get_input_embeddings(), k=100)
        hid_prob = [F.softmax(self.model.lm_head(self.model.model.norm(h)), dim=-1) for h in layers]
        expected_ipr = official._LUMINA__compute_ipr(hid_prob, p, targets)
        np.testing.assert_allclose(values["token_lumina_mmd"], expected_mmd.numpy(), rtol=1e-3, atol=2e-7)
        np.testing.assert_allclose(values["token_lumina_ipr"], expected_ipr.numpy(), rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(values["token_lumina_score"], ((expected_ipr-expected_mmd)*0.5).numpy(), rtol=2e-5, atol=2e-7)
        self.assertEqual(values["item_ids"].tolist(), [x["item_id"] for x in generated["items"]])
        self.assertEqual(metadata["forward_passes"], 2)
        self.assertEqual(metadata["top_k"], 100)
        self.assertFalse(metadata["top_k_renormalized"])

    @torch.inference_mode()
    def test_first_item_does_not_use_second_item_or_future_tokens(self):
        tok, row, generated, random_row = make_fixture()
        full, _ = port.extract_lumina(tok, self.model, row, generated, random_row, chunk_size=4)
        short = copy.deepcopy(generated)
        end = short["items"][0]["end"]
        short["response"] = short["response"][:end]
        short["response_token_ids"] = short["response_token_ids"][:end]
        short["response_token_offsets"] = short["response_token_offsets"][:end]
        short["items"] = short["items"][:1]
        truncated, _ = port.extract_lumina(tok, self.model, row, short, random_row, chunk_size=4)
        for key in ("lumina_mmd", "lumina_ipr", "lumina_score"):
            np.testing.assert_allclose(full[key][0], truncated[key][0], rtol=3e-5, atol=2e-7)
            np.testing.assert_allclose(full["token_"+key][:end], truncated["token_"+key], rtol=3e-5, atol=2e-7)

    def test_random_prompt_preserves_target_question_and_rejects_overlap(self):
        _, row, _, random_row = make_fixture()
        prompt = port.build_random_prompt(row, random_row)
        self.assertIn("Where is A?", prompt)
        self.assertNotIn("IGNORE OTHER QUESTION", prompt)
        self.assertNotIn("A is in Peru.", prompt)
        self.assertIn("C is a river.", prompt)
        with self.assertRaises(ValueError):
            port.build_random_prompt(row, row)
        bad = copy.deepcopy(row)
        bad["prompt"] += "\nOther text"
        with self.assertRaises(ValueError):
            port.build_random_prompt(bad, random_row)

    def test_unscorable_items_are_retained_and_prefix_changes_fail(self):
        tok, row, generated, random_row = make_fixture()
        generated["items"][1].update(parse_ok=False, parse_reason="missing_number")
        values, metadata = port.extract_lumina(tok, self.model, row, generated, random_row, chunk_size=4)
        self.assertEqual(len(values["item_ids"]), 2)
        self.assertTrue(np.isnan(values["lumina_score"][1]))
        self.assertFalse(metadata["items"][1]["score_available"])
        generated["input_token_ids"][0] += 1
        with self.assertRaises(ValueError):
            port.extract_lumina(tok, self.model, row, generated, random_row)

    @torch.inference_mode()
    def test_bfloat16_backbone_uses_finite_float32_formula_statistics(self):
        tok, row, generated, random_row = make_fixture()
        bf16_model = copy.deepcopy(self.model).to(dtype=torch.bfloat16).eval()
        values, metadata = port.extract_lumina(tok, bf16_model, row, generated, random_row, chunk_size=3)
        for name in ("token_lumina_mmd", "token_lumina_ipr", "token_lumina_score"):
            self.assertTrue(np.isfinite(values[name]).all())
        self.assertEqual(metadata["probability_and_reduction_dtype"], "float32")
        self.assertEqual(metadata["hidden_state_dtype"], "torch.bfloat16")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LuminaFormulaTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "passed" if result.wasSuccessful() else "failed",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "device": "CPU only", "model": "random 3-layer Qwen2, 128-token vocabulary; no downloaded weights",
              "official_commit": port.OFFICIAL_COMMIT, "official_code_sha256": port.OFFICIAL_SHA256,
              "port_code_sha256": hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
              "test_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": ["official dense kernel MMD versus exact algebra including unequal top-k masses",
                        "official all-layer IPR versus chunked accumulation",
                        "tiny Qwen end-to-end per-token and item score equivalence",
                        "full versus truncated answer causality", "chunk-size invariance",
                        "exact prompt/answer ID contract and parser-failure coverage",
                        "bfloat16 backbone with float32 formula arithmetic"],
              "not_validated": ["7B model GPU memory and timing", "NF4 numerical behavior", "task accuracy"]}
    (ROOT/"references/lumina_cpu_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
