"""CPU-only mathematical and tiny-Qwen checks; never load 7B weights."""
import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attention7 import (causal_attention_rows, external_context_score, extract_attention,
                        lookback_from_attention, standard_jsd_from_logits,
                        visible_context_positions, _response_offsets)


class FormulaTests(unittest.TestCase):
    def test_lookback_normalizes_each_side_by_length(self):
        weights = torch.tensor([[[0.1, 0.1, 0.8]]])
        value = lookback_from_attention(weights, torch.tensor([0, 1]), 2, torch.tensor([2]))
        self.assertAlmostEqual(value.item(), 1 / 9, places=6)
        self.assertNotAlmostEqual(value.item(), 0.2, places=4)

    def test_standard_jsd_symmetry_bounds_identity(self):
        p, q = torch.tensor([[100.0, -100.0]]), torch.tensor([[-100.0, 100.0]])
        self.assertAlmostEqual(standard_jsd_from_logits(p, q).item(), math.log(2), places=6)
        self.assertEqual(standard_jsd_from_logits(p, p).item(), 0.0)
        torch.testing.assert_close(standard_jsd_from_logits(p, q), standard_jsd_from_logits(q, p))

    def test_ecs_pools_only_top_attended_source(self):
        weights = torch.tensor([[[0.05, 0.25, 0.2, 0.5]]])
        context = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        current = torch.tensor([[0.0, 1.0]])
        # Most attended token overall is generated; source-only top1 is source1.
        value = external_context_score(weights, torch.tensor([0, 1, 2]), context, current)
        self.assertAlmostEqual(value.item(), 1.0, places=6)

    def test_causal_attention_never_reads_future(self):
        torch.manual_seed(15)
        q, k = torch.randn(2, 2, 4), torch.randn(2, 6, 4)
        result = causal_attention_rows(q, k, torch.tensor([2, 4]), 0.5)
        self.assertEqual(float(result[:, 0, 3:].abs().sum()), 0.0)
        self.assertEqual(float(result[:, 1, 5:].abs().sum()), 0.0)
        torch.testing.assert_close(result.sum(-1), torch.ones(2, 2))
        changed = k.clone()
        changed[:, 5] = 1e5
        torch.testing.assert_close(result, causal_attention_rows(q, changed, torch.tensor([2, 4]), 0.5))


class TinyQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(10)
        cls.tok = AutoTokenizer.from_pretrained(ROOT.parent / "models/Qwen2.5-7B-Instruct-bnb-4bit",
                                               local_files_only=True)
        cfg = Qwen2Config(vocab_size=len(cls.tok), hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=2048, attention_dropout=0.0,
                          use_sliding_window=False, sliding_window=None)
        cfg._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.row = {
            "row_id": "tiny", "system": "You are a helpful assistant.",
            "prompt": "Please answer these questions.\n\nQuestions:\n1. When was Alpha founded?\n2. When was Beta founded?\n\nSearch results:\n[1] Alpha\nAlpha was founded in 1900.\n\n[2] Beta\nBeta was founded in 2000.",
            "passages": [{"title": "Alpha", "text": "Alpha was founded in 1900."},
                         {"title": "Beta", "text": "Beta was founded in 2000."}],
        }
        cls.generated = cls.make_generated("1. Alpha was founded in 1900.\n2. Beta was founded in 2000.")

    @classmethod
    def make_generated(cls, response):
        prefix = cls.tok.apply_chat_template(
            [{"role": "system", "content": cls.row["system"]},
             {"role": "user", "content": cls.row["prompt"]}], tokenize=True, add_generation_prompt=True)
        response_ids = cls.tok.encode(response, add_special_tokens=False)
        lines = response.splitlines(keepends=True)
        items, cursor = [], 0
        for i, line in enumerate(lines):
            start, end = cursor + 3, cursor + len(line.rstrip())
            items.append({"item_id": f"tiny__{i+1}", "item_index": i + 1, "start": start,
                          "end": end, "text": response[start:end], "parse_ok": True})
            cursor += len(line)
        return {"input_token_ids": prefix, "response_token_ids": response_ids,
                "response": response, "items": items,
                "response_token_offsets": _response_offsets(cls.tok, response_ids, response).tolist()}

    def test_real_template_alignment_excludes_questions_and_chat_tokens(self):
        pos, meta = visible_context_positions(self.tok, self.row, self.generated["input_token_ids"])
        decoded = self.tok.decode([self.generated["input_token_ids"][i] for i in pos])
        self.assertIn("Alpha was founded in 1900.", decoded)
        self.assertNotIn("When was Alpha", decoded)
        self.assertNotIn("assistant", decoded)
        bad = copy.deepcopy(self.generated["input_token_ids"])
        bad[-1] = 5
        with self.assertRaises(ValueError):
            visible_context_positions(self.tok, self.row, bad)

    def test_full_extraction_matches_eager_reference_and_item_mean(self):
        values, metadata = extract_attention(self.tok, self.model, self.row, self.generated)
        prefix = self.generated["input_token_ids"]
        ids = torch.tensor([prefix + self.generated["response_token_ids"]])
        context, _ = visible_context_positions(self.tok, self.row, prefix)
        context = torch.tensor(context)
        positions = torch.arange(len(prefix), ids.shape[1])
        captures, handles = {}, []
        for layer, block in enumerate(self.model.model.layers):
            def pre_hook(module, args, i=layer):
                captures[(i, "before")] = args[0][0].detach().clone()
            def post_hook(module, args, output, i=layer):
                captures[(i, "after")] = output[0][0].detach().clone()
            handles += [block.post_attention_layernorm.register_forward_pre_hook(pre_hook),
                        block.register_forward_hook(post_hook)]
        try:
            with torch.inference_mode():
                result = self.model.model(ids, use_cache=False, output_attentions=True)
                for layer in range(2):
                    attn = result.attentions[layer][0].index_select(1, positions)
                    expected_lb = lookback_from_attention(attn, context, len(prefix), positions)
                    expected_ecs = external_context_score(attn, context,
                        result.last_hidden_state[0].index_select(0, context),
                        result.last_hidden_state[0].index_select(0, positions))
                    expected_pks = standard_jsd_from_logits(
                        self.model.lm_head(self.model.model.norm(captures[(layer, "before")][positions])),
                        self.model.lm_head(self.model.model.norm(captures[(layer, "after")][positions])))
                    np.testing.assert_allclose(values["token_lookback"][:, layer*4:(layer+1)*4],
                                               expected_lb.T.numpy(), atol=1e-6, rtol=1e-5)
                    np.testing.assert_allclose(values["token_redeep_ecs"][:, layer*4:(layer+1)*4],
                                               expected_ecs.T.numpy(), atol=1e-5, rtol=1e-5)
                    np.testing.assert_allclose(values["token_redeep_pks"][:, layer],
                                               expected_pks.numpy(), atol=1e-6, rtol=1e-4)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(values["lookback_features"].shape, (2, 8))
        self.assertEqual(values["redeep_pks"].shape, (2, 2))
        for item, indices in enumerate([m["response_token_indices"] for m in metadata["items"]]):
            np.testing.assert_allclose(values["redeep_ecs"][item], values["token_redeep_ecs"][indices].mean(0))
        self.assertEqual(self.model.config._attn_implementation, "sdpa")
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in self.model.modules()))

    def test_features_unchanged_by_later_item_and_hidden_labels(self):
        original, _ = extract_attention(self.tok, self.model, self.row, self.generated)
        changed = self.make_generated("1. Alpha was founded in 1900.\n2. Beta does not exist and 777777 aliens won.")
        row = dict(self.row, gold_answer="SECRET", condition="partial", split="test", risk=1)
        other, _ = extract_attention(self.tok, self.model, row, changed)
        for key in ("lookback_features", "redeep_ecs", "redeep_pks"):
            np.testing.assert_allclose(original[key][0], other[key][0], atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
