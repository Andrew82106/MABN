"""CPU parity and failure-mode checks, not claims of detector accuracy."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import feature10 as f10


def visible(questions=None, passages=None):
    questions = questions or ["In what year was Alice Smith born?"]
    passages = passages or [
        {"title": "Alice Smith", "text": "Alice Smith won a writing award."},
        {"title": "Bob Jones", "text": "Bob Jones was born in 1975."}]
    prompt = ("Answer each numbered question in a short sentence using the search results."
              + "\n\nQuestions:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
              + "\n\nSearch results:\n" + "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)))
    return {"system": "You are a helpful assistant.", "prompt": prompt,
            "questions": questions, "passages": passages}


class Feature10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok = AutoTokenizer.from_pretrained(
            ROOT.parent / "models/Qwen2.5-7B-Instruct-bnb-4bit", local_files_only=True)
        torch.manual_seed(94)
        cfg = Qwen2Config(vocab_size=len(cls.tok), hidden_size=16, intermediate_size=32,
                          num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=1,
                          max_position_embeddings=4096, attention_dropout=0.0, use_sliding_window=False)
        cfg._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(cfg).eval().requires_grad_(False)

    def record(self, vis, text="1. Alice was born in 1975."):
        ids = self.tok.apply_chat_template(
            [{"role": "system", "content": vis["system"]}, {"role": "user", "content": vis["prompt"]}],
            tokenize=True, add_generation_prompt=True)
        response = self.tok.encode(text, add_special_tokens=False)
        return {"input_token_ids": ids, "response_token_ids": response,
                "response": self.tok.decode(response, clean_up_tokenization_spaces=False),
                "items": [{"text": "must not read future parsed item", "risk": 1}]}

    def test_hand_formula_and_uniform_token_prior(self):
        attention = torch.tensor([[[.1, .1, .3, .2, .3]]])
        body = torch.arange(3)
        assignment = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        prior = torch.tensor([[.8, .2]])
        similarity = torch.tensor([[1., 0., 1.]])
        out = f10.soft_alignment_from_attention(attention, body, assignment, prior, similarity,
                                                3, torch.tensor([4]), torch.tensor([True]))
        expected = [.14/(.14+.25), np.sqrt(.4*.8)+np.sqrt(.6*.2), .8, .75/1.75]
        np.testing.assert_allclose(out[0, 0].numpy(), expected, rtol=1e-6)
        uniform = torch.tensor([[2/3, 1/3]])
        result = f10.soft_alignment_from_attention(attention, body, assignment, uniform, similarity,
                                                   3, torch.tensor([4]), torch.tensor([True]))
        old = f10.lookback_from_attention(attention, body, 3, torch.tensor([4]))
        np.testing.assert_allclose(result[0, :, 0], old[0], rtol=1e-6)
        self.assertAlmostEqual(float(result[0, 0, 3]), .5, places=6)
        no_copy = f10.soft_alignment_from_attention(attention, body, assignment, prior, torch.zeros_like(similarity),
                                                   3, torch.tensor([4]), torch.tensor([True]))
        self.assertEqual(float(no_copy[0, 0, 3]), 0.)
        invalid = f10.soft_alignment_from_attention(attention, body, assignment, prior, similarity,
                                                    3, torch.tensor([4]), torch.tensor([False]))
        self.assertTrue(torch.equal(invalid, torch.zeros_like(invalid)))

    def test_soft_relevance_has_no_entity_gate_and_empty_query_is_uniform(self):
        # Query [1,0] is semantically like source [1,0], without any text labels.
        embedding = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        plan = {"question_plans": [{"content_token_indices": [0], "content_token_weights": [1.]}],
                "sentence_units": [{"representation_token_indices": [1]}, {"representation_token_indices": [2]}]}
        prior, scores = f10.semantic_sentence_priors(embedding, plan)
        self.assertGreater(float(prior[0, 0]), float(prior[0, 1]))
        self.assertTrue(torch.all(prior > 0))
        self.assertAlmostEqual(float(prior.sum()), 1., places=6)
        plan["question_plans"][0] = {"content_token_indices": [], "content_token_weights": []}
        empty, _ = f10.semantic_sentence_priors(embedding, plan)
        np.testing.assert_allclose(empty.numpy(), [[.5, .5]])

    def test_visible_layout_and_hidden_metadata_invariance(self):
        vis = visible()
        gen = self.record(vis)
        plan = f10.visible_layout(self.tok, vis, gen["input_token_ids"])
        polluted = copy.deepcopy(vis)
        polluted.update(subjects=["Secret Name"], aliases=["1975"], standard_answer="1980", condition="partial", risk=1)
        polluted["passages"][0].update(source_sent_ids=[0], label="supported", hidden_evidence="born in 1980")
        other = f10.visible_layout(self.tok, polluted, gen["input_token_ids"])
        self.assertEqual(plan, other)
        self.assertEqual(len(plan["sentence_units"]), 2)
        self.assertNotIn("entities", plan["question_plans"][0])
        self.assertEqual(set(plan["body_token_indices"]),
                         set(k for u in plan["sentence_units"] for k in u["token_indices"]))
        a, _ = f10.extract_features(self.tok, self.model, vis, gen)
        b, meta = f10.extract_features(self.tok, self.model, polluted, gen)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(meta["hidden_metadata_used"])
        self.assertFalse(meta["labels_or_detector_scores_read"])

    def test_prompt_mismatch_and_no_evidence_fail_explicitly(self):
        vis = visible()
        gen = self.record(vis)
        edit = copy.deepcopy(vis)
        edit["passages"][0]["text"] = "Secret evidence omitted from the real prompt."
        with self.assertRaisesRegex(ValueError, "Passage list"):
            f10.visible_layout(self.tok, edit, gen["input_token_ids"])
        edit = copy.deepcopy(vis)
        edit["questions"] = ["What was the secret answer?"]
        with self.assertRaisesRegex(ValueError, "Question list"):
            f10.visible_layout(self.tok, edit, gen["input_token_ids"])
        empty = visible(passages=[{"title": "Empty source", "text": ""}])
        with self.assertRaisesRegex(ValueError, "No tokenized evidence"):
            f10.visible_layout(self.tok, empty, self.record(empty)["input_token_ids"])

    def test_current_piece_surface_controls_do_not_complete_future_words(self):
        vis = visible(questions=["Who is Unfamiliarname?"], passages=[
            {"title": "Source", "text": "Unfamiliarname and Familiarname are different names."}])
        gen = self.record(vis, "1. Unfamiliarname was here. Familiarname was there.")
        plan = f10.visible_layout(self.tok, vis, gen["input_token_ids"])
        ids = gen["response_token_ids"]
        routes, valid = f10.causal_question_routes(self.tok, ids, 1)
        full = f10.surface_controls(self.tok, plan, gen["input_token_ids"], ids, routes, valid)
        # This includes prefixes that end within a multi-BPE proper name.
        for n in range(1, len(ids)+1):
            out = f10.surface_controls(self.tok, plan, gen["input_token_ids"], ids[:n], routes[:n], valid[:n])
            np.testing.assert_array_equal(out, full[:n])

    def test_tiny_qwen_eager_parity(self):
        vis = visible()
        gen = self.record(vis, "1. 1975.")
        arrays, meta = f10.extract_features(self.tok, self.model, vis, gen)
        n = len(gen["response_token_ids"])
        self.assertEqual(arrays["new_features"].shape, (n, 16))
        self.assertEqual(arrays["surface_features"].shape, (n, 8))
        self.assertEqual(arrays["lookback_features"].shape, (n, 56))
        self.assertEqual(len(meta["new_feature_names"]), 16)
        ids = torch.tensor([gen["input_token_ids"]+gen["response_token_ids"]])
        p = len(gen["input_token_ids"])
        pos = torch.arange(p, ids.shape[1])
        original = self.model.config._attn_implementation
        self.model.config._attn_implementation = "eager"
        try:
            with torch.inference_mode():
                out = self.model.model(input_ids=ids, use_cache=False, output_attentions=True, output_hidden_states=True)
        finally:
            self.model.config._attn_implementation = original
        plan = meta["layout"]
        body = torch.tensor(plan["body_token_indices"])
        context = torch.tensor(plan["context_token_indices"])
        assignment = torch.zeros(len(body), len(plan["sentence_units"]))
        index = {int(v): i for i, v in enumerate(body)}
        for si, u in enumerate(plan["sentence_units"]):
            assignment[[index[k] for k in u["token_indices"]], si] = 1
        with torch.inference_mode():
            embedding = torch.nn.functional.normalize(self.model.model.embed_tokens(ids)[0].float(), dim=-1)
            prior, _ = f10.semantic_sentence_priors(embedding[:p], plan)
            copy_sim = (embedding[p:] @ embedding[body].T).clamp(0, 1)
        new = np.zeros((n, 4, 4), dtype=np.float32)
        lb = []
        for layer, att in enumerate(out.attentions):
            attention = att[0, :, pos]
            lb.append(f10.lookback_from_attention(attention, context, p, pos).T.numpy())
            value = f10.soft_alignment_from_attention(attention, body, assignment, prior.expand(n, -1), copy_sim,
                                                       p, pos, torch.ones(n, dtype=torch.bool))
            new[:, layer//7] += value.mean(0).numpy()/7
        np.testing.assert_allclose(arrays["new_features"], new.reshape(n, -1), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(arrays["lookback_features"], np.stack(lb, 1).reshape(n, -1), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(arrays["hidden_28"], out.last_hidden_state[0, pos].numpy(), rtol=2e-5, atol=2e-6)
        with torch.inference_mode():
            lp = self.model.lm_head(out.last_hidden_state[0, pos-1]).float().log_softmax(-1)
            expected_nll = -lp.gather(1, ids[0, pos, None]).squeeze(-1)
        np.testing.assert_allclose(arrays["token_nll"], expected_nll.numpy(), atol=2e-5)

    def test_future_and_finished_item_metadata_do_not_change_current_features(self):
        vis = visible()
        gen = self.record(vis)
        old, _ = f10.extract_features(self.tok, self.model, vis, gen)
        cut = len(gen["response_token_ids"])//2
        edit = copy.deepcopy(gen)
        replacement = self.tok.encode(" yes", add_special_tokens=False)[0]
        edit["response_token_ids"][cut+1:] = [replacement]*(len(edit["response_token_ids"])-cut-1)
        edit["response"] = self.tok.decode(edit["response_token_ids"], clean_up_tokenization_spaces=False)
        edit["items"] = [{"risk": 0, "text": "Future parser/gold must never affect features"}]
        changed, _ = f10.extract_features(self.tok, self.model, vis, edit)
        for key in ("new_features", "surface_features", "lookback_features", "hidden_28", "token_nll", "token_entropy"):
            np.testing.assert_allclose(old[key][:cut+1], changed[key][:cut+1], rtol=2e-5, atol=2e-6, err_msg=key)
        short = copy.deepcopy(gen)
        short["response_token_ids"] = short["response_token_ids"][:cut+1]
        short["response"] = self.tok.decode(short["response_token_ids"], clean_up_tokenization_spaces=False)
        truncated, _ = f10.extract_features(self.tok, self.model, vis, short)
        for key in ("new_features", "surface_features", "lookback_features", "hidden_28"):
            np.testing.assert_allclose(old[key][:cut+1], truncated[key], rtol=2e-5, atol=2e-6, err_msg=key)


if __name__ == "__main__":
    unittest.main()
