"""Meaningful CPU checks: candidate controls, causal routes, tensor formulas and Qwen parity."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import binding9 as b9


def visible(question, passages, questions=None):
    qs = questions or [question]
    prompt = ("Please answer the following questions using these search results. "
              "Write one short sentence for each numbered item.\n\nQuestions:\n"
              + "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
              + "\n\nSearch results:\n"
              + "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(passages)))
    return {"system": "You are a helpful assistant.", "prompt": prompt,
            "questions": qs, "passages": passages}


class Binding9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok = AutoTokenizer.from_pretrained(
            ROOT.parent / "models/Qwen2.5-7B-Instruct-bnb-4bit", local_files_only=True)
        torch.manual_seed(92)
        config = Qwen2Config(
            vocab_size=len(cls.tok), hidden_size=16, intermediate_size=32,
            num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=4096, attention_dropout=0.0,
            use_sliding_window=False)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).eval()
        cls.model.requires_grad_(False)

    def prefix(self, vis):
        return self.tok.apply_chat_template(
            [{"role":"system","content":vis["system"]},
             {"role":"user","content":vis["prompt"]}],
            tokenize=True, add_generation_prompt=True)

    def record(self, vis, text):
        answer = self.tok.encode(text, add_special_tokens=False)
        return {"input_token_ids":self.prefix(vis), "response_token_ids":answer,
                "response":self.tok.decode(answer, clean_up_tokenization_spaces=False),
                "items":[{"text":"future metadata must be ignored", "parse_ok":False}]}

    def plan(self, vis):
        return b9.build_candidate_plan(self.tok, vis, self.prefix(vis))

    def test_visible_anchor_attribute_and_other_entity(self):
        vis = visible("In what year was Alice Smith born?", [
            {"title":"Alice Smith","text":"Alice Smith was born in 1980."},
            {"title":"Bob Jones","text":"Bob Jones was born in 1975."},
            {"title":"Alice Smith awards","text":"Alice Smith won a writing prize."},
        ])
        plan = self.plan(vis)
        p = plan["question_plans"][0]
        self.assertEqual(p["entities"], ["Alice Smith"])
        self.assertTrue(p["target_union_token_indices"])
        self.assertTrue(p["entity_only_token_indices"])
        self.assertTrue(p["attribute_distractor_token_indices"])
        sets = [set(p[k]) for k in ("target_union_token_indices","entity_only_token_indices",
                                    "attribute_distractor_token_indices","other_body_token_indices")]
        self.assertEqual(set.union(*sets), set(plan["body_token_indices"]))
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
        for u in plan["sentence_units"]:
            expected = "target_union_token_indices" if u["passage_index"]==0 else ("attribute_distractor_token_indices" if u["passage_index"]==1 else "entity_only_token_indices")
            self.assertTrue(set(u["token_indices"]) <= set(p[expected]))

    def test_present_entity_wrong_attribute_is_not_candidate(self):
        vis = visible("In what year was Alice Smith born?", [
            {"title":"Alice Smith","text":"Alice Smith won a writing prize."},
            {"title":"Bob Jones","text":"Bob Jones was born in 1975."},
        ])
        p = self.plan(vis)["question_plans"][0]
        self.assertFalse(p["target_union_token_indices"])
        self.assertTrue(p["entity_only_token_indices"])
        self.assertTrue(p["attribute_distractor_token_indices"])
        absent = visible("In what year was Charlie Green born?",vis["passages"])
        a = self.plan(absent)["question_plans"][0]
        self.assertFalse(a["target_union_token_indices"])
        self.assertFalse(a["entity_only_token_indices"])

    def test_no_supplied_alias_reference_or_condition_access(self):
        vis = visible("Who LG AI Research collaborated with?",[
            {"title":"Manus (AI agent)","text":"Manus uses multiple models."},
            {"title":"Google Cloud","text":"Google Cloud partnered with Someone Else."},
        ])
        polluted = copy.deepcopy(vis)
        polluted.update(subjects=["Google Cloud"], aliases=["Manus"], condition="complete",
                        reference_evidence=["Manus"], risk=1, coverage=True)
        polluted["passages"][0].update(source_sent_ids=[7], label="supported", aliases=["LG AI Research"])
        self.assertEqual(self.plan(vis),self.plan(polluted))
        p = self.plan(vis)["question_plans"][0]
        self.assertEqual(p["entities"],["LG AI Research"])
        self.assertFalse(p["target_union_token_indices"])
        self.assertFalse(p["entity_only_token_indices"])

    def test_other_person_lifespan_not_assigned_to_title_subject(self):
        vis = visible("In what year did Eduardo Schilling die?",[
            {"title":"Eduardo Schilling","text":"Eduardo Schilling married Alice Smith, whose daughter Maria (1921–1940) became a writer."}
        ])
        p = self.plan(vis)["question_plans"][0]
        self.assertTrue(p["entity_only_token_indices"])
        self.assertFalse(p["target_union_token_indices"])
        self.assertFalse(b9._entity_grade("Bob Jones was born in 1975.","Alice Smith",["Alice Smith"],True)[0])
        self.assertFalse(b9._entity_grade("Jane Smith was born in 1975.","John Smith",["John Smith"],True)[0])
        self.assertFalse(b9._entity_grade("Research was founded in 1990.","Research",["LG AI Research"],True)[0])

    def test_time_property_not_assigned_across_explicit_subject_change(self):
        vis=visible("In what year was John Smith born?",[
            {"title":"John Smith","text":"John Smith met Jane Jones, who was born in 1975."}])
        p=self.plan(vis)["question_plans"][0]
        self.assertFalse(p["target_union_token_indices"])
        self.assertTrue(p["entity_only_token_indices"])
        actual=visible("In what year was Richard Llewellyn born?",[
            {"title":"Richard Llewellyn","text":"Richard Dafydd Vivian Llewellyn Lloyd (8 December 1906 – 30 November 1983), known by his pen name Richard Llewellyn, was a British novelist."}])
        self.assertTrue(self.plan(actual)["question_plans"][0]["target_union_token_indices"])

    def test_comparison_has_independent_subject_masks(self):
        vis = visible("",[
            {"title":"Alice Smith","text":"Alice Smith was born in 1980."},
            {"title":"Bob Jones","text":"Bob Jones won an award."},
        ],questions=["In what year was Alice Smith born?","In what year was Bob Jones born?",
                     "How do the birth years of Alice Smith and Bob Jones compare?"])
        p = self.plan(vis)["question_plans"][2]
        self.assertEqual(p["entities"],["Alice Smith","Bob Jones"])
        self.assertEqual(len(p["target_token_indices_by_entity"]),2)
        self.assertTrue(p["target_token_indices_by_entity"][0])
        self.assertFalse(p["target_token_indices_by_entity"][1])

    def test_incremental_routes_do_not_look_at_future_numbering(self):
        ids = self.tok.encode("1. Alice is correct.\n2. Bob is here.\n1. Repeated.",add_special_tokens=False)
        routes, valid = b9.causal_question_routes(self.tok,ids,2)
        for n in range(1,len(ids)+1):
            short, sv = b9.causal_question_routes(self.tok,ids[:n],2)
            np.testing.assert_array_equal(short,routes[:n])
            np.testing.assert_array_equal(sv,valid[:n])
        self.assertIn(0,routes)
        self.assertIn(1,routes)
        self.assertEqual(routes[-1],-1)
        single, sv = b9.causal_question_routes(self.tok,ids,1)
        np.testing.assert_array_equal(single,np.zeros(len(ids),dtype=np.int32))
        self.assertTrue(sv.all())

    def test_attention_formulas_distinguish_density_and_mass(self):
        # T={0}, E={1}, A={2,3}; previous response={4}; current self={5}.
        a = torch.tensor([[[.2,.1,.1,.1,.3,.2]]],dtype=torch.float32)
        masks = torch.zeros(1,5,6)
        masks[0,0,0]=1; masks[0,1,1]=1; masks[0,2,2:4]=1
        masks[0,3,0]=1; masks[0,4,1]=1
        f, valid = b9.binding_from_attention(a,masks,torch.arange(4),4,torch.tensor([5]),
                                             torch.tensor([True]),torch.tensor([True]))
        np.testing.assert_allclose(f.numpy()[0,0],[.5,.4,2/3,.4,.2,.4,.2,.5],rtol=1e-6)
        self.assertTrue(valid.all())
        no = masks.clone();no[:,0]=0;no[:,3]=0
        f, valid = b9.binding_from_attention(a,no,torch.arange(4),4,torch.tensor([5]),
                                             torch.tensor([True]),torch.tensor([True]))
        self.assertFalse(valid[0,1]);self.assertFalse(valid[0,2]);self.assertFalse(valid[0,7])
        self.assertTrue(torch.isfinite(f).all())
        first = a[...,:5].clone();first/=first.sum(-1,keepdim=True)
        _, valid = b9.binding_from_attention(first,masks[...,:5],torch.arange(4),4,torch.tensor([4]),
                                             torch.tensor([True]),torch.tensor([True]))
        self.assertFalse(valid[0,3])

    def test_exact_prompt_alignment_rejects_hidden_passage_edits(self):
        vis = visible("In what year was Alice Smith born?",[{"title":"Alice Smith","text":"Alice Smith was born in 1980."}])
        prefix = self.prefix(vis)
        changed = copy.deepcopy(vis);changed["passages"][0]["text"]="Alice Smith was born in 2000."
        with self.assertRaisesRegex(ValueError,"source body"):
            b9.build_candidate_plan(self.tok,changed,prefix)
        changed=copy.deepcopy(vis);changed["questions"]=["In what year was Hidden Answer born?"]
        with self.assertRaisesRegex(ValueError,"Question list"):
            b9.build_candidate_plan(self.tok,changed,prefix)
        prefix[-1]=(prefix[-1]+1)%len(self.tok)
        with self.assertRaisesRegex(ValueError,"Input IDs"):
            b9.build_candidate_plan(self.tok,vis,prefix)

    def test_tiny_qwen_eager_attention_hidden_and_probability_alignment(self):
        vis = visible("In what year was Alice Smith born?",[
            {"title":"Alice Smith","text":"Alice Smith was born in 1980."},
            {"title":"Bob Jones","text":"Bob Jones was born in 1975."}])
        gen = self.record(vis,"1. 1980.")
        arrays, meta = b9.extract_binding_features(self.tok,self.model,vis,gen)
        self.assertEqual(arrays["binding_features"].shape,(len(gen["response_token_ids"]),32))
        self.assertEqual(arrays["lookback_features"].shape,(len(gen["response_token_ids"]),56))
        ids = torch.tensor([gen["input_token_ids"]+gen["response_token_ids"]])
        p = len(gen["input_token_ids"]);pos=torch.arange(p,ids.shape[1])
        old = self.model.config._attn_implementation
        self.model.config._attn_implementation="eager"
        try:
            with torch.inference_mode():
                output=self.model.model(input_ids=ids,use_cache=False,output_attentions=True,output_hidden_states=True)
        finally:
            self.model.config._attn_implementation=old
        context=torch.tensor(meta["candidate_plan"]["context_token_indices"])
        expected=[]
        for attn in output.attentions:
            expected.append(b9.lookback_from_attention(attn[0,:,pos],context,p,pos).T.numpy())
        expected=np.stack(expected,axis=1).reshape(len(pos),-1)
        np.testing.assert_allclose(arrays["lookback_features"],expected,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(arrays["hidden_28"],output.last_hidden_state[0,pos].numpy(),rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(arrays["hidden_21"],output.hidden_states[21][0,pos].numpy(),rtol=2e-5,atol=2e-6)
        with torch.inference_mode():
            lp=self.model.lm_head(output.last_hidden_state[0,pos-1]).float().log_softmax(-1)
            nll=-lp.gather(1,ids[0,pos,None]).squeeze(-1)
            ent=-(lp.exp()*lp).sum(-1)
        np.testing.assert_allclose(arrays["token_nll"],nll.numpy(),atol=2e-5)
        np.testing.assert_allclose(arrays["token_entropy"],ent.numpy(),atol=2e-5)
        self.assertFalse(meta["labels_or_detector_scores_read"])
        self.assertEqual(meta["causal_passes"],1)

    def test_full_future_and_metadata_cannot_change_current_features(self):
        vis = visible("In what year was Alice Smith born?",[
            {"title":"Alice Smith","text":"Alice Smith was born in 1980."},
            {"title":"Bob Jones","text":"Bob Jones was born in 1975."}])
        gen=self.record(vis,"1. 1980. Bob.")
        original,_=b9.extract_binding_features(self.tok,self.model,vis,gen)
        # Same-shape perturbation, preserving byte-round-trippable ordinary tokens.
        altered=copy.deepcopy(gen)
        k=len(gen["response_token_ids"])//2
        repl=self.tok.encode(" yes",add_special_tokens=False)[0]
        altered["response_token_ids"][k+1:]=[repl]*(len(gen["response_token_ids"])-k-1)
        altered["response"]=self.tok.decode(altered["response_token_ids"],clean_up_tokenization_spaces=False)
        altered.update(items=[{"text":"different future, with gold label", "risk":1}],condition="different")
        new,_=b9.extract_binding_features(self.tok,self.model,vis,altered)
        for name in ("hidden_21","hidden_28","lookback_features","binding_features","token_nll","token_entropy"):
            np.testing.assert_allclose(original[name][:k+1],new[name][:k+1],rtol=2e-5,atol=2e-6,err_msg=name)
        short=copy.deepcopy(gen);short["response_token_ids"]=gen["response_token_ids"][:k+1]
        short["response"]=self.tok.decode(short["response_token_ids"],clean_up_tokenization_spaces=False)
        truncated,_=b9.extract_binding_features(self.tok,self.model,vis,short)
        for name in ("hidden_28","lookback_features","binding_features"):
            np.testing.assert_allclose(original[name][:k+1],truncated[name],rtol=2e-5,atol=2e-6,err_msg=name)


if __name__ == "__main__":
    unittest.main()
