"""CPU tests for immutable input binding, visible-only calls and resumable caches."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import run9 as r9


def make_visible(text):
    questions=["In what year was Alice Smith born?"]
    passages=[{"title":"Alice Smith","text":text}]
    return {"system":"You are a helpful assistant.","questions":questions,"passages":passages,
            "prompt":"Please answer the question using the search results.\n\nQuestions:\n1. "+questions[0]
                     +"\n\nSearch results:\n[1] Alice Smith\n"+text}


class Runner9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=AutoTokenizer.from_pretrained(r9.model7.MODEL,local_files_only=True)
        cls.pool=r9.readl(r9.R7/"data/dev_inputs.jsonl")
        cls.model=SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1))))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"src").mkdir()
        (self.root/"src/fixture.py").write_text("VALUE = 1\n",encoding="utf-8")
        (self.root/"PLAN.md").write_text("Predeclared CPU fixture.\n",encoding="utf-8")
        self.rows=[]
        for condition,text in (("complete","Alice Smith was born in 1980."),
                               ("partial","Alice Smith won a literature prize.")):
            self.rows.append({**make_visible(text),"row_id":"example__"+condition,"question_id":"example",
                              "group_id":"example","split":"train","condition":condition,"category":"time",
                              "dataset":"synthetic_control","subjects":["SECRET SUBJECT"],"aliases":["HIDDEN"],
                              "refs":{"gold":"1980"},"coverage":"HIDDEN","expected_items":1})
        r9.savel(self.root/"data/inputs.jsonl",self.rows)
        r9.save(self.root/"protocol.json",{"generation":{"max_new_tokens":256}})
        p=self.pool[0]["passages"][0]
        self.manifest={"schema":"round9-random-context-v1",
                       "pool_file":str((r9.R7/"data/dev_inputs.jsonl").resolve()),
                       "pool_sha256":r9.sha(r9.R7/"data/dev_inputs.jsonl"),
                       "assignments":{"example":{"donor_id":"fixed_example",
                                                "passages":[{"title":p["title"],"text":p["text"],
                                                             "origin_row_id":self.pool[0]["row_id"]}]}}}
        r9.save(self.root/r9.DEFAULT_RANDOM,self.manifest)
        r9.prepare_freeze(self.root)
        self.context=r9.preflight(self.root)

    def calls(self):
        counts={k:0 for k in ("load","generate","core","attention","lumina")}
        def forbidden(value):
            self.assertFalse(set(value)&{"condition","category","subjects","aliases","refs","coverage","split"})
        def loader():
            counts["load"]+=1
            return self.tok,self.model
        def generate(tok,model,row):
            counts["generate"]+=1
            self.assertEqual(row["condition"],"unlabelled")
            self.assertEqual(row["split"],"unlabelled")
            self.assertFalse(set(row)&{"category","subjects","aliases","refs","coverage"})
            text="1. Alice Smith was born in 1980."
            ids=tok.encode(text,add_special_tokens=False)
            offsets=r9.model7.token_offsets(tok,ids,text).tolist()
            items,parser=r9.model7.parse_items(text,row["row_id"],expected_items=1)
            return {"row_id":row["row_id"],"question_id":row["question_id"],"split":row["split"],
                    "condition":row["condition"],"response":text,"response_token_ids":ids,
                    "input_token_ids":r9.model7.chat_ids(tok,row["prompt"],row["system"]),
                    "response_token_offsets":offsets,"items":items,"parser":parser,
                    "unexpected_special_token_ids":[]}
        def core(tok,model,vis,gen):
            counts["core"]+=1;forbidden(vis);forbidden(gen)
            self.assertTrue(all(set(p)=={"title","text"} for p in vis["passages"]))
            n=len(gen["response_token_ids"])
            off=np.asarray(gen["response_token_offsets"],dtype=np.int32)
            arrays={"hidden_28":np.zeros((n,3584),np.float32),"lookback_features":np.ones((n,784),np.float32),
                    "binding_features":np.zeros((n,32),np.float32),"token_nll":np.zeros(n,np.float32),
                    "token_entropy":np.zeros(n,np.float32),"token_ids":np.asarray(gen["response_token_ids"]),
                    "token_start":off[:,0],"token_end":off[:,1]}
            return arrays,{"binding_feature_names":[f"feature_{i}" for i in range(32)],
                           "target_candidate_missing_questions":0,"route_invalid_tokens":0}
        def attention(tok,model,row,gen):
            counts["attention"]+=1;forbidden(row);forbidden(gen)
            n=len(gen["response_token_ids"])
            return {"token_redeep_ecs":np.zeros((n,784),np.float32),
                    "token_redeep_pks":np.zeros((n,28),np.float32),
                    "token_lookback":np.ones((n,784),np.float32),
                    "response_token_offsets":np.asarray(gen["response_token_offsets"],np.int32)},{}
        def lumina(tok,model,row,gen,donor):
            counts["lumina"]+=1;forbidden(row);forbidden(gen);forbidden(donor)
            self.assertEqual(donor,self.context["donors"]["example"])
            n=len(gen["response_token_ids"])
            return {"token_lumina_score":np.zeros(n,np.float32),
                    "response_token_offsets":np.asarray(gen["response_token_offsets"],np.int32)},{}
        stack=ExitStack()
        stack.enter_context(patch.object(r9,"generation_signature",return_value={"test_signature":"fixed"}))
        stack.enter_context(patch.object(r9,"stage_signature",side_effect=lambda s:{"test_signature":s}))
        stack.enter_context(patch.object(r9.model7,"generate_answer",side_effect=generate))
        stack.enter_context(patch.object(r9.binding9,"extract_binding_features",side_effect=core))
        stack.enter_context(patch.object(r9.attention7,"extract_attention",side_effect=attention))
        stack.enter_context(patch.object(r9.lumina7,"extract_lumina",side_effect=lumina))
        stack.enter_context(patch.object(r9,"load_tokenizer",return_value=self.tok))
        stack.enter_context(redirect_stdout(StringIO()))
        return stack,counts,loader

    def test_frozen_inputs_and_protocol_change_rejected(self):
        for key in ("input_path","protocol_path","random_path","freeze_path"):
            path=self.context[key]
            original=path.read_bytes()
            path.write_bytes(original+b" ")
            with self.assertRaises((ValueError,RuntimeError)):
                r9.unchanged(self.context)
            if key!="freeze_path":
                with self.assertRaises(ValueError):
                    r9.preflight(self.root)
            path.write_bytes(original)
        with self.assertRaises(FileExistsError):
            r9.prepare_freeze(self.root)

    def test_visible_questions_and_passages_must_match_prompt(self):
        bad=copy.deepcopy(self.rows[0]);bad["questions"][0]="What is the secret answer?"
        with self.assertRaisesRegex(ValueError,"questions"):
            r9.validate_visible_strings(bad)
        bad=copy.deepcopy(self.rows[0]);bad["passages"][0]["text"]="Hidden gold."
        with self.assertRaisesRegex(ValueError,"passage"):
            r9.validate_visible_strings(bad)

    def test_code_and_plan_are_frozen_and_model_section_checked(self):
        frozen=self.context["frozen"]
        self.assertIn("PLAN.md",frozen["files_sha256"])
        self.assertIn("src/fixture.py",frozen["files_sha256"])
        for name in ("PLAN.md","src/fixture.py"):
            path=self.root/name;before=path.read_bytes()
            path.write_bytes(before+b"\n# change")
            with self.assertRaisesRegex(RuntimeError,"Frozen document, code or data"):
                r9.unchanged(self.context)
            path.write_bytes(before)
        for key,value in (("max_new_tokens",8),("do_sample",True),("seed",1),("attn_implementation","eager")):
            with self.assertRaisesRegex(ValueError,"model\\."):
                r9.validate_protocol(self.root,{"model":{key:value}})
        r9.validate_protocol(self.root,{"model":{"name":r9.model7.MODEL.name,"do_sample":False,"seed":20260910}})
        (self.root/"src/new_module.py").write_text("VALUE = 2\n",encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError,"inventory"):
            r9.unchanged(self.context)

    def test_external_dependencies_are_fixed_to_real_source_root(self):
        external=self.context["frozen"]["external_source_sha256"]
        self.assertEqual(set(external),set(r9.EXTERNAL_SOURCES))
        for name,value in external.items():
            self.assertEqual(value,r9.sha((r9.ROOT/name).resolve()))
        changed=dict(external);changed[r9.EXTERNAL_SOURCES[0]]="0"*64
        with patch.object(r9,"external_source_hashes",return_value=changed):
            with self.assertRaisesRegex(ValueError,"External frozen"):
                r9.preflight(self.root)
            with self.assertRaisesRegex(RuntimeError,"External frozen"):
                r9.unchanged(self.context)

    def test_random_assignment_origin_and_pair_fixed(self):
        donor=self.context["donors"]["example"]
        self.assertEqual(len(donor["passages"]),1)
        self.assertEqual(set(donor),{"row_id","question_id","passages"})
        bad=copy.deepcopy(self.manifest)
        bad["assignments"]["example"]["passages"][0]["text"]+=" Fabricated."
        r9.save(self.root/"bad_manifest.json",bad)
        with self.assertRaisesRegex(ValueError,"identical"):
            r9.validate_random_manifest(self.root,self.root/"bad_manifest.json",self.rows)
        bad=copy.deepcopy(self.manifest)
        bad["assignments"]["example"]["passages"][0]["origin_row_id"]="not_a_dev_row"
        r9.save(self.root/"bad_manifest.json",bad)
        with self.assertRaisesRegex(ValueError,"fixed development"):
            r9.validate_random_manifest(self.root,self.root/"bad_manifest.json",self.rows)
        rows=copy.deepcopy(self.rows);rows[1]["passages"].append(rows[1]["passages"][0])
        with self.assertRaisesRegex(ValueError,"different source counts"):
            r9.validate_random_manifest(self.root,self.root/r9.DEFAULT_RANDOM,rows)

    def test_all_stages_resume_without_loading_and_audit(self):
        stack,counts,loader=self.calls()
        with stack:
            r9.run(self.context,["generate"],limit=1,loader=loader)
            aggregate=r9.readl(self.root/"data/generated.jsonl")
            self.assertEqual(len(aggregate),1)
            self.assertEqual(aggregate[0]["condition"],"complete")
            r9.run(self.context,["generate","core","attention","lumina"],loader=loader)
            self.assertEqual(counts,{"load":2,"generate":2,"core":2,"attention":2,"lumina":2})
            before={p:r9.sha(p) for p in (self.root/"data/generation_records").glob("*.json")}
            fail_loader=lambda: self.fail("Cached stages must not load a model")
            r9.run(self.context,["generate","core","attention","lumina"],loader=fail_loader)
            report=r9.audit(self.context,require_complete=True)
            self.assertTrue(report["complete"]);self.assertFalse(report["model_loaded"])
            self.assertEqual(report["stages"],{"core":2,"attention":2,"lumina":2})
            self.assertEqual(before,{p:r9.sha(p) for p in before})
            # Exercise the real downstream loader, without creating or reading gold.
            from evaluate9 import Bank
            records={g["row_id"]:(g,r9.sha(self.root/"data/generation_records"/(g["row_id"]+".json")))
                     for g in r9.readl(self.root/"data/generated.jsonl")}
            bank=Bank(self.root,records)
            for rid,(generated,_) in records.items():
                data=bank.row(rid)
                self.assertEqual(data["bound"].shape,(len(generated["response_token_ids"]),816))
            for row in self.rows:
                gp=self.root/"data/generation_records"/(row["row_id"]+".json")
                for folder in r9.FEATURE_FOLDERS.values():
                    side=json.loads((self.root/"data"/folder/(row["row_id"]+".json")).read_text())
                    self.assertEqual(side["source_generation_sha256"],r9.sha(gp))
                    self.assertEqual(side["arrays_sha256"],r9.sha(self.root/"data"/folder/(row["row_id"]+".npz")))
                    self.assertFalse(side["actual_condition_passed_to_extractor"])

    def test_missing_generation_and_stale_cache_rejected_before_loader(self):
        stack,counts,loader=self.calls()
        with stack:
            with self.assertRaisesRegex(RuntimeError,"Generate"):
                r9.run(self.context,["core"],loader=loader)
            self.assertEqual(counts["load"],0)
            r9.run(self.context,["generate","core"],limit=1,loader=loader)
            rid=self.rows[0]["row_id"]
            npz=self.root/"data/features"/(rid+".npz")
            npz.write_bytes(npz.read_bytes()+b"tamper")
            loaded=counts["load"]
            with self.assertRaisesRegex(RuntimeError,"Committed array cache"):
                r9.run(self.context,["core"],limit=1,loader=loader)
            self.assertEqual(counts["load"],loaded)

    def test_generation_change_breaks_feature_source_binding(self):
        stack,counts,loader=self.calls()
        with stack:
            r9.run(self.context,["generate","core"],limit=1,loader=loader)
            gp=self.root/"data/generation_records"/(self.rows[0]["row_id"]+".json")
            g=json.loads(gp.read_text());g["seconds"]+=1
            r9.save(gp,g)
            loaded=counts["load"]
            with self.assertRaisesRegex(RuntimeError,"source_generation_sha256"):
                r9.run(self.context,["core"],limit=1,loader=loader)
            self.assertEqual(counts["load"],loaded)

    def test_changed_snapshot_rejected_before_loader(self):
        stack,counts,loader=self.calls()
        with stack:
            path=self.context["protocol_path"];path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaisesRegex(RuntimeError,"Frozen material changed"):
                r9.run(self.context,["generate"],loader=loader)
            self.assertEqual(counts["load"],0)

    def test_safe_artifact_paths_and_row_ids(self):
        with self.assertRaises(ValueError):
            r9.inside(self.root,"../outside.json")
        with self.assertRaises(ValueError):
            r9.safe_identity({"row_id":"../outside"})

    def test_uncommitted_npz_is_recomputed(self):
        stack,counts,loader=self.calls()
        with stack:
            r9.run(self.context,["generate"],limit=1,loader=loader)
            path=self.root/"data/features"/(self.rows[0]["row_id"]+".npz")
            path.parent.mkdir(exist_ok=True);path.write_bytes(b"uncommitted")
            r9.run(self.context,["core"],limit=1,loader=loader)
            self.assertEqual(counts["core"],1)
            with np.load(path) as a:
                self.assertEqual(a["binding_features"].shape[1],32)

    def test_signature_records_wrappers_and_model(self):
        sig=r9.generation_signature()
        self.assertEqual(sig["model_config_sha256"],r9.sha(r9.model7.MODEL/"config.json"))
        self.assertEqual(len(sig["generator_functions_sha256"]),64)
        self.assertIn("binding9.py",r9.stage_signature("core")["code_hashes"])


if __name__=="__main__":
    unittest.main()
