"""Build Round7 inputs from source-only, explicitly reviewed curation records.

No generation, scoring, or inferred output labels occur here. All text snippets
are complete original source sentences; both complete and partial conditions
use the same extraction rule. Full upstream rows remain in data/raw/hotpotqa.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260910
SYSTEM = "You are a helpful assistant."
TEMPLATE = "Please answer the following questions using these search results. Write one short sentence for each numbered item.\n\nQuestions:\n{questions}\n\nSearch results:\n{search_results}"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def raw_rows():
    rows={}
    for p in sorted((ROOT/"data/raw/hotpotqa").glob("train_[0-9]*.json")):
        if p.name.endswith(".meta.json"):continue
        d=load(p)
        assert not d.get("partial") and not any(r.get("truncated_cells") for r in d["rows"])
        for r in d["rows"]:rows[r["row_idx"]]=r["row"]
    return rows


def assign_splits(specs):
    """Freeze 120/40/40 with removal-side and broad attribute stratification."""
    from sklearn.model_selection import train_test_split
    assert len(specs)==200
    def stratum(s):
        attr=s["attribute"].casefold()
        category="temporal" if any(w in attr for w in ["year","date"]) else "non_temporal"
        return category+"_removed_"+str(s["removed_subject_index"])
    strata=[stratum(s) for s in specs]
    indices=list(range(len(specs)))
    train,held=train_test_split(indices,train_size=120,random_state=SEED,stratify=strata)
    valid,test=train_test_split(held,train_size=40,random_state=SEED+1,stratify=[strata[i] for i in held])
    for name,ids in [("train",train),("validation",valid),("test",test)]:
        for i in ids:specs[i]["split"]=name
    return {"seed_main_split":SEED,"seed_validation_test_split":SEED+1,"stratification":"temporal (attribute includes year/date) versus non_temporal, crossed with removed_subject_index","strata":dict(Counter(strata)),"split_counts":dict(Counter(s["split"] for s in specs))}


def build(specs):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT.parent/"models/Qwen2.5-7B-Instruct-bnb-4bit",local_files_only=True)
    raw=raw_rows()
    old=ROOT.parent/"round6_evidence_grounding"
    old_rows=[json.loads(x) for x in (old/"data/inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    preview=load(old/"planning/hotpotqa_preview.json")["rows"]
    excluded_titles={p["title"].casefold() for r in old_rows for p in r["passages"]}|{t.casefold() for r in preview for t in r["context"]["title"]}
    excluded_ids={r.get("source_question_id",r["question_id"]).removeprefix("hotpot_") for r in old_rows}|{r["id"] for r in preview}
    excluded_subjects={s.casefold() for r in old_rows for s in r["subjects"]}
    seen_titles=set();seen_subjects=set();seen_texts=set();inputs=[];refs=[];inventory=[]
    for spec in specs:
        source=raw[spec["source_index"]]
        qid="hotpot_"+source["id"]
        assert source["id"] not in excluded_ids
        def passage(desc):
            row=raw[desc.get("source_index",spec["source_index"])]
            texts=dict(zip(row["context"]["title"],row["context"]["sentences"]))
            ss=[texts[desc["title"]][i] for i in desc["sent_ids"]]
            return {"title":desc.get("display_title",desc["title"]),"text":"".join(ss).strip(),"sentences":ss,"source_title":desc["title"],"source_sent_ids":desc["sent_ids"],"source_question_id":row["id"]}
        keys=[passage({"title":t,"sent_ids":ids}) for t,ids in zip(spec["titles"],spec["evidence_sent_ids"])]
        backgrounds=[passage(x) for x in spec["background"]]
        replacement=passage(spec["replacement"])
        allp=keys+backgrounds+[replacement]
        titles={p["source_title"].casefold() for p in allp}
        subjects={s.casefold() for s in spec["subjects"]}
        text_keys={" ".join(p["text"].split()).casefold() for p in allp}
        assert not titles&excluded_titles,(qid,"old title",titles&excluded_titles)
        assert not subjects&excluded_subjects,(qid,"old subject",subjects&excluded_subjects)
        assert not titles&seen_titles,(qid,"shared title",titles&seen_titles)
        assert not subjects&seen_subjects,(qid,"shared subject",subjects&seen_subjects)
        assert not text_keys&seen_texts,(qid,"shared displayed text",text_keys&seen_texts)
        seen_titles|=titles;seen_subjects|=subjects
        seen_texts|=text_keys
        removed=spec["removed_subject_index"]
        a=len(tok.encode(keys[removed]["text"],add_special_tokens=False));b=len(tok.encode(replacement["text"],add_special_tokens=False))
        assert abs(a-b)<=max(20,.2*a),(qid,a,b,"length mismatch")
        order=list(range(4));random.Random(SEED+int(hashlib.sha256(qid.encode()).hexdigest()[:8],16)).shuffle(order)
        for condition in ["complete","partial"]:
            pp=list(keys)+backgrounds
            if condition=="partial":pp[removed]=replacement
            pp=[pp[i] for i in order]
            prompt=TEMPLATE.format(questions="\n".join(f"{i+1}. {q}" for i,q in enumerate(spec["questions"])),search_results="\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(pp)))
            nt=len(tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True))
            assert nt<=3072
            inputs.append({"row_id":qid+"__"+condition,"question_id":qid,"group_id":qid,"source_question_id":source["id"],"split":spec["split"],"condition":condition,"dataset":"HotpotQA","expected_items":3,"questions":spec["questions"],"subjects":spec["subjects"],"passages":pp,"prompt":prompt,"system":SYSTEM})
        evidence=[]
        for p in keys:
            evidence.append([{"title":p["title"],"source_title":p["source_title"],"sent_id":sid,"displayed_sent_id":j,"text":sentence} for j,(sid,sentence) in enumerate(zip(p["source_sent_ids"],p["sentences"]))])
        items=[]
        answers=spec["values"]+[spec["comparison_answer"]]
        for i in range(3):
            items.append({"item_index":i+1,"reference_answer":answers[i],"aliases":spec["aliases"][i],"evidence":evidence[i] if i<2 else evidence[0]+evidence[1],"rationale":spec["rationales"][i]})
            caveats=spec.get("item_annotation_caveats",[None,None,None])
            if caveats[i]:items[-1]["annotation_caveat"]=caveats[i]
        coverage=[True,True,False];coverage[removed]=False
        refs.append({"question_id":qid,"source_row_id":source["id"],"source_index":spec["source_index"],"source_split":"train","dataset":"HotpotQA","original_question":source["question"],"original_answer":source["answer"],"attribute":spec["attribute"],"subjects":spec["subjects"],"items":items,"coverage":{"complete":[True,True,True],"partial":coverage},"removed_subject_index":removed,"evidence_review":spec["review"],"topic":spec["topic"]})
        refs[-1].update(group_id=qid,derived_attribute_changed=spec.get("derived_attribute_changed",False),derivation_note=spec.get("derivation_note","The original attribute is preserved; the third question is an open comparison independently derived from both source values."))
        if spec.get("upstream_gold_issue"):refs[-1]["upstream_gold_issue"]=spec["upstream_gold_issue"]
        inventory.append({"question_id":qid,"source_index":spec["source_index"],"split":spec["split"],"source_titles":sorted(titles),"subjects":spec["subjects"],"removed_subject_index":removed,"replacement_keeps_subject_source":replacement["source_title"]==keys[removed]["source_title"],"removed_tokens":a,"replacement_tokens":b,"absolute_token_difference":abs(a-b),"passage_order":order})
    return inputs,refs,inventory


def main():
    specs=load(ROOT/"data/curation/reviewed_specs.json")
    inputs,refs,inventory=build(specs)
    dev=[x for x in inputs if x["split"]=="development"]
    official=[x for x in inputs if x["split"]!="development"]
    write_jsonl(ROOT/"data/dev_inputs.jsonl",dev)
    write_jsonl(ROOT/"data/dev_references.jsonl",[x for x in refs if any(r["question_id"]==x["question_id"] for r in dev)])
    official_refs=[x for x in refs if any(r["question_id"]==x["question_id"] for r in official)]
    main_inputs=list(official)
    external=[]
    if official:
        assert len(official)==400 and len(official_refs)==200,"Formal input creation requires the complete200-group source freeze."
        external=read_jsonl(ROOT/"data/external_ragognize/external_inputs.jsonl")
        external_refs=read_jsonl(ROOT/"data/external_ragognize/external_references.jsonl")
        assert len(external)==100 and len(external_refs)==50
        source_titles={p.get("source_title",p["title"]).casefold() for r in inputs for p in r["passages"]}
        subjects={s.casefold() for r in inputs for s in r["subjects"]}
        assert not source_titles&{p.get("source_title",p["title"]).casefold() for r in external for p in r["passages"]},"Main/development and external share source titles."
        assert not subjects&{s.casefold() for r in external for s in r["subjects"]},"Main/development and external share key subjects."
        official+=external
        official_refs+=external_refs
    write_jsonl(ROOT/"data/inputs.jsonl",official)
    write_jsonl(ROOT/"data/references.jsonl",official_refs)
    dump(ROOT/"data/curation/group_inventory.json",inventory)
    summary={"status":"curation_in_progress_not_frozen","development_groups":len(dev)//2,"official_main_groups":len(main_inputs)//2,"external_groups":len(external)//2,"splits":dict(Counter(x["split"] for x in inventory)),"generation_outputs_consulted":False,"review_kind":"Assistant source-only semantic review; not independent human annotation","snippet_policy":"Only complete original sentences; exact sentence indices preserved. Same extraction policy for complete and partial. Full upstream rows preserved for curation; only displayed snippets are model-visible.","files_sha256":{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"data/dev_inputs.jsonl",ROOT/"data/dev_references.jsonl",ROOT/"data/inputs.jsonl",ROOT/"data/references.jsonl",ROOT/"data/curation/reviewed_specs.json",ROOT/"data/curation/group_inventory.json"]}}
    dump(ROOT/"data/data_manifest.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="files_sha256"},ensure_ascii=False))


if __name__=="__main__":main()
