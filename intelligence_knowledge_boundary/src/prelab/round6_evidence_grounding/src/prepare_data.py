"""Deterministic data preparation, without generation or output-based selection.

The semantic decisions live in curated/candidate_specs.json and review_notes.json.
This script only builds the reviewed inputs and checks mechanical constraints.
"""
import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PRELAB = ROOT.parent
SEED = 20260910


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(s):
    return re.sub(r"\s+", " ", s.casefold()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="Freeze only after both independent source reviews accept the exact final data")
    args = ap.parse_args()
    curated = ROOT / "data/curated"
    curated.mkdir(parents=True, exist_ok=True)
    rows = {}
    provenance = []
    for p in sorted((ROOT / "data/raw").glob("hotpotqa_validation_[0-9]*.json")):
        if p.name.endswith(".meta.json"):
            continue
        meta = read_json(p.with_suffix(".meta.json"))
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"]
        provenance.append(meta)
        for x in read_json(p)["rows"]:
            rows[x["row_idx"]] = x["row"]
    tok = AutoTokenizer.from_pretrained(PRELAB / "models/Qwen2.5-7B-Instruct-bnb-4bit", local_files_only=True)
    n_tokens = lambda text: len(tok.encode(text, add_special_tokens=False))
    preview = read_json(ROOT / "planning/hotpotqa_preview.json")["rows"]
    preview_ids = {r["id"] for r in preview}
    preview_titles = {norm(t) for r in preview for t in r["context"]["title"]}
    specs = read_json(curated / "candidate_specs.json")
    protected = {norm(s) for sp in specs for s in sp["subjects"]}
    global_background = []
    pool_seen = set()
    for source_index, r in rows.items():
        if source_index < 500 or r["type"] != "bridge":
            continue
        for title,sentences in zip(r["context"]["title"],r["context"]["sentences"]):
            txt="".join(sentences).strip()
            if norm(title) in pool_seen or norm(title) in preview_titles or any(s in norm(title+" "+txt) for s in protected):
                continue
            pool_seen.add(norm(title))
            global_background.append({"title":title,"text":txt,"sentences":sentences,"origin_question_id":r["id"],"origin_row_index":source_index})
    token_cache={}
    def tokens(text):
        if text not in token_cache: token_cache[text]=n_tokens(text)
        return token_cache[text]
    decisions, proposals = [], []
    for spec in specs:
        r = rows[spec["source_index"]]
        ctx = dict(zip(r["context"]["title"], r["context"]["sentences"]))
        passages = [{"title": title, "text": "".join(sentences).strip(), "sentences": sentences} for title, sentences in ctx.items()]
        by_title = {p["title"]: p for p in passages}
        subjects = spec["subjects"]
        keys = spec.get("titles", subjects)
        if r["id"] in preview_ids or any(norm(k) in preview_titles for k in keys):
            decisions.append({"source_index": spec["source_index"], "question_id": r["id"], "status": "rejected", "reason": "planning preview or shared key source"})
            continue
        keypass = [by_title[k] for k in keys]
        # Names are only a conservative candidate screen; full passages receive semantic review.
        aliases = spec.get("subject_aliases", [[s] for s in subjects])
        aliases = [norm(s) for group in aliases for s in group] + [norm(s) for s in subjects]
        background = [p for p in passages if p["title"] not in keys and norm(p["title"]) not in preview_titles]
        options = []
        for removed in spec.get("allowed_removed_subjects", [0, 1]):
            target_n = tokens(keypass[removed]["text"])
            local_rep=[rep for rep in background if abs(tokens(rep["text"])-target_n)<=max(20,.2*target_n) and len([p for p in background if p["title"]!=rep["title"]])>=2]
            replacement_pool=local_rep or sorted(global_background,key=lambda x:(abs(tokens(x["text"])-target_n),x["title"]))[:20]
            for rep in replacement_pool:
                rep_n = tokens(rep["text"])
                if abs(rep_n-target_n) <= max(20, .2*target_n):
                    others = sorted([p for p in background if p["title"] != rep["title"]], key=lambda p:(tokens(p["text"]),p["title"]))
                    if len(others) >= 2:
                        options.append({"removed_subject_index":removed,"replacement":rep,"background":others[:2],"removed_tokens":target_n,"replacement_tokens":rep_n,"token_difference":abs(rep_n-target_n),"replacement_scope":"same_question" if rep in background else "other_question_natural_paragraph"})
        if not options:
            decisions.append({"source_index":spec["source_index"],"question_id":r["id"],"status":"rejected","reason":"No full natural replacement paragraph meeting token length constraint with two other eligible backgrounds"})
            continue
        proposals.append({"source_index":spec["source_index"],"question_id":r["id"],"original_question":r["question"],"original_answer":r["answer"],"spec":spec,"key_passages":keypass,"options":options})
    write_json(curated / "proposals.json", proposals)
    write_json(curated / "mechanical_candidate_exclusions.json", decisions)
    print(json.dumps({"raw_rows":len(rows),"specified":len(specs),"eligible":len(proposals),"exclusions":decisions},ensure_ascii=False))
    if args.draft:
        return
    review = read_json(curated / "review_notes.json")
    accepted = [x for x in review if x["decision"] == "accepted"]
    assert len(accepted) == 30
    proposed = {p["source_index"]:p for p in proposals}
    rng = random.Random(SEED)
    inputs, references, selected, used_titles, used_texts, used_subjects = [], [], [], set(), set(), set()
    template = (ROOT / "prompts/answer.txt").read_text(encoding="utf-8").strip()
    for review_item in accepted:
        p = proposed[review_item["source_index"]]
        spec = p["spec"]
        r = rows[p["source_index"]]
        opt = p["options"][review_item["option_index"]]
        if "override_replacement_title" in review_item:
            opt=dict(opt)
            matches=[x for x in global_background if x["title"]==review_item["override_replacement_title"]]
            assert len(matches)==1
            opt["replacement"]=matches[0]
            opt["replacement_scope"]="other_question_natural_paragraph"
            opt["replacement_tokens"]=tokens(matches[0]["text"])
            opt["token_difference"]=abs(opt["replacement_tokens"]-opt["removed_tokens"])
            assert opt["token_difference"]<=max(20,.2*opt["removed_tokens"])
        if "override_background_titles" in review_item:
            source_ctx=dict(zip(r["context"]["title"],r["context"]["sentences"]))
            opt=dict(opt)
            opt["background"]=[{"title":t,"sentences":source_ctx[t],"text":"".join(source_ctx[t]).strip()} for t in review_item["override_background_titles"]]
        removed = opt["removed_subject_index"]
        keypass = p["key_passages"]
        allp = keypass + opt["background"] + [opt["replacement"]]
        titles = {norm(x["title"]) for x in allp}
        texts = {hashlib.sha256(norm(x["text"]).encode()).hexdigest() for x in allp}
        subjects = {norm(x) for x in spec["subjects"]}
        assert not titles & used_titles, (r["id"], titles & used_titles)
        assert not texts & used_texts
        assert not subjects & used_subjects
        assert not titles & preview_titles
        used_titles |= titles; used_texts |= texts; used_subjects |= subjects
        split = review_item["split"]
        order = list(range(4)); random.Random(SEED + int(hashlib.sha256(r["id"].encode()).hexdigest()[:8],16)).shuffle(order)
        questions = spec["questions"]
        assert len(questions) == 3
        for condition in ["complete", "partial"]:
            display = list(keypass) + opt["background"]
            if condition == "partial":
                display[removed] = opt["replacement"]
            display = [display[i] for i in order]
            search_results = "\n\n".join(f"[{i+1}] {x['title']}\n{x['text']}" for i,x in enumerate(display))
            prompt = template.format(questions="\n".join(f"{i+1}. {q}" for i,q in enumerate(questions)), search_results=search_results)
            messages=[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}]
            nt=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
            assert nt <= 3072, (r["id"],nt)
            inputs.append({"row_id":r["id"]+"__"+condition,"question_id":r["id"],"split":split,"condition":condition,"questions":questions,"subjects":spec["subjects"],"passages":display,"prompt":prompt,"system":"You are a helpful assistant."})
        ev=[]
        for j,k in enumerate(keypass):
            ev.append([{"title":k["title"],"sent_id":idx,"text":k["sentences"][idx]} for idx in spec["evidence_sent_ids"][j]])
        items=[]
        for j in range(3):
            items.append({"item_index":j+1,"reference_answer":spec["answers"][j],"aliases":spec["aliases"][j],"evidence":ev[j] if j<2 else ev[0]+ev[1],"rationale":spec["rationales"][j]})
        coverage=[True,True,False];coverage[removed]=False
        references.append({"question_id":r["id"],"original_question":r["question"],"original_answer":r["answer"],"attribute":spec["attribute"],"subjects":spec["subjects"],"items":items,"removed_subject_index":removed,"coverage":{"complete":[True,True,True],"partial":coverage},"evidence_review":review_item["evidence_review"],"source_row_id":r["id"],"source_dataset_row_index":p["source_index"]})
        selected.append({"question_id":r["id"],"source_index":p["source_index"],"split":split,"removed_subject_index":removed,"attribute":spec["attribute"],"key_titles":[x["title"] for x in keypass],"background_titles":[x["title"] for x in opt["background"]],"replacement_title":opt["replacement"]["title"],"replacement_scope":opt["replacement_scope"],"replacement_origin_question_id":opt["replacement"].get("origin_question_id",r["id"]),"replacement_origin_row_index":opt["replacement"].get("origin_row_index",p["source_index"]),"removed_tokens":opt["removed_tokens"],"replacement_tokens":opt["replacement_tokens"],"token_difference":opt["token_difference"],"passage_order":order,"review":review_item["evidence_review"]})
    assert Counter(x["split"] for x in selected)=={"train":18,"validation":6,"test":6}
    assert Counter(x["removed_subject_index"] for x in selected)=={0:15,1:15}
    for name,values in [("inputs",inputs),("references",references)]:
        (ROOT/f"data/{name}.jsonl").write_text("".join(json.dumps(v,ensure_ascii=False)+"\n" for v in values),encoding="utf-8")
    write_json(curated/"selected_groups.json",selected)
    comparison_inspection=[]
    specs_by_index={s["source_index"]:s for s in specs}
    status_by_index={s["source_index"]:s for s in review}
    exclusions_by_index={s["source_index"]:s for s in decisions}
    for i,r in rows.items():
        if r["type"]!="comparison": continue
        rec={"source_index":i,"question_id":r["id"],"question":r["question"],"original_answer":r["answer"]}
        if i in status_by_index: rec.update(status_by_index[i])
        elif i in exclusions_by_index: rec.update(exclusions_by_index[i])
        elif i in specs_by_index:rec.update(decision="not_selected",reason="Eligible reserve; stopped at 30 pre-generation reviewed independent groups")
        else:rec.update(decision="not_selected",reason="Outside the predeclared reviewed attribute shortlist: not chosen for pilot; no model outputs observed")
        comparison_inspection.append(rec)
    write_json(curated/"candidate_decisions.json",comparison_inspection)
    owned_files = list((ROOT/"data/raw").rglob("*")) + list((ROOT/"data/curated").rglob("*"))
    owned_files += [ROOT/"data/inputs.jsonl",ROOT/"data/references.jsonl",ROOT/"data/dev_inputs.jsonl",ROOT/"prompts/answer.txt",Path(__file__)]
    files={str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(owned_files) if p.is_file()}
    manifest={"status":"prepared_semantically_reviewed_awaiting_root_freeze","dataset":"hotpotqa/hotpot_qa","config":"distractor","upstream_split":"validation","source_rows_fetched":len(rows),"source_comparison_rows":sum(r["type"]=="comparison" for r in rows.values()),"license":"CC BY-SA 4.0","license_url":"https://hotpotqa.github.io/","retrievals":provenance,"seed":SEED,"question_groups":30,"input_rows":60,"facts_max":180,"splits":dict(Counter(x["split"] for x in selected)),"removal_counts":dict(Counter(x["removed_subject_index"] for x in selected)),"attribute_counts":dict(Counter(x["attribute"] for x in selected)),"source_and_subject_disjoint":True,"preview_groups_excluded":sorted(preview_ids),"full_natural_source_paragraphs_only":True,"material_length_matching":"absolute Qwen token difference <= max(20, 20% removed full paragraph tokens)","maximum_chat_tokens":max(len(tok.apply_chat_template([{"role":"system","content":x["system"]},{"role":"user","content":x["prompt"]}],tokenize=True,add_generation_prompt=True)) for x in inputs),"semantic_review":"Assistant evidence-based source curation before generation; not independent human gold. No Qwen generation or score consulted.","files_sha256":files}
    if args.freeze:
        head=read_json(curated/"independent_review_head.json")
        tail=read_json(curated/"independent_review_tail.json")
        assert head["final_inputs_file_sha256"]==files["data/inputs.jsonl"]
        assert head["final_references_file_sha256"]==files["data/references.jsonl"]
        assert not head["blocking_issues_remaining"]
        assert all(x["decision"]=="accept" for x in tail["v2_effective_findings"])
        reviewed=set(head["accepted_source_indices"])|{x["source_index"] for x in tail["v2_effective_findings"]}
        assert reviewed=={x["source_index"] for x in selected}
        manifest["status"]="frozen_pilot_data"
        manifest["frozen_utc"]=datetime.now(timezone.utc).isoformat()
        manifest["freeze_note"]="Root authorized freeze after independent head/tail source reviews and exact input/reference hash checks. Development train question33 smoke is reused unchanged; no generated answer or score was used by the data curator."
    write_json(ROOT/"data/data_manifest.json",manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ["files_sha256","retrievals"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
