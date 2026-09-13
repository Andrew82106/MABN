"""Resumable Round9 generation and white-box stages; labels are never loaded.

audit is CPU-only. prepare-freeze hashes already prepared inputs/protocol/random
assignments; it does not select donors, generate data, or run a model.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
R7 = ROOT.parent / "round7_evidence_grounding"
if str(R7 / "src") not in sys.path:
    sys.path.insert(0, str(R7 / "src"))
import model7
import attention7
import lumina7
import binding9

VERSION = "round9-runner-v1"
FEATURE_FOLDERS = {"core": "features", "attention": "attention", "lumina": "lumina"}
IDENTITY = ("row_id", "question_id", "group_id", "split", "condition", "dataset", "category")
GENERATION_KEYS = ("input_token_ids", "response_token_ids", "response",
                   "response_token_offsets", "unexpected_special_token_ids")
ITEM_KEYS = ("item_id", "item_index", "text", "start", "end", "last_content_character",
             "parse_ok", "parse_reason")
DEFAULT_RANDOM = "data/lumina_random_manifest.json"
LOCKED_DOCUMENTS = ("PLAN.md", "ANNOTATION_GUIDE.md", "EVALUATION_DRAFT.md", "FEATURES_DRAFT.md")
EXTERNAL_SOURCES = (
    "../round7_evidence_grounding/src/model7.py",
    "../round7_evidence_grounding/src/attention7.py",
    "../round7_evidence_grounding/src/lumina7.py",
    "../round8_token_localization/src/evaluate8.py",
)


def readl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return model7.digest(value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def savel(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def inside(root, name):
    root, path = Path(root).resolve(), (Path(root) / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Round9 artifact path escapes selected root: " + str(path))
    return path


def relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")


def safe_identity(row):
    import re
    rid = row["row_id"]
    if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", rid) or rid in (".", ".."):
        raise ValueError("row_id must be a safe opaque artifact identifier")
    return {key: row[key] for key in IDENTITY if key in row}


def visible(row):
    """Explicitly exclude category, subjects, aliases, references and condition."""
    return binding9.visible_only(row)


def extractor_row(row):
    result = visible(row)
    # Legacy attention/LUMINA use these opaque identifiers only for integrity.
    result.update(row_id=row["row_id"], question_id=row["question_id"])
    return result


def extractor_generation(generated):
    result = {key: generated[key] for key in GENERATION_KEYS if key in generated}
    result["row_id"] = generated["row_id"]
    result["items"] = [{k: item[k] for k in ITEM_KEYS if k in item}
                       for item in generated.get("items", [])]
    return result


def generation_row(row):
    result = extractor_row(row)
    # model7.generate_answer requires these result-metadata fields. Constant
    # placeholders avoid passing actual hidden condition/split to the backbone
    # wrapper; authoritative run identity is attached only after generation.
    result.update(split="unlabelled", condition="unlabelled",
                  expected_items=len(result["questions"]))
    return result


def validate_visible_strings(row):
    vis = visible(row)
    if not vis["questions"] or vis["prompt"].count(binding9.SEARCH_MARKER) != 1:
        raise ValueError("Invalid question/search prompt layout")
    before, source = vis["prompt"].split(binding9.SEARCH_MARKER)
    qm = "\n\nQuestions:\n"
    expected = "\n".join(f"{i+1}. {q}" for i,q in enumerate(vis["questions"]))
    if qm not in before or before.split(qm, 1)[1] != expected:
        raise ValueError("Visible questions disagree with prompt")
    expected_sources = "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(vis["passages"]))
    if source != expected_sources:
        raise ValueError("Visible passage bodies disagree with prompt")
    return vis


def generation_signature():
    funcs = [model7.generate_answer, model7.parse_items, model7.generation_config,
             model7.chat_ids, model7.token_offsets, generation_row, extractor_row,
             visible, binding9.visible_only]
    return {
        "config": dict(model7.CONFIG),
        "generator_functions_sha256": digest("\n".join(inspect.getsource(f) for f in funcs)),
        "model_config_sha256": sha(model7.MODEL / "config.json"),
        "tokenizer_config_sha256": sha(model7.MODEL / "tokenizer_config.json"),
        "wrapper_version": VERSION,
    }


def stage_signature(stage):
    files = [Path(__file__), Path(model7.__file__)]
    if stage == "core":
        files += [Path(binding9.__file__), Path(attention7.__file__)]
    elif stage == "attention":
        files.append(Path(attention7.__file__))
    elif stage == "lumina":
        files.append(Path(lumina7.__file__))
    return {"stage": stage, "schema": VERSION,
            "code_hashes": {p.name: sha(p) for p in files},
            "generator": generation_signature(),
            "torch_version": torch.__version__}


def validate_random_manifest(root, path, rows):
    """Read fixed donor assignments; never select by label or generation."""
    root, path = Path(root), Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "round9-random-context-v1":
        raise ValueError("Unknown random-context manifest schema")
    pool_path = (root / manifest["pool_file"]).resolve()
    expected_pool = (R7 / "data/dev_inputs.jsonl").resolve()
    if pool_path != expected_pool:
        raise ValueError("Random pool must be the independently fixed R7 dev_inputs.jsonl")
    if sha(pool_path) != manifest["pool_sha256"]:
        raise ValueError("Random source pool changed")
    pool = {r["row_id"]: r for r in readl(pool_path)}
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    assignments = manifest["assignments"]
    if set(assignments) != set(groups):
        raise ValueError("Random assignments must cover exactly the frozen question groups")
    donors = {}
    for gid, group in groups.items():
        choice = assignments[gid]
        if not choice.get("donor_id"):
            raise ValueError("Fixed donor identity is required")
        passages = []
        if len({len(r["passages"]) for r in group}) != 1:
            raise ValueError("Paired inputs have different source counts; one fixed LUMINA donor cannot serve both")
        count = len(group[0]["passages"])
        if not count or len(choice["passages"]) != count:
            raise ValueError("LUMINA donor must have same nonzero passage count")
        target_titles = {p["title"] for r in group for p in visible(r)["passages"]}
        target_texts = {p["text"] for r in group for p in visible(r)["passages"]}
        origins = []
        for passage in choice["passages"]:
            origin = passage["origin_row_id"]
            if origin not in pool:
                raise ValueError("Donor origin is not in fixed development pool")
            if pool[origin].get("split") != "development":
                raise ValueError("Donor source has invalid development designation")
            pv = {"title": passage["title"], "text": passage["text"]}
            if pv not in visible(pool[origin])["passages"]:
                raise ValueError("Donor title/text not identical to the claimed dev source")
            if pv["title"] in target_titles or pv["text"] in target_texts:
                raise ValueError("Donor overlaps actual target source")
            passages.append(pv)
            origins.append(origin)
        if len({digest(p) for p in passages}) != len(passages):
            raise ValueError("Duplicate random donor passage")
        donors[gid] = {
            "row_id": choice["donor_id"], "question_id": "independent_" + digest(origins)[:20],
            "passages": passages,
        }
    return manifest, donors


def validate_protocol(root, protocol):
    for section in ("generation", "model"):
        for key,value in protocol.get(section, {}).items():
            if key in model7.CONFIG and model7.CONFIG[key] != value:
                raise ValueError("Protocol value differs from reused model7 configuration: " + section + "." + key)
    model = protocol.get("model", {})
    if "name" in model and model["name"] != model7.MODEL.name:
        raise ValueError("Protocol checkpoint name differs from actual checkpoint")
    if "path" in model and (Path(root)/model["path"]).resolve() != model7.MODEL.resolve():
        raise ValueError("Protocol checkpoint path differs from actual checkpoint")
    for key,value in {"frozen_weights":True,"quantization":"NF4","dtype":"bfloat16"}.items():
        if key in model and model[key] != value:
            raise ValueError("Unsupported model protocol value: " + key)


def external_source_hashes():
    # Imports refer to the actual sibling source trees even for a --root fixture.
    return {name:sha((ROOT/name).resolve()) for name in EXTERNAL_SOURCES}


def prepare_freeze(root=ROOT, input_name="data/inputs.jsonl",
                   protocol_name="protocol.json", random_name=DEFAULT_RANDOM,
                   freeze_name="data/freeze.json"):
    """Explicit CPU preparation; no automatic donor or source selection."""
    root = Path(root)
    target = inside(root, freeze_name)
    if target.exists():
        raise FileExistsError("Do not overwrite an existing data freeze")
    inputs, protocol, random_path = (inside(root, n) for n in (input_name, protocol_name, random_name))
    rows = readl(inputs)
    if not rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Inputs empty or repeated row IDs")
    validate_protocol(root,json.loads(protocol.read_text(encoding="utf-8")))
    for row in rows:
        safe_identity(row)
        validate_visible_strings(row)
    validate_random_manifest(root, random_path, rows)
    documents = [root/name for name in LOCKED_DOCUMENTS if (root/name).is_file()]
    sources = sorted((root/"src").rglob("*.py"))
    value = {
        "schema": "round9-input-freeze-v1", "status": "frozen",
        "protocol_sha256": sha(protocol), "protocol_file": relative(root, protocol),
        "files_sha256": {relative(root, p): sha(p) for p in (inputs, random_path, *documents, *sources)},
        "locked_source_files": [relative(root,p) for p in sources],
        "external_source_sha256": external_source_hashes(),
        "expected_rows": len(rows), "expected_groups": len({r["group_id"] for r in rows}),
        "created_without_generation_or_labels": True,
    }
    save(target, value)
    return value


def preflight(root=ROOT, input_name="data/inputs.jsonl", protocol_name="protocol.json",
              random_name=DEFAULT_RANDOM, freeze_name="data/freeze.json"):
    root = Path(root)
    inputs, protocol, random_path, lock = (inside(root, n) for n in (input_name, protocol_name, random_name, freeze_name))
    frozen = json.loads(lock.read_text(encoding="utf-8"))
    if frozen.get("schema") != "round9-input-freeze-v1" or frozen.get("status") != "frozen":
        raise ValueError("A completed Round9 data freeze is required before any model load")
    if frozen["protocol_sha256"] != sha(protocol):
        raise ValueError("Protocol changed after data freeze")
    for path in (inputs, random_path):
        if frozen["files_sha256"].get(relative(root,path)) != sha(path):
            raise ValueError("Frozen input/random manifest changed: " + str(path))
    for name, expected in frozen["files_sha256"].items():
        if sha(inside(root,name)) != expected:
            raise ValueError("Additional frozen data changed: " + name)
    if frozen.get("external_source_sha256") != external_source_hashes():
        raise ValueError("External frozen source inventory or hashes changed")
    rows = readl(inputs)
    if not rows or len({r["row_id"] for r in rows}) != len(rows) or len(rows) != frozen["expected_rows"]:
        raise ValueError("Frozen input coverage is invalid")
    if len({r["group_id"] for r in rows}) != frozen["expected_groups"]:
        raise ValueError("Frozen question-group count is invalid")
    for row in rows:
        safe_identity(row)
        validate_visible_strings(row)
    prot = json.loads(protocol.read_text(encoding="utf-8"))
    validate_protocol(root,prot)
    source_files = [relative(root,p) for p in sorted((root/"src").rglob("*.py"))]
    if source_files != frozen["locked_source_files"]:
        raise ValueError("Round9 source file inventory changed after freeze")
    random_manifest, donors = validate_random_manifest(root,random_path,rows)
    return {
        "root": root, "rows": rows, "donors": donors, "protocol": prot,
        "freeze_path": lock, "protocol_path": protocol, "input_path": inputs,
        "random_path": random_path, "frozen": frozen,
        "freeze_sha256": sha(lock), "protocol_sha256": sha(protocol),
        "input_sha256": sha(inputs), "random_manifest_sha256": sha(random_path),
    }


def unchanged(context):
    for pathkey, digestkey in (("freeze_path","freeze_sha256"),("protocol_path","protocol_sha256"),
                               ("input_path","input_sha256"),("random_path","random_manifest_sha256")):
        if sha(context[pathkey]) != context[digestkey]:
            raise RuntimeError("Frozen material changed while the runner was active: " + pathkey)
    for name,expected in context["frozen"]["files_sha256"].items():
        if sha(inside(context["root"],name)) != expected:
            raise RuntimeError("Frozen document, code or data changed while the runner was active: " + name)
    source_files = [relative(context["root"],p) for p in sorted((context["root"]/"src").rglob("*.py"))]
    if source_files != context["frozen"]["locked_source_files"]:
        raise RuntimeError("Round9 source file inventory changed while the runner was active")
    if context["frozen"].get("external_source_sha256") != external_source_hashes():
        raise RuntimeError("External frozen source inventory or hashes changed while the runner was active")


def checked(path, expected, arrays=False):
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    for k,v in expected.items():
        if record.get(k) != v:
            raise RuntimeError(f"Stale committed cache {path.name}: {k}; do not overwrite")
    if arrays:
        npz = path.with_suffix(".npz")
        if not npz.exists() or sha(npz) != record.get("arrays_sha256"):
            raise RuntimeError("Committed array cache missing or changed: " + str(npz))
    return record


def generation_expected(context,row,generation_hash=None):
    return {"generation_signature_hash": generation_hash or digest(generation_signature()),
            "prompt_hash": model7.prompt_hash(visible(row)),
            "visible_input_sha256": digest(visible(row)),
            "data_freeze_sha256": context["freeze_sha256"],
            "protocol_sha256": context["protocol_sha256"]}


def validate_generated(tok,row,generated):
    if generated["row_id"] != row["row_id"]:
        raise ValueError("Generation row identity mismatch")
    vis = visible(row)
    if model7.chat_ids(tok,vis["prompt"],vis["system"]) != generated["input_token_ids"]:
        raise ValueError("Saved input token IDs are not the exact visible prompt")
    ids, text = generated["response_token_ids"], generated["response"]
    if tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False) != text:
        raise ValueError("Saved exact response token IDs do not decode to response")
    offsets = model7.token_offsets(tok,ids,text).reshape(-1,2)
    if not np.array_equal(offsets,np.asarray(generated["response_token_offsets"]).reshape(-1,2)):
        raise ValueError("Saved response offsets changed")
    for item in generated["items"]:
        if item.get("parse_ok"):
            if text[item["start"]:item["end"]] != item["text"]:
                raise ValueError("Item offsets disagree with actual output")


def validate_arrays(stage,arrays,generated,meta):
    n = len(generated["response_token_ids"])
    offsets = np.asarray(generated["response_token_offsets"]).reshape(-1,2)
    if stage == "core":
        expected = {"hidden_28":(n,3584),"lookback_features":(n,784),"binding_features":(n,32),
                    "token_nll":(n,),"token_entropy":(n,),"token_ids":(n,),
                    "token_start":(n,),"token_end":(n,)}
        for k, shape in expected.items():
            if k not in arrays or arrays[k].shape != shape:
                raise ValueError(f"{stage} {k} dimensions changed")
        if arrays["token_ids"].tolist() != generated["response_token_ids"]:
            raise ValueError("Core token IDs mismatch")
        if not np.array_equal(arrays["token_start"],offsets[:,0]) or not np.array_equal(arrays["token_end"],offsets[:,1]):
            raise ValueError("Core token positions mismatch")
        names = meta["binding_feature_names"]
        if len(names) != 32 or len(set(names)) != 32:
            raise ValueError("Binding names invalid")
    elif stage == "attention":
        expected={"token_redeep_ecs":(n,784),"token_redeep_pks":(n,28),"token_lookback":(n,784)}
        for k,shape in expected.items():
            if arrays[k].shape!=shape:
                raise ValueError("Attention axes mismatch: "+k)
        if not np.array_equal(arrays["response_token_offsets"],offsets):
            raise ValueError("Attention token offsets mismatch")
    elif stage == "lumina":
        expected={"token_lumina_score":(n,)}
        if arrays["token_lumina_score"].shape!=(n,) or not np.array_equal(arrays["response_token_offsets"],offsets):
            raise ValueError("LUMINA token positions mismatch")
    else:
        raise ValueError("Unknown array stage")
    # Legacy auxiliary item means may legitimately be NaN for parse failures;
    # native token matrices used by Bank must always remain finite.
    for k in expected:
        if not np.isfinite(arrays[k]).all():
            raise FloatingPointError("Nonfinite native features: " + k)


def pack(context, area):
    """Atomic aggregate from committed per-row files; never rewrite row records."""
    rows, generations = context["rows"], []
    gen_hash = digest(generation_signature())
    for row in rows:
        p = area/"generation_records"/(row["row_id"]+".json")
        rec = checked(p,generation_expected(context,row,gen_hash))
        if rec is not None:
            generations.append(rec)
    savel(area/"generated.jsonl",generations)
    save(area/"generation_manifest.json",{
        "schema":VERSION,"expected_rows":len(rows),"completed_rows":len(generations),
        "complete":len(rows)==len(generations),"generated_file_sha256":sha(area/"generated.jsonl"),
        "data_freeze_sha256":context["freeze_sha256"],"protocol_sha256":context["protocol_sha256"],
        "labels_read":False,"external_generation_model_calls":0,
        "record_sha256":{r["row_id"]:sha(area/"generation_records"/(r["row_id"]+".json")) for r in generations}})
    for stage,folder in FEATURE_FOLDERS.items():
        summaries=[]
        signature_hash=digest(stage_signature(stage))
        for row in rows:
            path=area/folder/(row["row_id"]+".json")
            if not path.exists():
                continue
            gp=area/"generation_records"/(row["row_id"]+".json")
            expected={**generation_expected(context,row,gen_hash),"stage_signature_hash":signature_hash,
                      "source_generation_sha256":sha(gp)}
            if stage=="lumina":
                expected["random_context_sha256"]=digest(context["donors"][row["group_id"]]["passages"])
            rec=checked(path,expected,arrays=True)
            summaries.append(rec)
        save(area/folder/"manifest.json",{
            "schema":VERSION,"stage":stage,"expected_rows":len(rows),"completed_rows":len(summaries),
            "complete":len(rows)==len(summaries),"seconds":sum(r.get("seconds",0) for r in summaries),
            "peak_allocated_gib":max((r.get("peak_allocated_gib") or 0 for r in summaries),default=0),
            "records":{r["row_id"]:sha(area/folder/(r["row_id"]+".json")) for r in summaries},
            "data_freeze_sha256":context["freeze_sha256"],"protocol_sha256":context["protocol_sha256"],
            "labels_read":False})


def load_tokenizer():
    return AutoTokenizer.from_pretrained(model7.MODEL,local_files_only=True)


def run(context, stages, *, area_name="data", row_ids=None, limit=None, loader=None):
    area = inside(context["root"],area_name)
    rows = context["rows"]
    work = rows
    if row_ids is not None:
        wanted=set(row_ids)
        work=[r for r in rows if r["row_id"] in wanted]
        if {r["row_id"] for r in work} != wanted:
            raise ValueError("Requested rows are absent from frozen inputs")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        work=work[:limit]
    tok=model=None
    loader=loader or model7.load_model
    gen_hash=digest(generation_signature())
    for stage in stages:
        if stage not in ("generate",*FEATURE_FOLDERS):
            raise ValueError("Unknown GPU stage")
        stage_sig=stage_signature(stage)
        for index,row in enumerate(work,1):
            unchanged(context)
            gp=area/"generation_records"/(row["row_id"]+".json")
            base=generation_expected(context,row,gen_hash)
            generated=checked(gp,base)
            target=gp if stage=="generate" else area/FEATURE_FOLDERS[stage]/(row["row_id"]+".json")
            expected=dict(base)
            if stage!="generate":
                if generated is None:
                    raise RuntimeError("Generate the requested row before extracting: "+row["row_id"])
                expected.update(stage_signature_hash=digest(stage_sig),source_generation_sha256=sha(gp))
            donor=context["donors"][row["group_id"]] if stage=="lumina" else None
            if donor is not None:
                expected["random_context_sha256"]=digest(donor["passages"])
            cache=checked(target,expected,arrays=stage!="generate")
            if cache is not None:
                print("CACHE",stage,index,len(work),row["row_id"],flush=True)
                continue
            # All data/protocol/cache checks above happen before model loading.
            if model is None:
                unchanged(context)
                tok,model=loader()
            device=model.model.embed_tokens.weight.device
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
            started=time.perf_counter()
            if stage=="generate":
                value=model7.generate_answer(tok,model,generation_row(row))
                value.update(safe_identity(row),expected_items=len(visible(row)["questions"]))
                validate_generated(tok,row,value)
                value.update(expected,generation_signature=generation_signature())
            else:
                validate_generated(tok,row,generated)
                source_gen=extractor_generation(generated)
                if stage=="core":
                    arrays,value=binding9.extract_binding_features(tok,model,visible(row),source_gen)
                elif stage=="attention":
                    arrays,value=attention7.extract_attention(tok,model,extractor_row(row),source_gen)
                else:
                    arrays,value=lumina7.extract_lumina(tok,model,extractor_row(row),source_gen,donor)
                validate_arrays(stage,arrays,generated,value)
                # Redundant exact token identities make every Bank input auditable.
                arrays.setdefault("token_ids",np.asarray(generated["response_token_ids"],dtype=np.int64))
                save_npz(target.with_suffix(".npz"),arrays)
                value.update(expected,row_id=row["row_id"],arrays_sha256=sha(target.with_suffix(".npz")))
            if device.type=="cuda":
                torch.cuda.synchronize(device)
            value.update(seconds=time.perf_counter()-started,
                         peak_allocated_gib=float(torch.cuda.max_memory_allocated(device)/2**30) if device.type=="cuda" else None,
                         stage_signature=stage_sig,runner_sha256=sha(__file__),
                         labels_read=False,actual_condition_passed_to_extractor=False)
            unchanged(context)
            save(target,value)
            # Incremental aggregate permits annotation while later generations run.
            if stage=="generate":
                pack(context,area)
            print("DONE",stage,index,len(work),row["row_id"],"seconds",round(value["seconds"],3),flush=True)
        pack(context,area)
    return {"stages":list(stages),"selected_rows":len(work),"area":str(area)}


def audit(context, area_name="data", require_complete=False):
    """Read-only CPU audit. No model load, no cache regeneration or fitting."""
    area=inside(context["root"],area_name)
    unchanged(context)
    tok=load_tokenizer()
    gen_hash=digest(generation_signature())
    signature_hashes={s:digest(stage_signature(s)) for s in FEATURE_FOLDERS}
    report={"schema":VERSION,"data_freeze_sha256":context["freeze_sha256"],
            "protocol_sha256":context["protocol_sha256"],"model_loaded":False,
            "expected_rows":len(context["rows"]),"generated":0,
            "stages":{s:0 for s in FEATURE_FOLDERS},"missing":[],
            "binding_names":None,"candidate_missing_questions":0,"route_invalid_tokens":0}
    generated_records=[]
    for row in context["rows"]:
        gp=area/"generation_records"/(row["row_id"]+".json")
        g=checked(gp,generation_expected(context,row,gen_hash))
        if g is None:
            report["missing"].append({"stage":"generate","row_id":row["row_id"]})
            continue
        validate_generated(tok,row,g)
        generated_records.append(g);report["generated"]+=1
        for stage,folder in FEATURE_FOLDERS.items():
            expected={**generation_expected(context,row,gen_hash),"stage_signature_hash":signature_hashes[stage],
                      "source_generation_sha256":sha(gp)}
            if stage=="lumina":
                expected["random_context_sha256"]=digest(context["donors"][row["group_id"]]["passages"])
            path=area/folder/(row["row_id"]+".json")
            side=checked(path,expected,arrays=True)
            if side is None:
                report["missing"].append({"stage":stage,"row_id":row["row_id"]})
                continue
            with np.load(path.with_suffix(".npz"),allow_pickle=False) as handle:
                arrays={k:handle[k].copy() for k in handle.files}
            validate_arrays(stage,arrays,g,side)
            report["stages"][stage]+=1
            if stage=="core":
                names=side["binding_feature_names"]
                if report["binding_names"] is not None and report["binding_names"]!=names:
                    raise ValueError("Binding axis names changed across rows")
                report["binding_names"]=names
                report["candidate_missing_questions"]+=side.get("target_candidate_missing_questions",0)
                report["route_invalid_tokens"]+=side.get("route_invalid_tokens",0)
    if (area/"generated.jsonl").exists() and readl(area/"generated.jsonl")!=generated_records:
        raise ValueError("Aggregate generation is stale; resume generation stage to repack")
    report["complete"]=not report["missing"]
    if require_complete and not report["complete"]:
        raise RuntimeError("Required extraction stages are incomplete")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--stage",required=True,choices=["prepare-freeze","generate","core","attention","lumina","all","audit"])
    ap.add_argument("--inputs",default="data/inputs.jsonl")
    ap.add_argument("--protocol",default="protocol.json")
    ap.add_argument("--freeze",default="data/freeze.json")
    ap.add_argument("--random-manifest",default=DEFAULT_RANDOM)
    ap.add_argument("--area",default="data")
    ap.add_argument("--row-ids")
    ap.add_argument("--limit",type=int)
    ap.add_argument("--require-complete",action="store_true")
    args=ap.parse_args()
    if args.stage=="prepare-freeze":
        result=prepare_freeze(args.root,args.inputs,args.protocol,args.random_manifest,args.freeze)
    else:
        ctx=preflight(args.root,args.inputs,args.protocol,args.random_manifest,args.freeze)
        if args.stage=="audit":
            result=audit(ctx,args.area,args.require_complete)
        else:
            stages=["generate","core","attention","lumina"] if args.stage=="all" else [args.stage]
            result=run(ctx,stages,area_name=args.area,
                       row_ids=args.row_ids.split(",") if args.row_ids else None,limit=args.limit)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
