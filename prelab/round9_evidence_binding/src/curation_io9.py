"""Source packets and exact-quote storage; the caller makes semantic decisions."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def readl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def pool():
    return {r["candidate_id"]: r for r in readl(ROOT/"data/curation/source_pool.jsonl")}


def segments(text):
    spans = []
    for paragraph in re.finditer(r"[^\n]+", text):
        raw = paragraph.group()
        if raw.strip().startswith("=="):
            continue
        pieces = list(re.finditer(r".*?(?:[.!?](?=\s+[A-Z0-9\"“])|$)", raw))
        pending = None
        for m in pieces:
            start, end = paragraph.start()+m.start(), paragraph.start()+m.end()
            while start < end and text[start].isspace(): start += 1
            while end > start and text[end-1].isspace(): end -= 1
            if start == end:
                continue
            if pending is not None:
                start = pending
            if re.search(r"(?:\b[A-Z]|\bMr|\bMrs|\bDr|\bProf|\bSt)\.$", text[start:end]):
                pending = start
                continue
            spans.append({"start": start, "end": end, "text": text[start:end]})
            pending = None
        if pending is not None:
            spans.append({"start": pending, "end": paragraph.end(), "text": text[pending:paragraph.end()]})
    return spans


def packet(ids):
    all_rows = pool()
    for cid in ids:
        r = all_rows[cid]
        print("SOURCE", cid, r["source_title"], r["source_url"], "revision", r["revision_id"])
        print("OLD_QUESTION", r["original_question"])
        print("OLD_QUOTE", r["original_answer_quote"])
        sentences = segments(r["source_content"])
        print("SEGMENTS", len(sentences))
        for index, s in enumerate(sentences):
            marker = " ANSWER_QUOTE" if r["original_answer_quote"].casefold() in s["text"].casefold() else ""
            print(index, s["text"], marker)


def add_manual(path, decisions):
    """Store explicit selections only, with unique original quotes and source identity."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"curator":"root","decisions":[]}
    old = {r["candidate_id"]: r for r in doc["decisions"]}
    all_rows = pool()
    for d in decisions:
        cid = d["candidate_id"]
        assert cid not in old, "Never overwrite a reviewed candidate silently"
        r = all_rows[cid]
        assert d["category"] in {"time","quantity","location","relation","action"}
        assert d["question"].strip() and d["reference_answer"].strip() and d["rationale"].strip()
        intervals = []
        for field in ("evidence_quote","common_quote","partial_quote"):
            q = d[field]
            assert q == q.strip() and r["source_content"].count(q) == 1, (cid, field, q)
            a = r["source_content"].index(q)
            d[field+"_range"] = [a,a+len(q)]
            intervals.append((a,a+len(q)))
        ordered = sorted(intervals)
        assert all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:])), (cid,"quote overlap")
        d.update(source_content_sha256=r["source_content_sha256"],source_title=r["source_title"],
                 status="source_reviewed_candidate",model_outputs_viewed=False)
        doc["decisions"].append(d)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MANUAL_SOURCE_DECISIONS",len(doc["decisions"]),str(path))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("ids",nargs="+")
    packet(p.parse_args().ids)
