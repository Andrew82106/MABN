"""Persist root's explicitly chosen source segment indices, without semantic inference."""
from pathlib import Path
from curation_io9 import ROOT, pool, segments, add_manual


def add(choices):
    sources = pool()
    decisions=[]
    for cid,category,question,subjects,answer,ei,ci,pi,reason in choices:
        s=segments(sources[cid]["source_content"])
        decisions.append({"candidate_id":cid,"category":category,"question":question,
          "subjects":subjects,"reference_answer":answer,"answer_aliases":[],
          "evidence_quote":s[ei]["text"],"common_quote":s[ci]["text"],"partial_quote":s[pi]["text"],
          "rationale":reason,"question_rewrite_reason":"Neutral self-contained query; remove upstream confirmation or supplied answer.",
          "review_scope":"All packet source segments personally read; three exact source quotes explicitly selected."})
    add_manual(ROOT/'data/curation/root_relation_action.json',decisions)
