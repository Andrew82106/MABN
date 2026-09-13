"""Question-conditioned candidate attention; candidate matches are not truth labels.

Only prompt-visible strings and exact generated IDs enter the extractor. No
references, condition, subjects metadata, gold labels, or future text selection.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half

R7_SRC = Path(__file__).resolve().parents[2] / "round7_evidence_grounding" / "src"
if str(R7_SRC) not in sys.path:
    sys.path.insert(0, str(R7_SRC))
from attention7 import causal_attention_rows, lookback_from_attention, _response_offsets

VERSION = "binding9-v1"
SEARCH_MARKER = "\n\nSearch results:\n"
BANDS = ((1, 7), (8, 14), (15, 21), (22, 28))
CHANNELS = (
    "body_mass", "target_share", "target_other_density_ratio",
    "target_prev_density_ratio", "entity_only_share",
    "attribute_distractor_share", "self_mass", "pair_balance",
)
QUESTION_LEADS = set("""a an the in on at to of for from how what which who whom when
where why can could would please tell specify detail identify name list give
explain describe according during as is was were did does do has have had
provide based using and or with by about determine state mention
""".split())
MONTHS = set("january february march april may june july august september october november december".split())
ATTRIBUTE_CUES = {
    "birth": ("born", "birth", "birthday"),
    "death": ("died", "death", "deceased", "passed away"),
    "family": ("family", "families"),
    "origin": ("from", "located", "location", "origin", "originated", "based", "formed", "founded", "headquartered"),
    "frequency": ("daily", "weekly", "fortnightly", "monthly", "bimonthly", "quarterly", "annual", "annually", "frequency", "published"),
    "cooperation": ("collaborated", "collaborate", "collaboration", "partner", "partnership", "partnered", "worked with", "working with"),
    "spouse": ("spouse", "wife", "husband", "married", "marry"),
    "quantity": ("number", "total", "amount", "million", "billion", "percent", "percentage", "dollar", "cost", "price"),
    "introduction": ("released", "introduced", "production", "produced", "founded", "formed", "established", "launched"),
    "administration": ("administer", "administers", "administered", "department", "ministry", "portfolio"),
}
_DATE_INTERVAL = re.compile(r"\([^()]{0,180}\b(?:1[0-9]{3}|20[0-9]{2})\b[^()]{0,40}[–—-][^()]{0,90}\b(?:1[0-9]{3}|20[0-9]{2})\b[^()]{0,40}\)")
_KINSHIP = re.compile(r"\b(?:daughter|son|wife|husband|spouse|mother|father|children|parents)\b", re.I)
_PROPER = re.compile(r"\b(?:[A-ZÀ-ÖØ-Þ][\w'’.-]*)(?:[ -]+(?:(?:de|van|von|of|the)[ -]+)?[A-ZÀ-ÖØ-Þ][\w'’.-]*)*")


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def visible_only(source):
    """Whitelist construction, including passage children; never access subjects."""
    return {
        "system": str(source["system"]),
        "prompt": str(source["prompt"]),
        "questions": [str(q) for q in source["questions"]],
        "passages": [{"title": str(p["title"]), "text": str(p["text"])}
                     for p in source["passages"]],
    }


def contains(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(norm(phrase)) + r"(?!\w)", norm(text)))


def aliases_of(entity):
    aliases = [entity.strip().strip(" ?.,")]
    possessive = re.sub(r"(?:'s|’s)$", "", aliases[0], flags=re.I)
    if possessive != aliases[0]:
        aliases.append(possessive)
    stripped = re.sub(r"\s*\((?:musician|band|director|film|magazine|writer|footballer|politician|actor)\)\s*", "", aliases[0], flags=re.I)
    if stripped != aliases[0]:
        aliases.append(stripped.strip())
    return list(dict.fromkeys(a for a in aliases if a))


def _proper_phrases(question):
    result = []
    for match in _PROPER.finditer(question):
        phrase = match.group().strip(". ")
        words = phrase.split()
        while words and norm(words[0]) in QUESTION_LEADS:
            words.pop(0)
        phrase = " ".join(words).strip()
        if phrase and norm(phrase) not in MONTHS and norm(phrase) not in QUESTION_LEADS:
            result.append(phrase)
    return list(dict.fromkeys(result))


def question_spec(question):
    """Conservative question-only anchors; no supplied entity metadata."""
    q = question.strip()
    entities = []
    comparison = re.search(r"how do (?:the )?(.+?) of (.+?) and (.+?) compare", q, re.I)
    if comparison:
        entities = [comparison.group(2), comparison.group(3)]
    patterns = [
        r"(?:what|which) year (?:was|is) (.+?) born",
        r"(?:what|which) year did (.+?) die",
        r"(?:which|what) country is (.+?) from",
        r"(?:in|on) (?:which|what) (?:(?:US|U\.S\.) )?(?:city|state|province|island|country) (?:is|was) (.+?) located",
        r"in (?:which|what) city did (.+?) originate",
        r"to which (?:plant )?family does (.+?) belong",
        r"how frequently is (.+?) published",
        r"(?:what|which) year was (.+?) released",
        r"(?:what|which) year did production of (.+?) begin",
        r"who (.+?) collaborated with",
        r"(?:spouse|wife|husband) of (.+?)(?:[?.]|$)",
    ]
    if not entities:
        for pattern in patterns:
            found = re.search(pattern, q, re.I)
            if found:
                entities = [found.group(1)]
                break
    if not entities:
        entities = _proper_phrases(q)
    entities = [e.strip(" ?.").removeprefix("the ") for e in entities]
    entities = list(dict.fromkeys(e for e in entities if e))
    lower = norm(q)
    if any(contains(lower, x) for x in ("born", "birth", "birth years", "birthday")):
        attribute = "birth"
    elif any(contains(lower, x) for x in ("death", "die", "died")):
        attribute = "death"
    elif any(contains(lower, x) for x in ("family", "families")):
        attribute = "family"
    elif any(contains(lower, x) for x in ("frequently", "frequency", "frequencies")):
        attribute = "frequency"
    elif any(contains(lower, x) for x in ("collaborated", "collaborate", "collaboration", "partner", "partnered", "partnership")):
        attribute = "cooperation"
    elif any(contains(lower, x) for x in ("city", "cities", "country", "countries", "state", "states", "island", "islands", "province", "provinces", "located", "headquarters")):
        attribute = "origin"
    elif any(contains(lower, x) for x in ("released", "introduction", "production", "founded", "established", "launched")):
        attribute = "introduction"
    elif any(contains(lower, x) for x in ("spouse", "wife", "husband", "marry", "married")):
        attribute = "spouse"
    elif any(contains(lower, x) for x in ("administers", "administer", "department", "ministry", "portfolio")):
        attribute = "administration"
    elif any(contains(lower, x) for x in ("how many", "how much", "amount", "cost", "price", "number")):
        attribute = "quantity"
    else:
        attribute = "lexical"
    if attribute == "lexical":
        entity_words = set(re.findall(r"\w+", norm(" ".join(entities))))
        terms = [w for w in re.findall(r"[a-z]{3,}", lower)
                 if w not in QUESTION_LEADS and w not in entity_words]
    else:
        terms = list(ATTRIBUTE_CUES[attribute])
    chain = ["spouse"] if attribute == "death" and any(contains(q, x) for x in ("spouse", "wife", "husband")) else []
    return {
        "question_text": question, "entities": entities,
        "aliases": [aliases_of(e) for e in entities],
        "attribute": attribute, "attribute_terms": sorted(set(terms)),
        "relation_chain": chain, "comparison": bool(comparison),
        "parser_status": "matched" if entities and terms else "unresolved",
    }


def _sentence_spans(text):
    """Keep initials, abbreviations and parenthesized lifespan dates intact."""
    result, start, depth = [], 0, 0
    boundaries = []
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        before = text[:m.start()]
        depth = before.count("(") - before.count(")")
        if depth > 0:
            continue
        tail = before[-12:]
        if "\n" not in m.group() and (re.search(r"\b[A-Z]\.$", tail) or re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|vs|e\.g|i\.e)\.$", tail)):
            continue
        boundaries.append((m.start(), m.end()))
    for end, next_start in boundaries + [(len(text), len(text))]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            result.append((start, end))
        start = next_start
    return result


def _entity_grade(text, title, aliases, is_first):
    if any(contains(text, a) for a in aliases):
        return 3, "explicit_question_anchor"
    title_base = re.sub(r"\s*\([^()]*\)\s*$", "", title).strip()
    if not any(norm(a) in (norm(title), norm(title_base)) for a in aliases):
        return 0, "no_anchor"
    lower = norm(text)
    if re.match(r"^(?:he|she|it|they|his|her|its|their|the band|the company|the genus|the taxonomy of the genus)\b", lower):
        return 1, "visible_title_topic_anaphora"
    if is_first:
        left = re.split(r"\b(?:born|died|is|was|are|were)\b|\(", lower, maxsplit=1)[0]
        left_words = set(re.findall(r"\w+", left))
        anchor_sets = [{w for w in re.findall(r"\w+", norm(a))
                        if w not in QUESTION_LEADS} for a in aliases]
        if any(anchors and anchors <= left_words for anchors in anchor_sets):
            return 1, "visible_title_first_biography"
    return 0, "title_does_not_bind_other_subject"


def _attribute_match(text, spec):
    hits = [t for t in spec["attribute_terms"] if contains(text, t)]
    if spec["attribute"] in ("birth", "death") and _DATE_INTERVAL.search(text):
        hits.append("lifespan_date_interval")
    if spec["relation_chain"] and not any(contains(text, t) for t in ATTRIBUTE_CUES["spouse"]):
        return [], "required_spouse_chain_not_present"
    return hits, "attribute_cue" if hits else "attribute_cue_absent"


def _time_role_ok(text, aliases, grade, spec):
    if spec["attribute"] not in ("birth", "death") or spec["relation_chain"]:
        return True
    lower = norm(text)
    cue = re.search(r"\b(?:born|birth|birthday)\b" if spec["attribute"] == "birth"
                    else r"\b(?:died|death|deceased|passed away)\b", lower)
    interval = _DATE_INTERVAL.search(text)
    cue_start = cue.start() if cue else (len(norm(text[:interval.start()])) if interval else None)
    if cue_start is None:
        return False
    if grade == 3:
        ends = [m.end() for a in aliases for m in re.finditer(r"(?<!\w)" + re.escape(norm(a)) + r"(?!\w)", lower)]
        prior = [p for p in ends if p <= cue_start]
        if prior and _KINSHIP.search(lower[max(prior):cue_start]):
            return False
        # A target merely named later in another person's birth clause is weak.
        raw_cue = re.search(r"\b(?:born|birth|birthday)\b" if spec["attribute"] == "birth"
                            else r"\b(?:died|death|deceased|passed away)\b", text, re.I)
        raw_head = text[:raw_cue.start()] if raw_cue else text[:interval.start()]
        people = _proper_phrases(raw_head)
        if people:
            last_words = set(re.findall(r"\w+", norm(people[-1])))
            anchors = [{w for w in re.findall(r"\w+", norm(a)) if w not in QUESTION_LEADS} for a in aliases]
            if not any(words and words <= last_words for words in anchors):
                return False
        elif not prior and not re.match(r"^(?:he|she|it|they)\b", lower):
            return False
    elif grade == 1 and _KINSHIP.search(lower[:cue_start]):
        return False
    return True


def build_candidate_plan(tok, visible, input_token_ids):
    vis = visible_only(visible)
    if not vis["questions"]:
        raise ValueError("At least one model-visible question is required")
    messages = [{"role": "system", "content": vis["system"]},
                {"role": "user", "content": vis["prompt"]}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(input_token_ids):
        raise ValueError("Input IDs disagree with exact rendered visible prompt")
    user_start = rendered.find(vis["prompt"])
    if user_start < 0 or rendered.find(vis["prompt"], user_start + 1) >= 0:
        raise ValueError("Prompt is not uniquely located in chat template")
    marker = vis["prompt"].find(SEARCH_MARKER)
    if marker < 0:
        raise ValueError("Expected literal Search results marker")
    question_marker = "\n\nQuestions:\n"
    question_start = vis["prompt"].find(question_marker)
    expected_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(vis["questions"]))
    if question_start < 0 or vis["prompt"][question_start + len(question_marker):marker] != expected_questions:
        raise ValueError("Question list differs from model-visible question section")
    expected_sources = "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(vis["passages"]))
    if vis["prompt"][marker + len(SEARCH_MARKER):] != expected_sources:
        raise ValueError("Displayed source body/list differs from prompt")
    source_start = user_start + marker + len(SEARCH_MARKER)
    source_end = user_start + len(vis["prompt"])
    offsets = encoded["offset_mapping"]
    context = [i for i, (a, b) in enumerate(offsets)
               if b > a and b > source_start and a < source_end]
    units, title_ranges, cursor = [], [], source_start
    for passage_index, passage in enumerate(vis["passages"]):
        header = f"[{passage_index + 1}] {passage['title']}\n"
        hs = rendered.find(header, cursor, source_end)
        if hs < 0:
            raise ValueError("Displayed passage header absent or out of order")
        body_start = hs + len(header)
        if rendered[body_start:body_start + len(passage["text"])] != passage["text"]:
            raise ValueError("Displayed source body differs from prompt")
        title_ranges.append((hs, body_start))
        for j, (a, b) in enumerate(_sentence_spans(passage["text"])):
            units.append({"unit_index": len(units), "passage_index": passage_index,
                          "title": passage["title"], "text": passage["text"][a:b],
                          "start": body_start + a, "end": body_start + b,
                          "first_in_passage": j == 0, "token_indices": []})
        cursor = body_start + len(passage["text"])
    body = []
    for i in context:
        a, b = offsets[i]
        candidates = [(sum(not c.isspace() for c in rendered[max(a,u["start"]):min(b,u["end"])]), -j)
                      for j, u in enumerate(units) if b > u["start"] and a < u["end"]]
        candidates = [x for x in candidates if x[0] > 0]
        if candidates:
            _, minusj = max(candidates)
            units[-minusj]["token_indices"].append(i)
            body.append(i)
    body_set = set(body)
    plans = []
    for qi, question in enumerate(vis["questions"]):
        spec = question_spec(question)
        # Only explicitly written aliases linked to a question anchor are added.
        for aliases in spec["aliases"]:
            extra = []
            for unit in units:
                for alias in list(aliases):
                    pattern = re.escape(alias) + r"\s*[,()]?\s*(?:also known as|known as|also called)\s+([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,5})"
                    extra.extend(m.group(1).strip() for m in re.finditer(pattern, unit["text"]))
            aliases.extend(a for a in extra if a not in aliases)
        selected, entity_units, attribute_units, trace = [], set(), set(), []
        for aliases in spec["aliases"]:
            ranked = []
            for u in units:
                grade, er = _entity_grade(u["text"], u["title"], aliases, u["first_in_passage"])
                terms, ar = _attribute_match(u["text"], spec)
                role_ok = _time_role_ok(u["text"], aliases, grade, spec)
                if grade:
                    entity_units.add(u["unit_index"])
                if terms:
                    attribute_units.add(u["unit_index"])
                if grade and terms and role_ok:
                    ranked.append((grade, len(terms), -u["unit_index"]))
                trace.append({"entity_aliases": list(aliases), "unit_index": u["unit_index"],
                              "entity_grade": grade, "entity_rule": er,
                              "attribute_hits": terms, "attribute_rule": ar,
                              "local_role_compatible": role_ok})
            chosen = [-x[2] for x in sorted(ranked, reverse=True)[:3]]
            selected.append(sorted({k for j in chosen for k in units[j]["token_indices"]}))
        # Attribute matches are also needed when the question anchor was absent.
        for u in units:
            if _attribute_match(u["text"], spec)[0]:
                attribute_units.add(u["unit_index"])
        target = set(k for group in selected for k in group)
        entity = set(k for j in entity_units for k in units[j]["token_indices"]) - target
        attr = set(k for j in attribute_units - entity_units for k in units[j]["token_indices"]) - target
        other = body_set - target - entity - attr
        plans.append({
            "question_index": qi, **spec,
            "target_token_indices_by_entity": selected,
            "target_union_token_indices": sorted(target),
            "entity_only_token_indices": sorted(entity),
            "attribute_distractor_token_indices": sorted(attr),
            "other_body_token_indices": sorted(other), "matching_trace": trace,
        })
    return {
        "version": VERSION, "visible_sha256": digest(vis),
        "rendered_prompt_sha256": digest(rendered),
        "input_token_ids_sha256": digest(list(input_token_ids)),
        "input_tokens": len(input_token_ids), "context_token_indices": context,
        "body_token_indices": body,
        "title_token_indices": [i for i in context if i not in body_set],
        "sentence_units": units, "question_plans": plans,
        "candidate_definition": "Question-derived lexical/role candidates; not gold evidence or entailment",
        "candidate_cap_per_entity": 3,
    }


def causal_question_routes(tok, response_ids, question_count):
    """Incremental byte decode: no complete final answer or later numbering."""
    if question_count == 1:
        return np.zeros(len(response_ids), dtype=np.int32), np.ones(len(response_ids), dtype=bool)
    decoder = {value: key for key, value in bytes_to_unicode().items()}
    prefix = b""
    routes, valid = [], []
    current, seen = -1, []
    consumed = 0
    marker = re.compile(r"(?m)^[ \t]*(?:\*\*|__)?[ \t]*(\d+)[.)][ \t]*(?:\*\*|__)?")
    for tid in response_ids:
        piece = bytes(decoder[c] for c in tok.convert_ids_to_tokens(int(tid)))
        prefix += piece
        text = prefix.decode("utf-8", errors="ignore")
        matches = list(marker.finditer(text))
        for m in matches[consumed:]:
            index = int(m.group(1)) - 1
            # Invalid/repeated/backward indices remain explicit, never repaired
            # using knowledge that a later complete answer has valid numbering.
            current = index if 0 <= index < question_count and index not in seen and (not seen or index > seen[-1]) else -1
            seen.append(index)
        consumed = len(matches)
        routes.append(current)
        valid.append(current >= 0)
    return np.asarray(routes, dtype=np.int32), np.asarray(valid, dtype=bool)


def binding_from_attention(attention, group_masks, body_positions, response_start,
                           query_positions, route_valid, pair_valid):
    """Pure tensor formulas; H,Q,S attention and Q,5,S masks [T,E,A,T1,T2]."""
    eps = torch.finfo(torch.float32).tiny
    counts = group_masks.sum(-1)
    masses = torch.einsum("hqs,qgs->hqg", attention, group_masks)
    body_mass = attention.index_select(-1, body_positions).sum(-1) if len(body_positions) else torch.zeros_like(masses[..., 0])
    body_count = len(body_positions)
    keys = torch.arange(attention.shape[-1], device=attention.device)
    prev_mask = (keys[None] >= response_start) & (keys[None] < query_positions[:, None])
    prev_count = prev_mask.sum(-1)
    prev_mass = (attention * prev_mask[None]).sum(-1)
    td = masses[..., 0] / counts[:, 0].clamp_min(1)[None]
    od = (body_mass - masses[..., 0]).clamp_min(0) / (body_count - counts[:, 0]).clamp_min(1)[None]
    pd = prev_mass / prev_count.clamp_min(1)[None]
    pair_lo = torch.minimum(masses[..., 3], masses[..., 4])
    pair_hi = torch.maximum(masses[..., 3], masses[..., 4])
    self_mass = attention.gather(-1, query_positions[None, :, None].expand(attention.shape[0], -1, -1)).squeeze(-1)
    values = torch.stack([
        body_mass, masses[..., 0] / body_mass.clamp_min(eps),
        td / (td + od).clamp_min(eps), td / (td + pd).clamp_min(eps),
        masses[..., 1] / body_mass.clamp_min(eps),
        masses[..., 2] / body_mass.clamp_min(eps), self_mass,
        pair_lo / pair_hi.clamp_min(eps),
    ], dim=-1)
    valid = torch.stack([
        torch.full_like(route_valid, body_count > 0),
        route_valid & (counts[:, 0] > 0) & (body_count > 0),
        route_valid & (counts[:, 0] > 0) & (body_count > counts[:, 0]),
        route_valid & (counts[:, 0] > 0) & (prev_count > 0),
        route_valid & (body_count > 0),
        route_valid & (body_count > 0),
        torch.ones_like(route_valid),
        route_valid & pair_valid & (counts[:, 3] > 0) & (counts[:, 4] > 0),
    ], dim=-1)
    values = values * valid[None].to(values.dtype)
    return values, valid


class _BindingHooks:
    def __init__(self, model, plan, routes, route_valid, response_count, query_batch):
        self.model, self.plan = model, plan
        self.response_count, self.query_batch = response_count, query_batch
        self.start, self.layers, self.heads = plan["input_tokens"], len(model.model.layers), model.config.num_attention_heads
        self.device = model.model.embed_tokens.weight.device
        self.positions = torch.arange(self.start, self.start + response_count, device=self.device)
        self.context = torch.as_tensor(plan["context_token_indices"], device=self.device, dtype=torch.long)
        self.body = torch.as_tensor(plan["body_token_indices"], device=self.device, dtype=torch.long)
        self.routes = torch.as_tensor(routes, device=self.device, dtype=torch.long)
        self.route_valid = torch.as_tensor(route_valid, device=self.device, dtype=torch.bool)
        total = self.start + response_count
        masks = torch.zeros(len(plan["question_plans"]), 5, total, device=self.device, dtype=torch.float32)
        pairs = []
        for qi, p in enumerate(plan["question_plans"]):
            sets = [p["target_union_token_indices"], p["entity_only_token_indices"], p["attribute_distractor_token_indices"]]
            sets += (p["target_token_indices_by_entity"] + [[], []])[:2]
            for gi, ids in enumerate(sets):
                masks[qi, gi, ids] = 1
            pairs.append(p["comparison"] and len(p["target_token_indices_by_entity"]) == 2)
        self.masks, self.pairs = masks, torch.as_tensor(pairs, device=self.device, dtype=torch.bool)
        self.lookback = np.empty((response_count, self.layers, self.heads), dtype=np.float32)
        self.lowdim = np.zeros((response_count, 4, len(CHANNELS)), dtype=np.float32)
        self.valid = np.zeros((response_count, len(CHANNELS)), dtype=bool)
        self.hidden21, self.pending, self.handles, self.seen = None, {}, [], set()

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            attn = block.self_attn
            def before(module, args, kwargs, layer=layer):
                if kwargs.get("past_key_value") is not None:
                    raise ValueError("Core replay must disable KV cache")
                self.pending[layer] = {"rope": kwargs["position_embeddings"]}
            def query(module, args, output, layer=layer):
                self.pending[layer]["q"] = output
            def key(module, args, output, layer=layer, attn=attn):
                self._attention(layer, attn, output)
            self.handles += [
                attn.register_forward_pre_hook(before, with_kwargs=True),
                attn.q_proj.register_forward_hook(query),
                attn.k_proj.register_forward_hook(key),
            ]
        def h21(module, args, result):
            value = result[0] if isinstance(result, tuple) else result
            self.hidden21 = value[0, self.positions].detach().float().cpu().numpy()
        self.handles.append(self.model.model.layers[20].register_forward_hook(h21))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.pending.clear()

    def _attention(self, layer, module, projection):
        state = self.pending.pop(layer)
        dim = module.head_dim
        q_all = state["q"].view(1, -1, self.heads, dim).transpose(1, 2)
        k = projection.view(1, -1, self.model.config.num_key_value_heads, dim).transpose(1, 2)
        cos, sin = state["rope"]
        k = repeat_kv(k * cos[:, None] + rotate_half(k) * sin[:, None], module.num_key_value_groups)[0]
        for begin in range(0, self.response_count, self.query_batch):
            end = min(begin + self.query_batch, self.response_count)
            pos = self.positions[begin:end]
            q = q_all[0].index_select(1, pos)
            qc, qs = cos[0].index_select(0, pos), sin[0].index_select(0, pos)
            q = q * qc[None] + rotate_half(q) * qs[None]
            attn = causal_attention_rows(q, k, pos, module.scaling)
            if len(self.context):
                lb = lookback_from_attention(attn, self.context, self.start, pos)
            else:
                lb = torch.zeros(attn.shape[:2], device=self.device)
            qi = self.routes[begin:end].clamp_min(0)
            rv = self.route_valid[begin:end]
            masks = self.masks.index_select(0, qi) * rv[:, None, None]
            head, valid = binding_from_attention(attn, masks, self.body, self.start, pos, rv, self.pairs.index_select(0, qi))
            self.lookback[begin:end, layer] = lb.T.cpu().numpy()
            self.lowdim[begin:end, layer // 7] += head.mean(0).cpu().numpy() / 7.0
            self.valid[begin:end] = valid.cpu().numpy()
        self.seen.add(layer)


@torch.inference_mode()
def extract_binding_features(tok, model, visible, generated):
    """One replay, fixed 32D binding features; no detector fitting or labels."""
    started = time.perf_counter()
    vis = visible_only(visible)
    if model.training or model.config.model_type != "qwen2" or len(model.model.layers) != 28:
        raise ValueError("Frozen 28-layer Qwen2 model required")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("Sliding-window masks not supported")
    prefix = [int(x) for x in generated["input_token_ids"]]
    answer = [int(x) for x in generated["response_token_ids"]]
    text = generated["response"]
    if not answer:
        raise ValueError("Empty generated answer has no token features")
    offsets = _response_offsets(tok, answer, text)
    if "response_token_offsets" in generated and not np.array_equal(offsets, generated["response_token_offsets"]):
        raise ValueError("Exact generated offsets disagree")
    if generated.get("unexpected_special_token_ids"):
        raise ValueError("Unexpected interior special token")
    plan = build_candidate_plan(tok, vis, prefix)
    routes, route_valid = causal_question_routes(tok, answer, len(vis["questions"]))
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([prefix + answer], device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - started
    replay_start = time.perf_counter()
    with _BindingHooks(model, plan, routes, route_valid, len(answer), 8) as hooks:
        output = model.model(input_ids=ids, use_cache=False, output_attentions=False, output_hidden_states=False)
    if len(hooks.seen) != 28 or hooks.hidden21 is None:
        raise RuntimeError("Missing layer hook signal")
    pos = torch.arange(len(prefix), len(prefix) + len(answer), device=device)
    last = output.last_hidden_state[0]
    hidden28 = last.index_select(0, pos).float().cpu().numpy()
    nll, entropy = [], []
    for positions in pos.split(16):
        lp = model.lm_head(last.index_select(0, positions - 1)).float().log_softmax(-1)
        nll.extend((-lp.gather(1, ids[0, positions, None])).squeeze(-1).cpu().tolist())
        entropy.extend((-(lp.exp() * lp).sum(-1)).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    arrays = {
        "hidden_28": hidden28, "hidden_21": hooks.hidden21,
        "lookback_features": hooks.lookback.reshape(len(answer), -1),
        "binding_features": hooks.lowdim.reshape(len(answer), -1),
        "binding_valid": hooks.valid,
        "token_nll": np.asarray(nll, dtype=np.float32),
        "token_entropy": np.asarray(entropy, dtype=np.float32),
        "token_ids": np.asarray(answer, dtype=np.int64),
        "token_start": offsets[:, 0], "token_end": offsets[:, 1],
        "response_token_offsets": offsets,
        "token_question_index": routes, "token_route_valid": route_valid,
    }
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite feature; do not silently replace")
    names = [f"layers_{a}_{b}__{channel}" for a,b in BANDS for channel in CHANNELS]
    meta = {
        "version": VERSION, "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attention7_source_sha256": hashlib.sha256((R7_SRC / "attention7.py").read_bytes()).hexdigest(),
        "visible_input_sha256": digest(vis), "generated_ids_sha256": digest(answer),
        "response_sha256": digest(text), "candidate_plan": plan,
        "binding_feature_names": names, "binding_channel_names": list(CHANNELS),
        "binding_dimensions": len(names), "layer_bands": [list(x) for x in BANDS],
        "lookback_axes": "token x layer-major/head-minor",
        "feature_timing": "post-read P+j; binding prev excludes j; original globalLB includes j; no future answer selection",
        "probability_timing": "predict observed token j from final hidden at P+j-1",
        "causal_passes": 1, "query_batch": 8, "logit_batch": 16,
        "preparation_seconds": preparation_seconds, "replay_seconds": time.perf_counter() - replay_start,
        "seconds": time.perf_counter() - started,
        "route_invalid_tokens": int((~route_valid).sum()),
        "target_candidate_missing_questions": sum(not p["target_union_token_indices"] for p in plan["question_plans"]),
        "question_parser_unresolved": sum(p["parser_status"] != "matched" for p in plan["question_plans"]),
        "labels_or_detector_scores_read": False,
        "hidden_metadata_used": False, "full_future_output_used_for_candidate_selection": False,
        "hidden_21_definition": "after decoder block 21 before final RMSNorm",
        "hidden_28_definition": "after decoder block 28 and final RMSNorm",
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None,
    }
    return arrays, meta
