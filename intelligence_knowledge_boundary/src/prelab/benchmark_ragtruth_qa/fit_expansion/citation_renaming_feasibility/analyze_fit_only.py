"""Read-only mechanical feasibility of citation ID renaming in existing fit data.

No transformed dataset is exported. Eligibility is conservative syntactic
screening, not a new human validation of semantic label invariance.
"""
from pathlib import Path
from collections import Counter
import hashlib
import itertools
import json
import re

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "data/fit.jsonl"
HEADER = re.compile(r"(?m)^passage ([123]):")
SINGLE = re.compile(r"\bpassage[ \t]+([123])(?!\w)", re.I)
WORD_ID = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|I|II|III)\b"
ORDINAL = r"(?:first|second|third|fourth|fifth|last|final|1st|2nd|3rd)"
POSITION = re.compile(rf"\b{ORDINAL}[ \t]+(?:(?:one|two|three|\d)[ \t]+)?(?:passages?|documents?|sources?|excerpts?)\b|\b(?:passages?|documents?|sources?|excerpts?)[ \t]+{WORD_ID}", re.I)
NUMBERED = re.compile(r"\b(?:passages?|documents?|sources?|excerpts?|references?)[ \t]*(?:[#(\[]?[ \t]*)?\d+", re.I)
BRACKET = re.compile(r"\[[ \t]*\d+(?:[ \t]*[-,–—][ \t]*\d+)*[ \t]*\]")
PLURAL = re.compile(rf"\bpassages[ \t]+(?:\d+|{WORD_ID})", re.I)
RANGE = re.compile(r"\bpassages?[ \t]+\d+[ \t]*(?:[-–—/]|to\b|through\b)[ \t]*\d+", re.I)
COMPOSITE = re.compile(r"\bpassage[ \t]+\d+[ \t]*(?:[,/&]|and\b|or\b)[ \t]*(?:passage[ \t]+)?\d+", re.I)
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`')
SOURCE_CITATION = re.compile("(?:" + NUMBERED.pattern + ")|(?:" + POSITION.pattern + ")|(?:" + BRACKET.pattern + ")", re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def edit_chars(text, locations, mapping):
    result = list(text)
    for index in locations:
        assert result[index] in mapping
        result[index] = mapping[result[index]]
    return "".join(result)


def snippet(text, start, end, radius=50):
    return text[max(0, start-radius):min(len(text), end+radius)]


def main():
    original_hash = sha(SOURCE)
    answers = [json.loads(s) for s in SOURCE.read_text(encoding="utf-8").splitlines() if s]
    assert len(answers) == 3680
    assert all(x["partition"] == "fit" and x["official_split"] == "train" and x["quality"] == "good" for x in answers)
    fmt_counts, reason_counts = Counter(), Counter()
    records, eligible, examples, failures = [], [], [], {}
    any_target = any_overlap = metadata_citation_answers = 0
    label_total = label_checked = changed_digit_total = 0
    no_citation_extra_safety = 0
    for x in answers:
        response = x["original_response"]
        context = x["retrieved_passages"]
        question = x["question"]
        targets = list(SINGLE.finditer(response))
        fmt_counts["written_number_reference"] += bool(re.search(rf"\bpassages?[ \t]+{WORD_ID}", response, re.I))
        fmt_counts["ordinal_or_position_reference"] += bool(re.search(rf"\b{ORDINAL}[ \t]+(?:(?:one|two|three|\d)[ \t]+)?(?:passages?|documents?|sources?|excerpts?)\b", response, re.I))
        offsets = [m.start(1) for m in targets]
        headers = list(HEADER.finditer(context))
        any_target += bool(targets)
        for label in x["labels"]:
            label_total += 1
            assert 0 <= label["start"] < label["end"] <= len(response)
            assert response[label["start"]:label["end"]] == label["text"]
            label_checked += 1
        reasons, details = [], {}

        def block(name, evidence=None):
            if name not in reasons:
                reasons.append(name)
                if evidence is not None:
                    details[name] = evidence

        if not targets:
            block("no_explicit_single_digit_citation")
        if [m.group(1) for m in headers] != ["1", "2", "3"]:
            block("noncanonical_context_headers")
        if x.get("released_prompt", "").count(context) != 1:
            block("released_prompt_context_not_unique")
        body_only = HEADER.sub("", context)
        # The source body and question are never rewritten. Even unrelated-looking
        # document footnotes are excluded conservatively, not claimed to be wrong.
        for text, field in ((body_only, "source_body_citation_or_ordinal"), (question, "question_citation_or_ordinal")):
            m = SOURCE_CITATION.search(text)
            if m:
                block(field, snippet(text, m.start(), m.end()))
        for pattern, name in ((PLURAL, "plural_reference"), (RANGE, "range_reference"),
                              (COMPOSITE, "compound_reference"), (POSITION, "written_or_position_reference"),
                              (BRACKET, "bare_bracket_reference")):
            m = pattern.search(response)
            if m:
                fmt_counts[name] += 1
                block(name, snippet(response, m.start(), m.end()))
        spans = {(m.start(), m.end()) for m in targets}
        for m in NUMBERED.finditer(response):
            if (m.start(), m.end()) not in spans:
                block("other_numeric_reference_format", snippet(response, m.start(), m.end()))
        overlap = [p for p in offsets if any(l["start"] <= p < l["end"] for l in x["labels"])]
        any_overlap += bool(overlap)
        # Gold fields NEVER determine augmentation eligibility. Wrong citations
        # are precisely an intended use case. These counts are descriptive only.
        metadata_citation_answers += any(SOURCE_CITATION.search(str(label.get("meta", ""))) for label in x["labels"])
        quote = next((m for m in QUOTED.finditer(response) if any(m.start() <= p < m.end() for p in offsets)), None)
        if quote:
            block("target_inside_quote_or_inline_code", snippet(response, quote.start(), quote.end()))
        if "```" in response and targets:
            block("answer_code_fence_requires_manual_scope_check")
        if any(re.search(r"[.,:]\d", response[p:p+3]) for p in offsets):
            block("numeric_continuation_requires_manual_scope_check")
        fmt_counts["answers_with_explicit_single"] += bool(targets)
        fmt_counts["explicit_single_occurrences"] += len(targets)
        if not targets and len(reasons) > 1:
            no_citation_extra_safety += 1
        passed = not reasons
        checked_permutations = 0
        if passed:
            for perm in itertools.permutations("123"):
                if perm == tuple("123"):
                    continue
                mapping = dict(zip("123", perm))
                renamed_response = edit_chars(response, offsets, mapping)
                renamed_context = edit_chars(context, [m.start(1) for m in headers], mapping)
                assert len(renamed_response) == len(response) and len(renamed_context) == len(context)
                assert all(a == b or i in offsets for i, (a, b) in enumerate(zip(response, renamed_response)))
                assert HEADER.sub("", renamed_context) == body_only
                for old_m, new_m in zip(targets, SINGLE.finditer(renamed_response)):
                    assert old_m.span(1) == new_m.span(1)
                    assert new_m.group(1) == mapping[old_m.group(1)]
                for label in x["labels"]:
                    relative = [p-label["start"] for p in offsets if label["start"] <= p < label["end"]]
                    derived_text = renamed_response[label["start"]:label["end"]]
                    assert derived_text == edit_chars(label["text"], relative, mapping)
                    assert len(derived_text) == label["end"]-label["start"]
                # Mapping a wrong source reference through the same bijection as
                # headers preserves that wrong link; never look up a correct ID.
                assert len(set(mapping.values())) == 3
                checked_permutations += 1
            eligible.append(x)
            changed_digit_total += len(offsets)
            if len(examples) < 6:
                m = targets[0]
                mapping = {"1": "2", "2": "3", "3": "1"}
                altered = edit_chars(response, offsets, mapping)
                examples.append({"response_id": x["response_id"], "source_id": x["source_id"],
                                 "original_risk": int(bool(x["labels"])), "question": question,
                                 "answer_excerpt_before": snippet(response, m.start(), m.end()),
                                 "answer_excerpt_after": snippet(altered, m.start(), m.end()),
                                 "example_mapping": mapping,
                                 "context_headers_after_in_original_body_order": [mapping["1"], mapping["2"], mapping["3"]],
                                 "original_label_provenance_untouched": True,
                                 "derived_span_text_uses_original_coordinates": True,
                                 "citation_digit_overlaps_original_gold": bool(overlap)})
        for reason in reasons:
            reason_counts[reason] += 1
            if reason not in failures and reason in details:
                failures[reason] = {"response_id": x["response_id"], "example": details[reason], "has_explicit_single": bool(targets)}
        records.append({"response_id": x["response_id"], "source_id": x["source_id"], "group_id": x["group_id"],
                        "strict_mechanical_candidate": passed, "explicit_single_count": len(targets),
                        "original_risk": int(bool(x["labels"])), "keep_original_reasons": reasons,
                        "citation_digit_overlaps_original_gold_descriptive_only": bool(overlap),
                        "five_bijections_in_memory_checked": checked_permutations == 5})
    assert sha(SOURCE) == original_hash
    for reason in ("plural_reference", "range_reference", "compound_reference", "written_or_position_reference", "bare_bracket_reference"):
        fmt_counts.setdefault(reason, 0)
    for reason in ("noncanonical_context_headers", "released_prompt_context_not_unique", "question_citation_or_ordinal", "bare_bracket_reference"):
        reason_counts.setdefault(reason, 0)
    # This concrete wrong-citation example was read against all three fit source
    # passages. Include it as an illustration, not as a selection criterion.
    wrong = next(x for x in answers if x["response_id"] == "15340")
    wrong_record = next(r for r in records if r["response_id"] == "15340")
    assert wrong_record["strict_mechanical_candidate"]
    mapping = {"1": "2", "2": "3", "3": "1"}
    wrong_new = edit_chars(wrong["original_response"], [m.start(1) for m in SINGLE.finditer(wrong["original_response"])], mapping)
    wrong_label = wrong["labels"][0]
    wrong_example = {"response_id": "15340", "source_id": wrong["source_id"],
                     "before_risk_span_text": wrong_label["text"],
                     "in_memory_derived_risk_span_text": wrong_new[wrong_label["start"]:wrong_label["end"]],
                     "unchanged_start": wrong_label["start"], "unchanged_end": wrong_label["end"],
                     "unchanged_label_type": wrong_label["label_type"], "mapping": mapping,
                     "original_answer_cited": "1", "source_supporting_passage_original": "2",
                     "renamed_answer_still_wrongly_cites": "2", "source_supporting_passage_renamed": "3",
                     "original_annotation_and_meta_untouched": True,
                     "interpretation": "The same incorrect link remains incorrect; this is not an automated citation correction."}
    counts = {"answers": len(answers), "source_ids": len({x["source_id"] for x in answers}),
              "groups": len({x["group_id"] for x in answers}),
              "strict_mechanical_candidates": len(eligible), "strict_candidate_sources": len({x["source_id"] for x in eligible}),
              "strict_candidate_groups": len({x["group_id"] for x in eligible}),
              "strict_candidate_citation_occurrences": changed_digit_total,
              "strict_candidate_original_risk_answers": sum(bool(x["labels"]) for x in eligible),
              "strict_candidate_original_no_risk_answers": sum(not x["labels"] for x in eligible),
              "unchanged_answers": len(answers)-len(eligible),
              "answers_with_explicit_single": any_target, "answers_without_explicit_single": len(answers)-any_target,
              "explicit_single_overlaps_gold_answers": any_overlap,
              "explicit_single_without_gold_overlap_answers": any_target-any_overlap,
              "strict_candidates_with_gold_overlap": sum(r['strict_mechanical_candidate'] and r['citation_digit_overlaps_original_gold_descriptive_only'] for r in records),
              "annotation_metadata_citation_answers_descriptive_only": metadata_citation_answers,
              "candidate_rejected_despite_explicit_single": sum(bool(r["explicit_single_count"]) and not r["strict_mechanical_candidate"] for r in records),
              "original_label_spans_checked": label_checked, "original_label_alignment_errors": label_total-label_checked,
              "no_explicit_citation_but_other_exclusion_also_detected": no_citation_extra_safety}
    result = {"status": "complete_fit_only_feasibility", "source_file": str(SOURCE.resolve()), "source_sha256": original_hash,
              "script_sha256": sha(__file__), "counts": counts, "format_counts": dict(fmt_counts),
              "keep_original_reason_counts_overlapping": dict(reason_counts), "strict_candidates_by_model_metadata_only": dict(Counter(x["model"] for x in eligible)),
              "examples": examples, "wrong_citation_preservation_example": wrong_example, "keep_original_examples": failures,
              "records": records,
              "checks": {"only_existing_fit_read": True, "source_unchanged": True, "all_labels_original_text_aligned": True,
                         "every_strict_candidate_all_five_nonidentity_bijections_checked": True,
                         "source_bodies_and_order_unchanged_in_memory": True, "answer_char_length_and_label_coordinates_unchanged": True,
                         "transformed_span_text_derived_from_same_coordinates_in_memory": True,
                         "gold_or_annotation_meta_used_for_eligibility": False, "original_human_provenance_unchanged": True,
                         "exported_training_data": False, "training_started": False, "GPU_used": False,
                         "calibration_or_test_opened": False, "human_semantic_invariance_validated": False,
                         "tokenization_invariance_claimed": False},
              "limits": ["These counts certify only a conservative, explicit character-edit precheck, not human revalidation of label semantics.",
                         "Risk is relative to source support; preserve existing wrong source links by the same bijection, never correct citations from gold.",
                         "A future renamed example would be a transformation derived from human annotation, not newly human-annotated gold. Original annotation and meta remain immutable provenance; its current span.text is derived at the same start/end in the renamed answer.",
                         "Character length invariance does not imply tokenizer ID, token count, raw-4-BPE windows or cached hidden-state invariance. Future training must rebuild token geometry and feature caches from the renamed strings.",
                         "Refused/out-of-range/multi-citation rows are kept unchanged, not removed from any future training set.",
                         "Never globally replace 1/2/3, factual quantities, dates, numbered-list markers, source body footnotes or question text.",
                         "Only one-character source-ID substitutions at canonical headers and explicit answer references are considered. No passage permutation or unsupported-to-supported relabeling."]}
    dump(OUT / "FEASIBILITY.json", result)
    lines = ["# 训练集引用编号重命名：可行性检查", "",
             f"仅读取现有 fit 的 {counts['answers']:,} 条回答；其中 {any_target} 条含明确的 `passage 1/2/3`，共 {fmt_counts['explicit_single_occurrences']:,} 处。按下面的保守规则，**{len(eligible)} 条通过字符级预检**，涉及 {counts['strict_candidate_sources']} 个来源、{counts['strict_candidate_groups']} 个资料组。其余 {counts['unchanged_answers']} 条保持原样。", "",
             f"通过项含 {counts['strict_candidate_original_risk_answers']} 条原风险回答、{counts['strict_candidate_original_no_risk_answers']} 条原无风险回答；不因筛选而删除任何训练答。", "",
             "仅允许资料头 `passage 1:`、`passage 2:`、`passage 3:` 与回答中独立、明确的同名单数字引用，按同一个一一映射改一个字符。例如 1→2、2→3、3→1；正文位置完全不动。编号列表、事实数字不改，也不把错误引用修成正确引用。", "",
             "| 保持原样的原因（可重叠） | 回答数 |", "|---|---:|"]
    names = {"no_explicit_single_digit_citation": "没有明确单数字引用（本检查不做只改资料头）",
             "noncanonical_context_headers": "资料头不满足唯一的1/2/3",
             "released_prompt_context_not_unique": "原提示词中的资料块不唯一",
             "source_body_citation_or_ordinal": "资料正文已有引用/脚注编号",
             "question_citation_or_ordinal": "问题依赖编号/位置",
             "plural_reference": "复数引用（如passages 1 and 2）", "range_reference": "范围引用（如passages 1–4）",
             "compound_reference": "单数开头的复合引用", "written_or_position_reference": "拼写数字或序数/位置引用",
             "bare_bracket_reference": "无来源名的方括号编号", "other_numeric_reference_format": "超出1/2/3或非白名单的引用格式",
             "target_inside_quote_or_inline_code": "目标编号位于引号/行内代码内",
             "answer_code_fence_requires_manual_scope_check": "回答含代码围栏，引用范围需人审",
             "numeric_continuation_requires_manual_scope_check": "编号后连数字表达，需人审"}
    for reason, count in reason_counts.items():
        lines.append(f"| {names[reason]} | {count} |")
    lines += ["", f"有 {any_overlap} 条回答的引用数字位于人标错误范围内，**不以此排除**；其中 {counts['strict_candidates_with_gold_overlap']} 条通过格式规则。标签只用于最后的描述和坐标校验，完全不参与适用性筛选。未来派生样本保持类别/start/end，span.text 从变换后的答案同坐标取出；原人标和meta另保留为不可变来源记录。", "",
              "已亲读的错引例子（训练答15340）：答案写 `Passage 1 mentions ... reduce muscle pain ...`，对应内容实际在资料2。按1→2、2→3、3→1改名后，答案引用资料2，内容实际在资料3，仍然是错引；原错误类别及坐标421–502保持，派生span.text按这个范围取新文字。", "",
              "每个通过项已在内存检查全部5种非恒等的一一映射：资料正文/顺序不变，回答长度不变，全部派生span按原坐标准确对齐。原错引A、实际属于B，重命名后仍错引P(A)、实际属于P(B)，不按gold纠正。未来若导出，应明确称为‘从人标派生的编号一致变换’，不能称重新人标。**这只是机械预检，不等于已由人工验证保标签。** 单字符替换也不保证分词和4词元窗口不变，后续若实施必须重建字符到token映射，不能复用旧特征。", "",
              "本次只保存计数、候选ID和少量片段样例；未导出新训练集，未读取cal/test，未启动训练，也未改变当前v2协议或数据。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "format_counts": dict(fmt_counts), "reasons": dict(reason_counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
