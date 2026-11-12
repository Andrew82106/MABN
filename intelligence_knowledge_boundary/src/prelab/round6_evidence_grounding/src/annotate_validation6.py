"""Serialize root's source-blind-to-scores manual review of the six validation groups.

These explicit decisions were made after reading every supplied paragraph and
actual answer. No coverage field, feature or detector prediction is consulted.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def readl(name):
    return [json.loads(s) for s in (ROOT / name).read_text(encoding='utf-8').splitlines()]


# Each entry is (decision, rationale, multiple_facts). S=supported/correct,
# U=unsupported/correct, W=unsupported/incorrect, A=pure abstention,
# X=unresolved scope, Y=unresolved support but reference-correct.
REVIEWS = {
    118: {
        'complete': [
            ('S', 'Skin Yard introduction explicitly states American and Seattle, Washington.', False),
            ('S', 'Ostava introduction explicitly states Bulgaria.', False),
            ('S', 'The same item explicitly explains Skin Yard is US and Ostava is Bulgarian, resolving its initial negation as not-both. Both attributes and the comparison are supported.', True)],
        'partial': [
            ('S', 'Retained Skin Yard and Jack Endino passages establish US origin.', False),
            ('A', 'Reports absence of Ostava information without asserting an origin; all supplied passages inspected and none discusses Ostava.', False),
            ('U', 'Explicit No asserts the not-both-US comparison, while its own explanation acknowledges Ostava origin is unknown. Skin Yard US is supported, but the comparison is not. Complete reference says Ostava Bulgaria, so the comparison happens to be correct.', True)]},
    166: {
        'complete': [
            ('S', 'Dinosaur film introduction gives 2000.', False),
            ('S', 'McFarland USA introduction gives 2015.', False),
            ('S', 'Both stated release years and the 2000-before-2015 ordering match the two visible film passages.', True)],
        'partial': [
            ('S', 'Dinosaur 2000 is still explicit in the retained film passage.', False),
            ('U', 'None of the four supplied passages gives McFarland release year; the omitted full reference supports 2015.', False),
            ('U', 'Dinosaur 2000 is supported, but saying it came first needs the missing McFarland year. The complete reference confirms the ordering.', True)]},
    229: {
        'complete': [
            ('S', 'Marsilea introduction explicitly gives approximately 65 species.', False),
            ('S', 'Brabejum introduction explicitly says single species.', False),
            ('S', 'Approximately 65 is more than one, directly supported by both genus passages.', False)],
        'partial': [
            ('Y', 'Initial blind label unsupported was reconsidered before any scoring: the question may presuppose one of the two genera strictly has more species; with Brabejum single-species and a genus having at least one species, this can indirectly entail Marsilea multiple. Support is therefore unresolved, not a clean missing-evidence positive. Full reference approximately65 confirms multiple; exact count remains unanswered.', False),
            ('S', 'Retained Brabejum source gives one species.', False),
            ('Y', 'Initial blind label unsupported was reconsidered before scoring: the forced-choice more-species question may presuppose strict inequality, and Brabejum single-species then indirectly identifies Marsilea as larger. Keep as a data-construction ambiguity, exclude from primary binary evaluation. Complete reference confirms the actual ordering.', False)]},
    289: {
        'complete': [
            ('S', 'The reunited first two Hey Pa source sentences give issued1966.', False),
            ('S', 'Poleconomy introduction gives first published1980.', False),
            ('S', 'Both visible publication years support Hey Pa coming first.', False)],
        'partial': [
            ('S', 'Retained Hey Pa passage explicitly gives1966.', False),
            ('A', 'Accurately reports no Poleconomy publication year in the supplied results and asserts no year.', False),
            ('U', 'The item asserts came to market first despite the previous abstention. Hey Pa1966 alone does not establish comparison with Poleconomy; complete reference1980 makes the ordering correct.', True)]},
    318: {
        'complete': [
            ('S', 'Duke introduction gives Charlotte, North Carolina.', False),
            ('S', 'Affiliated Managers Group introduction gives Massachusetts.', False),
            ('X', 'No, both A and B are not based in Massachusetts has distributive versus not-both scope ambiguity, without an explanation inside this item. The former conflicts with AMG Massachusetts and the latter is supported. Preserve ambiguity under the pre-score adjudication rule.', False)],
        'partial': [
            ('U', 'Duke headquarters are absent. Its pipeline ownership and routing do not establish headquarters state. Complete reference gives North Carolina.', False),
            ('S', 'Affiliated Managers Group Massachusetts remains directly supplied.', False),
            ('U', 'Explicit not both is unambiguous. Retained AMG Massachusetts cannot establish Duke is elsewhere; the complete reference North Carolina confirms No.', False)]},
    528: {
        'complete': [
            ('S', 'Wavves second source sentence states formed2008.', False),
            ('S', 'Social Code introduction states formed1999.', False),
            ('S', '1999 precedes2008, supported by both visible sources.', False)],
        'partial': [
            ('W', 'Wavves formation year is absent from all supplied passages; generated2005 conflicts with omitted reference2008, so unsupported by current sources and incorrect relative to full reference.', False),
            ('S', 'Retained Social Code introduction gives1999.', False),
            ('U', 'No visible Wavves chronology supports the ordering, although full references1999 and2008 confirm Social Code first.', False)]},
}


def main():
    generated = [g for g in readl('data/generated.jsonl') if g['split'] == 'validation']
    inputs = {r['row_id']: r for r in readl('data/inputs.jsonl')}
    refs = {r['question_id']: r for r in readl('data/references.jsonl')}
    annotations = []
    outside = []
    kinds = {
        'S': ('asserted', 'supported', 'correct', 0),
        'U': ('asserted', 'unsupported', 'correct', 1),
        'W': ('asserted', 'unsupported', 'incorrect', 1),
        'A': ('abstained', 'unresolved', 'not_applicable', None),
        'X': ('asserted', 'unresolved', 'unresolved', None),
        'Y': ('asserted', 'unresolved', 'correct', None)}
    for g in generated:
        ref = refs[g['question_id']]
        decisions = REVIEWS[ref['source_dataset_row_index']][g['condition']]
        assert len(g['items']) == len(decisions) == 3
        visible = {p['title'] for p in inputs[g['row_id']]['passages']}
        for item, (decision, rationale, multi) in zip(g['items'], decisions):
            stance, relation, correctness, risk = kinds[decision]
            evidence = [{**e, 'visible_in_current_materials': e['title'] in visible}
                        for e in ref['items'][item['item_index']-1]['evidence']]
            annotations.append({
                'item_id': item['item_id'], 'row_id': g['row_id'], 'question_id': g['question_id'],
                'item_index': item['item_index'], 'text': item['text'],
                'stance': stance, 'evidence_relation': relation,
                'reference_correctness': correctness, 'risk': risk,
                'rationale': rationale, 'evidence_refs': evidence, 'multiple_facts': multi,
                'answer_specificity': 'lower_bound_only_not_exact_requested_count' if ref['source_dataset_row_index'] == 229 and g['condition'] == 'partial' and item['item_index'] == 1 else 'not_flagged',
                'annotator': 'root assistant source review, blind to detector and baseline scores',
                'human_gold': False})
        assert g['parser']['unparsed_preamble'] == '' and not g['parser']['extra_numbered_items']
        outside.append({'row_id': g['row_id'], 'reviewed_complete_response': True,
                        'outside_numbered_item_assertions': [], 'note': 'No preamble, extra numbered item or separate unassigned tail assertion.'})
    assert len(annotations) == 36
    (ROOT / 'data/annotations_validation.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in annotations), encoding='utf-8')
    (ROOT / 'data/annotations_validation_notes.json').write_text(json.dumps({'scope': 'six validation groups, all sources and actual answers reviewed before fitting', 'outside_item_review': outside, 'ambiguous_item': '5abbd3ac55429931dba1458b__complete__3', 'question_presupposition_ambiguity': ['5ac289ff5542996366519a02__partial__1', '5ac289ff5542996366519a02__partial__3'], 'initial_presupposition_item_labels': 'unsupported/correct/risk1; corrected to unresolved/correct/risknull after independent blind review, before any fitting or score inspection; original outputs and frozen inputs preserved'}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('36 validation labels serialized from explicit source-reviewed decisions; no scores read.')


if __name__ == '__main__':
    main()
