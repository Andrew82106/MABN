"""Independent delivery checks for the Round6 pilot; no GPU or model judgements."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def readl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def data_checks():
    rows = readl(ROOT / 'data/inputs.jsonl')
    refs = readl(ROOT / 'data/references.jsonl')
    config = json.loads((ROOT / 'design_v1.json').read_text(encoding='utf-8'))
    template = (ROOT / config['generation_prompt']['user_template_file']).read_text(encoding='utf-8')
    byq = defaultdict(list)
    used_sources = defaultdict(set)
    used_subjects = defaultdict(set)
    checks = {}
    for row in rows:
        byq[row['question_id']].append(row)
        for passage in row['passages']:
            used_sources[passage['title'].strip().casefold()].add(row['question_id'])
        for subject in row['subjects']:
            used_subjects[subject.strip().casefold()].add(row['question_id'])
    checks['60_unique_input_rows'] = len(rows) == len({r['row_id'] for r in rows}) == 60
    checks['30_unique_reference_questions'] = len(refs) == len({r['question_id'] for r in refs}) == len(byq) == 30
    checks['references_cover_questions'] = {r['question_id'] for r in refs} == set(byq)
    checks['pairs_complete_partial'] = all(len(pair) == 2 and {r['condition'] for r in pair} == {'complete', 'partial'} for pair in byq.values())
    checks['pair_split_same'] = all(len({r['split'] for r in pair}) == 1 for pair in byq.values())
    split_counts = Counter(pair[0]['split'] for pair in byq.values())
    checks['question_split_18_6_6'] = split_counts == {'train': 18, 'validation': 6, 'test': 6}
    checks['four_passages_three_questions'] = all(len(r['passages']) == 4 and len(r['questions']) == 3 and len(r['subjects']) == 2 for r in rows)
    checks['same_task_and_subjects_in_pair'] = all(pair[0]['questions'] == pair[1]['questions'] and pair[0]['subjects'] == pair[1]['subjects'] for pair in byq.values())
    checks['neutral_system'] = all(r['system'] == config['generation_prompt']['system'] == 'You are a helpful assistant.' for r in rows)
    cues = ['insufficient', 'abstain', 'cannot be determined', 'unverified', 'do not', 'only information', 'not enough']
    checks['neutral_template'] = not any(c in template.lower() for c in cues)
    # Rendering is checked separately against the builder's declared passage format.
    checks['task_instruction_present'] = all(template.splitlines()[0] in r['prompt'] for r in rows)
    checks['all_passages_in_prompt'] = all(all(p['text'] in r['prompt'] for p in r['passages']) for r in rows)
    checks['all_questions_in_prompt'] = all(all(q in r['prompt'] for q in r['questions']) for r in rows)
    checks['prompt_exactly_neutral_template'] = all(r['prompt'] == template.strip().format(
        questions='\n'.join(f'{i+1}. {q}' for i, q in enumerate(r['questions'])),
        search_results='\n\n'.join(f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(r['passages']))) for r in rows)
    dev = readl(ROOT / 'data/dev_inputs.jsonl')
    byrow = {r['row_id']: r for r in rows}
    checks['development_pair_unchanged_train_only'] = all(r == byrow.get(r['row_id']) and r['split'] == 'train' for r in dev)
    checks['no_shared_displayed_sources_between_questions'] = all(len(qids) == 1 for qids in used_sources.values())
    checks['no_shared_subjects_between_questions'] = all(len(qids) == 1 for qids in used_subjects.values())
    changes = []
    for qid, pair in byq.items():
        c = next(r for r in pair if r['condition'] == 'complete')
        p = next(r for r in pair if r['condition'] == 'partial')
        indexes = [i for i in range(4) if c['passages'][i] != p['passages'][i]]
        changes.append({'question_id': qid, 'changed_positions': indexes})
    checks['one_passage_replaced_same_position'] = all(len(x['changed_positions']) == 1 for x in changes)
    preview = json.loads((ROOT / 'planning/hotpotqa_preview.json').read_text(encoding='utf-8'))
    preview_ids = {r['id'] for r in preview['rows']}
    preview_sources = {t for r in preview['rows'] for t in r['context']['title']}
    test = [r for r in rows if r['split'] == 'test']
    checks['test_excludes_preview_ids'] = not {r['question_id'] for r in test} & preview_ids
    checks['test_excludes_preview_sources'] = not {p['title'] for r in test for p in r['passages']} & preview_sources
    checks['model_files_available'] = (ROOT.parents[1] / config['model_path']).is_dir()
    return {'checks': checks, 'split_counts': dict(split_counts), 'changes': changes,
            'shared_sources': {k: sorted(v) for k, v in used_sources.items() if len(v) > 1},
            'input_sha256': sha(ROOT / 'data/inputs.jsonl'), 'reference_sha256': sha(ROOT / 'data/references.jsonl')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['data', 'final'], default='data')
    args = p.parse_args()
    result = data_checks()
    if args.stage == 'final':
        checks = result['checks']
        generated = readl(ROOT / 'data/generated.jsonl')
        annotations = readl(ROOT / 'data/annotations.jsonl')
        predictions = readl(ROOT / 'results/predictions.jsonl')
        metrics = json.loads((ROOT / 'results/metrics.json').read_text(encoding='utf-8'))
        freeze = json.loads((ROOT / 'results/freeze.json').read_text(encoding='utf-8'))
        checks['all_60_original_generations_retained'] = len(generated) == 60
        itemids = {i['item_id'] for r in generated for i in r['items']}
        checks['all_item_annotations_present_once'] = len(annotations) == len(itemids) == len({a['item_id'] for a in annotations}) and {a['item_id'] for a in annotations} == itemids
        checks['test_exactly_frozen_items'] = sorted(p['item_id'] for p in predictions) == freeze['test_item_ids']
        checks['test_contains_six_question_groups'] = len({p['question_id'] for p in predictions}) == 6
        checks['test_only_after_freeze'] = freeze['utc'] < metrics['utc']
        checks['frozen_files_unchanged'] = all(sha(ROOT / path) == value for path, value in freeze['input_sha256'].items())
        checks['generation_manifest_hash_valid'] = sha(ROOT / 'data/generated.jsonl') == json.loads((ROOT / 'data/generation_manifest.json').read_text(encoding='utf-8'))['generated_file_sha256']
        for filename in ['data/generation_manifest.json', 'data/features/manifest.json', 'data/baseline_manifest.json']:
            checks[filename + '_complete'] = json.loads((ROOT / filename).read_text(encoding='utf-8'))['complete']
        independent = {}
        for method, summary in metrics['methods'].items():
            cohort = [p for p in predictions if p['annotation']['stance'] == 'asserted' and p['annotation']['risk'] in [0, 1]]
            counts = Counter()
            for p in cohort:
                y = p['annotation']['risk']
                alert = p['methods'][method]['prediction'] == 1
                counts['tp' if y and alert else 'fn' if y else 'fp' if alert else 'tn'] += 1
            tp, fp, fn = [counts[k] for k in ['tp', 'fp', 'fn']]
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
            m = summary['end_to_end']
            checks[method + '_confusion_independently_matches'] = all(counts[k] == m[k] for k in ['tp', 'fp', 'fn', 'tn'])
            checks[method + '_f1_independently_matches'] = (f1 is None and m['f1'] is None) or (f1 is not None and abs(f1 - m['f1']) < 1e-12)
            independent[method] = {**dict(counts), 'f1': f1}
        result['independent_test_counts'] = independent
    result.update({'stage': args.stage, 'utc': datetime.now(timezone.utc).isoformat(),
                   'scope': 'Independent structural checks. Semantic evidence/labels require recorded source review.',
                   'passed': all(result['checks'].values()), 'python': sys.executable})
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / ('data_audit.json' if args.stage == 'data' else 'final_audit.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
