"""Read only all3680 human fit answers; inventory two nonexclusive risk types."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
DATA = QA/'fit_expansion/data'
KINDS = ('Evident Baseless Info', 'Subtle Baseless Info', 'Evident Conflict', 'Subtle Conflict')
FAMILY = {KINDS[0]: 1, KINDS[1]: 1, KINDS[2]: 2, KINDS[3]: 2}
NAMES = ('clean', 'baseless_only', 'conflict_only', 'both')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def save(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def savel(p, rows):
    with p.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')


def bindings():
    names = [Path(__file__), DATA/'fit.jsonl', DATA/'tokens_fit.jsonl', DATA/'answers_fit.jsonl',
        QA/'fit_expansion/type_balanced_v1/REPORT.md', QA/'fit_expansion/type_balanced_v1/protocol.json',
        QA/'results/semantic_multitask_v1/REPORT.md', QA/'results/semantic_multitask_v1/protocol.json']
    return {str(p.resolve()): sha(p) for p in names}


def prepare():
    protocol = {'version': 'fit3680-human-type-partition-inventory-v1',
        'input_scope': 'Only fit_expansion/data/{fit,tokens_fit,answers_fit}.jsonl;3680fit. No calibration/test raw data, model inference or fitting.',
        'type_map': FAMILY,
        'token_mapping': 'Original clipped response BPE offsets. A type marks a token iff token/span intersection contains an actual isalnum character. Independently rebuild all spans and require exact existing span_token_mapping.',
        'partitions': 'Lexical tokens only: neither(clean), baseless only, conflict only, both. Nonlexical raw tokens separate, never treated as train negatives.',
        'union': 'OR of two type masks must exactly equal original binary risk_mask on allraw tokens; all four original types and labels unchanged.',
        'answer_groups': 'Co-occurrence at answer/source/group level differs from overlap on a single token; report separately. Source/group IDs are metadata, not proposed features.',
        'old_results': 'Read only completed aggregate reports for type-balanced binary LR and original634fit auxiliary-type TCN; no re-evaluation or new calibration.',
        'output': 'Counts/provenance only, not exported new training labels. Any proposed head is an untrained candidate.',
        'trained': False, 'GPU_used': False, 'calibration_or_test_rows_read': False}
    assert not (OUT/'design_freeze.json').exists()
    save(OUT/'protocol.json', protocol)
    save(OUT/'design_freeze.json', {'source_sha256': bindings(), 'protocol_sha256': sha(OUT/'protocol.json')})
    print('FIT_TYPE_INVENTORY_FROZEN', flush=True)


def run():
    frozen = read(OUT/'design_freeze.json')
    assert frozen['source_sha256'] == bindings() and frozen['protocol_sha256'] == sha(OUT/'protocol.json')
    assert not (OUT/'complete.json').exists()
    answers = list(lines(DATA/'fit.jsonl'))
    answer_meta = {a['response_id']: a for a in lines(DATA/'answers_fit.jsonl')}
    assert len(answers) == len(answer_meta) == 3680
    ids = [a['response_id'] for a in answers]; assert len(set(ids)) == 3680
    tokens = list(lines(DATA/'tokens_fit.jsonl'))
    assert [t['response_id'] for t in tokens] == ids
    allcounts = Counter(); type_tokens = Counter(); span_counts = Counter(); answer_presence = Counter()
    by_group, by_source, by_generator = defaultdict(list), defaultdict(list), defaultdict(list)
    per_answer = []; no_lexical_span = []; mixed_tokens = []
    for answer, tok in zip(answers, tokens):
        rid = answer['response_id']; am = answer_meta[rid]; text = answer['original_response']
        assert answer['partition'] == tok['partition'] == am['partition'] == 'fit'
        assert answer['official_split'] == 'train' and answer['quality'] == 'good' and am['eligible']
        for key in ('response_id', 'source_id', 'group_id', 'answer_sha256'):
            assert answer[key] == tok[key] == am[key]
        assert text == tok['original_response'] == am['original_response']
        assert answer['labels'] == tok['original_labels'] == am['original_labels']
        assert hashlib.sha256(text.encode()).hexdigest() == answer['answer_sha256']
        offsets = np.asarray(tok['response_token_offsets'], int)
        n = tok['token_count']; assert offsets.shape == (n, 2)
        alnum = np.asarray([c.isalnum() for c in text], np.int64)
        prefix = np.concatenate(([0], alnum.cumsum()))
        assert (offsets[:,0] >= 0).all() and (offsets[:,1] <= len(text)).all()
        lex = prefix[offsets[:,1]] > prefix[offsets[:,0]]
        assert np.array_equal(lex, np.asarray(tok['lexical_mask'], bool))
        y4 = np.zeros((n, 4), bool)
        assert len(tok['span_token_mapping']) == len(answer['labels'])
        for j, span in enumerate(answer['labels']):
            kind = span['label_type']; k = KINDS.index(kind)
            a, b = span['start'], span['end']
            assert 0 <= a < b <= len(text) and text[a:b] == span['text']
            lo = np.maximum(offsets[:,0], a); hi = np.minimum(offsets[:,1], b)
            nonempty = lo < hi; hit = np.zeros(n, bool)
            hit[nonempty] = prefix[hi[nonempty]] > prefix[lo[nonempty]]
            assert np.flatnonzero(hit).tolist() == tok['span_token_mapping'][j]['risk_token_indices']
            y4[:,k] |= hit; span_counts[kind] += 1
            if not hit.any(): no_lexical_span.append({'response_id': rid, 'span_index': j})
        y1, y2 = y4[:,:2].any(1), y4[:,2:].any(1)
        bit = y1.astype(np.int8) + 2*y2.astype(np.int8)
        assert np.array_equal(bit > 0, np.asarray(tok['risk_mask'], bool))
        assert not y4[~lex].any()
        assert int((bit > 0).sum()) == am['risk_token_count']
        assert int((bit > 0).any()) == am['label'] == tok['answer_risk']
        counts = np.bincount(bit[lex], minlength=4)
        presence = int(y1.any()) + 2*int(y2.any())
        type_tokens.update({kind: int(y4[:,k].sum()) for k, kind in enumerate(KINDS)})
        allcounts.update({'raw_tokens': n, 'lexical_tokens': int(lex.sum()), 'nonlexical_tokens': int((~lex).sum()),
            'risk_tokens': int((bit > 0).sum()), **{name: int(counts[i]) for i, name in enumerate(NAMES)}})
        answer_presence[NAMES[presence]] += 1
        row = {'response_id': rid, 'source_id': answer['source_id'], 'group_id': answer['group_id'],
            'generator_metadata': answer['model'], 'raw_tokens': n, 'lexical_tokens': int(lex.sum()),
            'token_partition': {name: int(counts[i]) for i, name in enumerate(NAMES)},
            'answer_type_presence': NAMES[presence], 'original_span_count': len(answer['labels']),
            'four_type_tokens': {kind: int(y4[:,k].sum()) for k,kind in enumerate(KINDS)}}
        per_answer.append(row)
        by_group[answer['group_id']].append(row); by_source[answer['source_id']].append(row)
        by_generator[answer['model']].append(row)
        if counts[3]: mixed_tokens.append({'response_id': rid, 'token_indices': np.flatnonzero(bit == 3).tolist()})
    def summarize(grouped, identity):
        result = []
        for ident, rr in sorted(grouped.items()):
            count = {name: sum(row['token_partition'][name] for row in rr) for name in NAMES}
            presence = int(count['baseless_only'] + count['both'] > 0) + 2*int(count['conflict_only'] + count['both'] > 0)
            result.append({identity: ident, 'answers': len(rr), 'source_ids': sorted({z['source_id'] for z in rr}),
                'group_ids': sorted({z['group_id'] for z in rr}), 'token_partition': count,
                'type_presence': NAMES[presence], 'answer_type_presence': dict(Counter(z['answer_type_presence'] for z in rr))})
        return result
    group_rows = summarize(by_group, 'group_id'); source_rows = summarize(by_source, 'source_id')
    generator_rows = summarize(by_generator, 'generator_metadata')
    assert len(by_group) == len(by_source) == 615
    assert sum(allcounts[k] for k in NAMES) == allcounts['lexical_tokens']
    assert allcounts['baseless_only'] + allcounts['conflict_only'] + allcounts['both'] == allcounts['risk_tokens']
    result = {'status': 'passed', 'answers': 3680, 'sources': len(by_source), 'groups': len(by_group),
        'token_counts': dict(allcounts), 'four_type_token_counts': dict(type_tokens),
        'original_span_counts': dict(span_counts), 'answer_type_presence': dict(answer_presence),
        'group_type_presence': dict(Counter(x['type_presence'] for x in group_rows)),
        'source_type_presence': dict(Counter(x['type_presence'] for x in source_rows)),
        'same_BPE_token_family_overlap_answers': len(mixed_tokens), 'same_BPE_token_family_overlaps': mixed_tokens,
        'spans_without_lexical_token': no_lexical_span,
        'OR_exact_all_tokens': True, 'original_span_mappings_exact': True,
        'answer_binary_exact': True, 'original_source_group_identity_exact': True,
        'nonlexical_not_training_negative': True,
        'calibration_or_test_rows_read': False, 'trained': False, 'GPU_used': False}
    save(OUT/'summary.json', result)
    savel(OUT/'per_answer_counts.jsonl', per_answer)
    savel(OUT/'per_group_counts.jsonl', group_rows)
    savel(OUT/'per_source_counts.jsonl', source_rows)
    save(OUT/'generator_metadata_counts.json', generator_rows)
    assert frozen['source_sha256'] == bindings()
    names = ['summary.json','per_answer_counts.jsonl','per_group_counts.jsonl','per_source_counts.jsonl','generator_metadata_counts.json']
    save(OUT/'complete.json', {'status': 'complete_readonly_inventory',
        'files_sha256': {n: sha(OUT/n) for n in names}, 'trained': False, 'GPU_used': False,
        'calibration_or_test_rows_read': False})
    print(json.dumps({k: result[k] for k in ('status','answers','sources','groups','token_counts','answer_type_presence','group_type_presence')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare','run'))
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
