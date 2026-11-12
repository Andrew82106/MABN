"""Text-only source citation alignment, original793 answers and unchanged windows."""
from pathlib import Path
from collections import Counter
import argparse
import re
import time
import numpy as np
import run_development as q
import run_claim_pooling as pooling

OUT = q.ROOT / 'results/citation_alignment_v1'
EXPR = r'\d+(?:\s*(?:[-–—/,;]|\band\b|\bto\b)\s*(?:and\s+)?\d+)*'
NAMED = re.compile(r'\bpassages?\s+(?:no\.?\s*)?(' + EXPR + r')', re.I)
BRACKET = re.compile(r'\[\s*(' + EXPR + r')\s*\]', re.I)
SOURCE = re.compile(r'^\s*passage\s+(\d+)\s*:', re.I | re.M)
WORD = re.compile(r'[^\W_]+', re.UNICODE)
NAMES = ['any_source_coverage_mean', 'cited_source_coverage_mean', 'coverage_gap_mean',
    'invalid_reference_ratio_mean', 'explicit_reference_claim_fraction', 'citation_alnum_overlap',
    'coverage_gap_x_overlap', 'invalid_ratio_x_overlap']


def protocol():
    return {'version': 'native-qa-citation-alignment-v1', 'answers': 793, 'fit': 634, 'calibration': 159,
        'windows': 210364, 'columns': NAMES,
        'citation_parser': 'Answer text only. Case-insensitive passage(s) N / optional No. and numeric [N]; support comma,slash,semicolon,and lists plus hyphen/en/em-dash/to ranges. IDs1..99, ascending ranges expanded, unique IDs per claim. Four-digit years/descending/too-large expressions are unknown, never automatically labeled risk. Plain dates/numbers without citation syntax ignored.',
        'source_parser': 'Only line-start passage N: headers in unmodified retrieved_passages; require exactly original identifiers1,2,3, each once. Keep source numbering; remove header text for lexical comparison.',
        'claim_geometry': 'Reuse exact existing automatic claims in semantic_baseline/cuda_variant/plans.jsonl; each original lexical token assigned by maximum alphanumeric-character overlap, ties earlier claim, same run_claim_pooling.geometry. No gold-driven splitting or scope propagation between claims.',
        'words': 'Unicode alphanumeric word sets, lowercase, no stemming or stopword filtering. Remove recognized citation text from claim before words. Coverage(claim,source)=intersection word count/claim word count, empty claim coverage0.',
        'per_claim': 'a=max all three source coverages; c=max coverage among cited valid sources, else0; has=any recognized citation; gap=(a-c) if has else0; invalid=number unique cited IDs absent from source IDs/number unique cited IDs, else0. All parsed IDs enter invalid denominator. These are features, not labels.',
        'per_window': 'Mean a,c,gap,invalid,has over original lexical tokens assigned to claims. Citation overlap=unique alphanumeric characters inside recognized citation text intersect original window raw-token character intervals / unique alphanumeric characters in window. Last two features are mean gap or invalid times overlap. Nonlexical tokens never enter means, but original4rawBPE window positions remain unchanged.',
        'no_citations': 'Preserve all windows; a still records general lexical coverage, all seven citation-related features0. Unknown syntax exported as parser status, no forced label.',
        'bounds_and_labels': 'Exactly q.metadata answer/window order, original risk labels untouched; no official test. No models, probabilities, feature-fitting or calibration thresholds used.',
        'limits': 'Word overlap is not entailment; high overlap can accompany contradiction. Numeric brackets can be citations or other notation. Explicit-reference cues can concern a source mention rather than an assertion; this is not an automatic citation-error label.'}


def parse_citations(text):
    """No source identity, question or gold is accepted by this parser."""
    found, unknown = [], []
    for pattern, kind in ((NAMED, 'named'), (BRACKET, 'bracket')):
        for match in pattern.finditer(text):
            expr = match.group(1)
            nums = list(map(int, re.findall(r'\d+', expr)))
            if not nums or any(n < 1 or n > 99 for n in nums):
                unknown.append({'start': match.start(), 'end': match.end(), 'text': match.group(), 'reason': 'unsupported_numeric_identifier'})
                continue
            ids = set(nums); valid = True
            for a, b in re.findall(r'(\d+)\s*(?:[-–—]|\bto\b)\s*(\d+)', expr, re.I):
                a, b = int(a), int(b)
                if a > b:
                    valid = False; break
                ids.update(range(a, b + 1))
            if not valid:
                unknown.append({'start': match.start(), 'end': match.end(), 'text': match.group(), 'reason': 'descending_range'})
                continue
            found.append({'start': match.start(), 'end': match.end(), 'text': match.group(), 'ids': sorted(ids), 'kind': kind})
    # Preserve likely numeric/word-number citation forms the fixed grammar cannot interpret.
    possible = re.compile(r'\bpassages?\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\b|\[[^\]\n]*\d[^\]\n]*\]', re.I)
    occupied = found + unknown
    for m in possible.finditer(text):
        if not any(m.start() < r['end'] and r['start'] < m.end() for r in occupied):
            unknown.append({'start': m.start(), 'end': m.end(), 'text': m.group(), 'reason': 'unrecognized_citation_like_syntax'})
    found.sort(key=lambda r: (r['start'], r['end']))
    return {'references': found, 'unknown': unknown,
        'status': 'recognized_and_unknown' if found and unknown else 'recognized' if found else 'unknown' if unknown else 'none'}


def split_sources(text):
    matches = list(SOURCE.finditer(text))
    assert [int(m.group(1)) for m in matches] == [1, 2, 3], 'Expected exact passage1/2/3 headers'
    return {int(m.group(1)): text[m.end(): matches[i+1].start() if i+1 < len(matches) else len(text)] for i, m in enumerate(matches)}


def word_set(text):
    return set(WORD.findall(text.lower()))


def claim_features(text, parsed, source_words):
    keep = list(text)
    for ref in parsed['references']:
        keep[ref['start']:ref['end']] = ' ' * (ref['end'] - ref['start'])
    words = word_set(''.join(keep))
    cover = {i: len(words & value) / len(words) if words else 0. for i, value in source_words.items()}
    ids = {n for ref in parsed['references'] for n in ref['ids']}
    any_cover = max(cover.values(), default=0.)
    cited = max((cover[i] for i in ids if i in cover), default=0.)
    invalid = sum(i not in cover for i in ids) / len(ids) if ids else 0.
    return np.asarray([any_cover, cited, any_cover - cited if ids else 0., invalid, float(bool(ids))], np.float64)


def token_claim_assignment(text, offsets, lexical, claims):
    result = np.full(len(offsets), -1, np.int64)
    for i, (a, b) in enumerate(offsets):
        chars = [j for j in range(a, b) if text[j].isalnum()]
        assert bool(chars) == bool(lexical[i])
        if not chars: continue
        overlaps = [sum(c['start'] <= j < c['end'] for j in chars) for c in claims]
        chosen = max(range(len(claims)), key=lambda j: (overlaps[j], -j))
        assert overlaps[chosen] and all(any(c['start'] <= j < c['end'] for c in claims) for j in chars)
        result[i] = chosen
    return result


def citation_overlap(text, offsets, indices, refs):
    chars = {j for i in indices for j in range(*offsets[i]) if text[j].isalnum()}
    marked = {j for r in refs for j in range(r['start'], r['end']) if text[j].isalnum()}
    return len(chars & marked) / len(chars) if chars else 0.


def self_test():
    assert parse_citations('See passages 2-6.')['references'][0]['ids'] == [2, 3, 4, 5, 6]
    for example in ('passages 1/2/3', 'passages 1, 2, and 3', '[1,2,3]', 'passages 1 to 3'):
        assert parse_citations(example)['references'][0]['ids'] == [1, 2, 3]
    assert parse_citations('The date is 2024-02-06, and 15 people attended.')['status'] == 'none'
    assert parse_citations('[2024]')['status'] == 'unknown'
    assert parse_citations('passage one')['status'] == 'unknown'
    assert parse_citations('passages 6-2')['status'] == 'unknown'
    sources = {1: word_set('red fox'), 2: word_set('blue bird'), 3: word_set('green tree')}
    text = 'red fox [2]'; p = parse_citations(text); f = claim_features(text, p, sources)
    assert np.array_equal(f, [1, 0, 1, 0, 1])
    assert claim_features('red fox passages 2-6', parse_citations('red fox passages 2-6'), sources)[3] == .6
    assert np.array_equal(claim_features('red fox', parse_citations('red fox'), sources), [1, 0, 0, 0, 0])
    assert np.array_equal(claim_features('[1]', parse_citations('[1]'), sources), [0, 0, 0, 0, 1])
    text = 'x [2].'; offsets = [(0, 1), (1, 3), (3, 4), (3, 4), (4, 6)]
    refs = parse_citations(text)['references']
    assert citation_overlap(text, offsets, [0, 1, 2, 3], refs) == .5
    assert citation_overlap(text, offsets, [1, 2, 3, 4], refs) == 1.
    assert citation_overlap(text, offsets, [1, 4], refs) == 0.
    return {'passed': True, 'range_list_none_dates_short_punctuation_duplicate_offsets_checked': True,
        'parser_accepts_answer_text_only': True, 'model_loaded': False, 'GPU_used': False}


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists()
    q.save(OUT / 'CPU_SELFCHECK.json', self_test())
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'design_freeze.json', {'source_sha256': q.sha(Path(__file__)), 'plans_sha256': q.sha(pooling.PLANS),
        'protocol_sha256': q.sha(OUT / 'protocol.json'), 'no_training': True})
    print('CITATION_ALIGNMENT_DESIGN_FROZEN', flush=True)


def build():
    frozen = q.read(OUT / 'design_freeze.json')
    assert q.read(OUT / 'protocol.json') == protocol() and frozen['source_sha256'] == q.sha(Path(__file__))
    assert frozen['plans_sha256'] == q.sha(pooling.PLANS)
    assert not (OUT / 'started.json').exists()
    q.save(OUT / 'started.json', {'time': time.time()})
    meta = q.metadata(); plans = {r['response_id']: r for r in q.lines(pooling.PLANS)}
    text_rows = {r['response_id']: r for part in q.PARTITIONS for r in q.lines(q.DATA / (part + '.jsonl'))}
    features = []; claim_records = []; parsed_stats = Counter(); by_answer = {}; all_assignments = []; global_claim = 0
    start = time.perf_counter()
    for answer in meta['answers']:
        rid = answer['response_id']; row = text_rows[rid]; text = row['original_response']; token = meta['by_response'][rid]['tokens']
        assert text == answer['original_response']
        source_words = {i: word_set(s) for i, s in split_sources(row['retrieved_passages']).items()}
        claims = plans[rid]['claims']; vals = []; refs = []
        for j, claim in enumerate(claims):
            claim_text = text[claim['start']:claim['end']]
            assert claim_text == claim['text']
            parsed = parse_citations(claim_text); parsed_stats[parsed['status']] += 1
            value = claim_features(claim_text, parsed, source_words); vals.append(value)
            refs += [{**r, 'start': claim['start'] + r['start'], 'end': claim['start'] + r['end']} for r in parsed['references']]
            claim_records.append({'response_id': rid, 'partition': answer['partition'], 'claim_index': j,
                'start': claim['start'], 'end': claim['end'], 'text': claim_text, 'parser': parsed, 'features': value.tolist()})
        groups = token_claim_assignment(text, token['response_token_offsets'], token['lexical_mask'], claims)
        all_assignments.extend(np.where(groups >= 0, groups + global_claim, -1).tolist()); global_claim += len(claims)
        by_answer[rid] = {'claim_values': np.asarray(vals), 'groups': groups, 'references': refs}
    # Exact re-use check against the existing automatic claim geometry; diagnostics are not used.
    old_groups, _, old_count, _ = pooling.geometry(meta)
    assert np.array_equal(np.asarray(all_assignments), old_groups) and old_count == global_claim
    for w in meta['windows']:
        rid = w['response_id']; token = meta['by_response'][rid]['tokens']; info = by_answer[rid]
        ix = [i for i in w['token_indices'] if token['lexical_mask'][i]]; assert ix
        value = info['claim_values'][info['groups'][ix]].mean(axis=0)
        overlap = citation_overlap(text_rows[rid]['original_response'], token['response_token_offsets'], w['token_indices'], info['references'])
        features.append([*value, overlap, value[2] * overlap, value[3] * overlap])
    x = np.asarray(features, np.float32)
    assert x.shape == (210364, 8) and np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
    window_ids = np.asarray([w['window_id'] for w in meta['windows']])
    answer_ids = np.asarray([w['response_id'] for w in meta['windows']])
    np.savez_compressed(OUT / 'features.npz', features=x, window_ids=window_ids, response_ids=answer_ids)
    np.save(OUT / 'window_features.npy', x)
    q.save(OUT / 'feature_names.json', NAMES)
    with (OUT / 'claims.jsonl').open('w', encoding='utf-8') as stream:
        import json
        for r in claim_records: stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    q.save(OUT / 'geometry.json', {'answers': len(meta['answers']), 'windows': len(x), 'bounds': meta['bounds'],
        'columns': NAMES, 'automatic_claims': global_claim, 'claim_assignment_equal_existing': True,
        'window_order_sha256': q.digest(window_ids.tolist()), 'parser_status_counts': dict(parsed_stats),
        'claim_with_reference_count': sum(r['features'][4] > 0 for r in claim_records),
        'claims_with_invalid_reference': sum(r['features'][3] > 0 for r in claim_records),
        'windows_with_citation_overlap': int((x[:, 5] > 0).sum()),
        'unknown_examples': [r for r in claim_records if r['parser']['unknown']][:20],
        'risk_labels_unchanged': True, 'no_risk_labels_in_parsing_or_lexical_features': True})
    source_paths = [q.DATA / (p + '.jsonl') for p in q.PARTITIONS] + [pooling.PLANS, q.DATA / 'gold_manifest.json']
    q.save(OUT / 'complete.json', {'status': 'features_only_not_fitted', 'seconds': time.perf_counter() - start,
        'rows': len(x), 'columns': len(NAMES), 'window_order_sha256': q.digest(window_ids.tolist()),
        'files_sha256': {n: q.sha(OUT / n) for n in ('features.npz', 'window_features.npy', 'feature_names.json', 'claims.jsonl', 'geometry.json', 'protocol.json', 'design_freeze.json')},
        'source_sha256': {str(p.resolve()): q.sha(p) for p in source_paths}, 'official_test_opened': False,
        'GPU_used': False, 'model_loaded': False, 'trained': False})
    q.save(OUT / 'preparation_complete.json', q.read(OUT / 'complete.json'))
    print('CITATION_ALIGNMENT_COMPLETE', x.shape, dict(parsed_stats), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('initialize', 'build'))
    args = parser.parse_args()
    {'initialize': initialize, 'build': build}[args.action]()
