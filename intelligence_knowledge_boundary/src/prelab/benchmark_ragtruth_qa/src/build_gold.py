"""Map released development human spans to the frozen QA token layout.

No tokenizer, model, feature arrays, predictions, official test, or withheld
answers are read. Official spans and source/group partitions are not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PARTITIONS = ('fit', 'calibration')
EXPECTED = {'fit': 634, 'calibration': 159}
VERSION = 'ragtruth-qa-gold-v1'
K = 4


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def merge_intervals(intervals):
    result = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def character_spans(text, mask, left=0, right=None):
    """Runs of the actually selected original characters; never word expansion."""
    right = len(text) if right is None else right
    result = []
    at = left
    while at < right:
        if not mask[at]:
            at += 1
            continue
        end = at + 1
        while end < right and mask[end]:
            end += 1
        result.append({'start': at, 'end': end, 'text': text[at:end]})
        at = end
    return result


def map_characters(text, labels, offsets):
    lexical_chars = [char.isalnum() for char in text]
    risk_chars = [False] * len(text)
    for label in labels:
        left, right = label['start'], label['end']
        assert isinstance(left, int) and isinstance(right, int)
        assert 0 <= left <= right <= len(text), 'Invalid official span geometry; do not repair'
        assert text[left:right] == label['text'], 'Official text/offset mismatch; do not repair'
        for at in range(left, right):
            risk_chars[at] |= lexical_chars[at]
    lexical_prefix, risk_prefix = [0], [0]
    for lexical, risk in zip(lexical_chars, risk_chars):
        lexical_prefix.append(lexical_prefix[-1] + int(lexical))
        risk_prefix.append(risk_prefix[-1] + int(risk))
    lexical_mask = [int(lexical_prefix[b] > lexical_prefix[a]) for a, b in offsets]
    risk_mask = [int(risk_prefix[b] > risk_prefix[a]) for a, b in offsets]
    return lexical_chars, risk_chars, lexical_mask, risk_mask


def windows_for_count(count):
    assert count > 0
    return [(start, min(start + K, count)) for start in range(max(1, count - K + 1))]


def check_fixed_layout(row, plan):
    """Independent re-selection from the complete sequence offset table."""
    for field in ('response_id', 'source_id', 'group_id', 'partition', 'original_response'):
        assert plan[field] == row[field], field
    text = row['original_response']
    assert digest_text(text) == row['answer_sha256'] == plan['answer_sha256']
    assert digest_text(row['released_prompt']) == row['prompt_sha256'] == plan['prompt_sha256']
    original = plan['original']
    begin, end = original['answer_character_range']
    assert end - begin == len(text) and original['prefix_character_length'] == begin
    full_offsets = original['input_token_offsets']
    positions = [j for j, (a, b) in enumerate(full_offsets) if b > begin and a < end and b > a]
    assert positions == original['answer_token_positions'], 'Missing or changed answer BPE position'
    assert positions and positions[0] > 0
    raw = [[full_offsets[j][0] - begin, full_offsets[j][1] - begin] for j in positions]
    clipped = [[max(0, a), min(len(text), b)] for a, b in raw]
    assert raw == original['response_token_offsets_raw']
    assert clipped == original['response_token_offsets']
    assert [original['input_ids'][j] for j in positions] == original['answer_token_ids']
    assert all(0 <= a < b <= len(text) for a, b in clipped)
    covered = [False] * len(text)
    for a, b in clipped:
        for at in range(a, b):
            covered[at] = True
    assert all(covered[j] or char.isspace() for j, char in enumerate(text)), 'Lost non-whitespace answer character'
    return original, covered


def independent_label_check(text, labels, offsets, lexical_mask, risk_mask):
    """Brute-force character oracle, independent of prefix sums/union masks."""
    expected_lexical, expected_risk = [], []
    for left, right in offsets:
        expected_lexical.append(int(any(text[at].isalnum() for at in range(left, right))))
        expected_risk.append(int(any(text[at].isalnum() and any(
            label['start'] <= at < label['end'] for label in labels) for at in range(left, right))))
    assert expected_lexical == lexical_mask
    assert expected_risk == risk_mask
    return expected_lexical, expected_risk


def explicit_edge_selfchecks():
    checks = []
    # The first clipped token includes the answer's A, not the template space.
    _, _, lexical, risk = map_characters('A, 20!', [{'start': 3, 'end': 5, 'text': '20'}],
                                        [[0, 1], [1, 2], [2, 3], [3, 5], [5, 6]])
    assert lexical == [1, 0, 0, 1, 0] and risk == [0, 0, 0, 1, 0]
    assert windows_for_count(5) == [(0, 4), (1, 5)]
    assert [int(any(risk[a:b])) for a, b in windows_for_count(5)] == [1, 1]
    checks.append('4 raw BPE windows include punctuation/space; no trailing short window')
    _, _, lexical, risk = map_characters('!', [{'start': 0, 'end': 1, 'text': '!'}], [[0, 1]])
    assert windows_for_count(1) == [(0, 1)] and lexical == [0] and risk == [0]
    checks.append('Punctuation-only span leaves answer risk=1 but window excluded, no invented risk token')
    _, _, lexical, risk = map_characters('Z', [{'start': 0, 'end': 0, 'text': ''}], [[0, 1]])
    assert lexical == [1] and risk == [0]
    checks.append('Zero-length official span leaves answer risk=1 and eligible window label=0')
    _, _, lexical, risk = map_characters('汉', [{'start': 0, 'end': 1, 'text': '汉'}], [[0, 1], [0, 1]])
    assert lexical == [1, 1] and risk == [1, 1] and windows_for_count(2) == [(0, 2)]
    checks.append('Unicode isalnum and duplicate byte-fallback offsets preserve both raw BPE tokens')
    return {'passed': True, 'checks': checks, 'synthetic_cases_not_added_to_dataset': True}


def build(overwrite=False):
    protocol = ROOT / 'ANNOTATION_PROTOCOL.md'
    assert protocol.exists(), 'Write and fix the annotation protocol before exporting labels'
    prep_manifest_path = DATA / 'feature_preparation/manifest.json'
    plans_path = DATA / 'feature_preparation/plans.jsonl'
    prep = read_json(prep_manifest_path)
    dev_manifest_path = DATA / 'development_manifest.json'
    dev = read_json(dev_manifest_path)
    assert sha(plans_path) == prep['plans_jsonl_sha256']
    assert sha(dev_manifest_path) == prep['signature']['development_manifest_sha256']
    paths = {'annotation_protocol': protocol, 'build_gold_code': Path(__file__),
             'development_manifest': dev_manifest_path, 'feature_preparation_manifest': prep_manifest_path,
             'feature_preparation_plans': plans_path}
    for partition in PARTITIONS:
        path = DATA / (partition + '.jsonl')
        expected = dev['files_sha256'].get('data/' + path.name, dev['files_sha256'].get('data\\' + path.name))
        assert sha(path) == expected == prep['signature']['development_data_sha256'][partition]
        paths[partition + '_source'] = path
    signature = {'version': VERSION, 'files_sha256': {key: sha(path) for key, path in paths.items()},
                 'window_k': K, 'window_stride': 1, 'label_types_filtered': False,
                 'refusal_classifier_used': False, 'tokenizer_rerun': False}
    manifest_path = DATA / 'gold_manifest.json'
    if manifest_path.exists() and not overwrite:
        previous = read_json(manifest_path)
        assert previous['signature'] == signature, 'Gold signature changed; do not silently overwrite'
        assert previous['complete']
        for record in previous['outputs']:
            assert sha(ROOT / record['path']) == record['sha256']
        print(json.dumps({'cached_complete': True, 'counts': previous['counts']}, ensure_ascii=False, indent=2))
        return
    plans = {row['response_id']: row for row in read_jsonl(plans_path)}
    assert len(plans) == prep['records'] == sum(EXPECTED.values())
    edges = {'zero_length_spans': [], 'spans_without_isalnum': [], 'uncovered_span_nonwhitespace_characters': [],
             'uncovered_span_isalnum_characters': [], 'risk_answers_without_risk_tokens': [],
             'answers_without_eligible_windows': []}
    counts = {}; files = []; global_ids = set(); groups_by_partition = {}
    total_boundary = total_spans = total_tokens = 0
    file_line_counts = Counter()
    toy = explicit_edge_selfchecks()
    with ExitStack() as stack:
        handles = {}
        for partition in PARTITIONS:
            for stem in ('tokens', 'windows_k4', 'answers', 'windows_excluded'):
                path = DATA / f'{stem}_{partition}.jsonl'
                pending = path.with_suffix(path.suffix + '.pending')
                handles[(stem, partition)] = stack.enter_context(pending.open('w', encoding='utf-8', newline='\n'))
                files.append((path, pending))
        for partition in PARTITIONS:
            counter = Counter(); groups_by_partition[partition] = set()
            for row in read_jsonl(DATA / (partition + '.jsonl')):
                assert row['partition'] == partition and row['official_split'] == 'train' and row['quality'] == 'good'
                rid = row['response_id']; assert rid not in global_ids; global_ids.add(rid)
                groups_by_partition[partition].add(row['group_id'])
                original, covered = check_fixed_layout(row, plans[rid])
                text, labels = row['original_response'], row['labels']
                offsets = original['response_token_offsets']; raw = original['response_token_offsets_raw']
                n = len(offsets); assert n == len(original['answer_token_ids']) > 0
                lexical_chars, risk_chars, lexical, risk = map_characters(text, labels, offsets)
                oracle_lexical, oracle_risk = independent_label_check(text, labels, offsets, lexical, risk)
                ident = {key: row[key] for key in ('response_id', 'source_id', 'group_id', 'partition')}
                ident['answer_id'] = rid
                flags = []
                span_tokens = []
                for index, label in enumerate(labels):
                    a, b = label['start'], label['end']
                    ref = {**ident, 'span_index': index, 'original_start': a, 'original_end': b, 'original_text': label['text']}
                    if a == b:
                        edges['zero_length_spans'].append(ref); flags.append('zero_length_span')
                    if not any(lexical_chars[a:b]):
                        edges['spans_without_isalnum'].append(ref); flags.append('span_without_isalnum')
                    missing_nonspace = [j for j in range(a, b) if not text[j].isspace() and not covered[j]]
                    missing_lexical = [j for j in range(a, b) if text[j].isalnum() and not covered[j]]
                    if missing_nonspace:
                        edges['uncovered_span_nonwhitespace_characters'].append({**ref, 'character_indices': missing_nonspace})
                    if missing_lexical:
                        edges['uncovered_span_isalnum_characters'].append({**ref, 'character_indices': missing_lexical})
                    indices = [j for j, (left, right) in enumerate(offsets) if any(
                        text[at].isalnum() for at in range(max(left, a), min(right, b)))]
                    span_tokens.append({'span_index': index, 'original_start': a, 'original_end': b,
                                        'risk_token_indices': indices})
                if labels and not any(risk):
                    edges['risk_answers_without_risk_tokens'].append({**ident, 'original_labels': labels})
                    flags.append('answer_risk_without_risk_token')
                token_record = {**ident, 'original_response': text, 'answer_sha256': row['answer_sha256'],
                    'original_labels': labels, 'token_count': n, 'token_ids': original['answer_token_ids'],
                    'answer_token_positions': original['answer_token_positions'],
                    'response_token_offsets_raw': raw, 'response_token_offsets': offsets,
                    'lexical_mask': lexical, 'risk_mask': risk,
                    'risk_character_spans': character_spans(text, risk_chars),
                    'token_risk_character_spans': [character_spans(text, risk_chars, a, b) if risk[j] else []
                                                   for j, (a, b) in enumerate(offsets)],
                    'span_token_mapping': span_tokens, 'answer_risk': int(bool(labels)),
                    'first_answer_token_preserved': True, 'edge_case_flags': sorted(set(flags))}
                dump_line(handles[('tokens', partition)], token_record)
                file_line_counts[f'tokens_{partition}.jsonl'] += 1
                eligible = positive = excluded = 0
                for start, end in windows_for_count(n):
                    indices = list(range(start, end))
                    lexical_indices = [j for j in indices if lexical[j]]
                    risk_indices = [j for j in indices if risk[j]]
                    intervals = merge_intervals(offsets[start:end])
                    left, right = min(a for a, _ in intervals), max(b for _, b in intervals)
                    window_risk_spans = merge_intervals([(r['start'], r['end']) for j in indices
                                                        for r in token_record['token_risk_character_spans'][j]])
                    assert bool(lexical_indices) == any(oracle_lexical[start:end])
                    # Independent window oracle walks original characters, not predicted labels.
                    oracle_gold = int(any(text[at].isalnum() and any(l['start'] <= at < l['end'] for l in labels)
                                          for j in indices for at in range(*offsets[j])))
                    assert int(bool(risk_indices)) == oracle_gold
                    window = {**ident, 'window_id': f'{rid}__k4_{start:05d}', 'k': K, 'stride': 1,
                        'token_start': start, 'token_end': end, 'token_indices': indices,
                        'answer_token_positions': original['answer_token_positions'][start:end],
                        'token_ids': original['answer_token_ids'][start:end],
                        'character_intervals': intervals, 'char_start': left, 'char_end': right,
                        'bounding_text': text[left:right], 'lexical_token_indices': lexical_indices,
                        'risk_token_indices': risk_indices,
                        'risk_character_spans': [{'start': a, 'end': b, 'text': text[a:b]} for a, b in window_risk_spans],
                        'eligible': bool(lexical_indices), 'label': int(bool(risk_indices)) if lexical_indices else None}
                    if lexical_indices:
                        dump_line(handles[('windows_k4', partition)], window)
                        file_line_counts[f'windows_k4_{partition}.jsonl'] += 1
                        eligible += 1; positive += int(bool(risk_indices))
                    else:
                        window['exclusion_reason'] = 'no_lexical_token'
                        dump_line(handles[('windows_excluded', partition)], window)
                        file_line_counts[f'windows_excluded_{partition}.jsonl'] += 1
                        excluded += 1
                if not eligible:
                    edges['answers_without_eligible_windows'].append(ident); flags.append('no_eligible_window')
                answer = {**ident, 'original_response': text, 'answer_sha256': row['answer_sha256'],
                    'original_labels': labels, 'quality': 'good', 'eligible': True, 'label': int(bool(labels)),
                    'token_count': n, 'lexical_token_count': sum(lexical), 'risk_token_count': sum(risk),
                    'official_span_count': len(labels), 'candidate_window_count': len(windows_for_count(n)),
                    'eligible_window_count': eligible, 'positive_window_count': positive,
                    'excluded_window_count': excluded, 'edge_case_flags': sorted(set(flags))}
                dump_line(handles[('answers', partition)], answer)
                file_line_counts[f'answers_{partition}.jsonl'] += 1
                counter.update({'answers': 1, 'positive_answers': int(bool(labels)), 'negative_answers': int(not labels),
                                'raw_tokens': n, 'lexical_tokens': sum(lexical), 'risk_tokens': sum(risk),
                                'official_spans': len(labels), 'candidate_windows': eligible + excluded,
                                'eligible_windows': eligible, 'positive_windows': positive,
                                'negative_windows': eligible - positive, 'excluded_no_lexical_windows': excluded})
                total_boundary += int(raw[0][0] < 0); total_spans += len(labels); total_tokens += n
            assert counter['answers'] == EXPECTED[partition]
            counter['groups'] = len(groups_by_partition[partition]); counts[partition] = dict(counter)
    assert not groups_by_partition['fit'] & groups_by_partition['calibration']
    assert global_ids == set(plans) and total_tokens == prep['response_tokens']
    assert total_boundary == prep['boundary_crossing_records']
    assert not edges['uncovered_span_nonwhitespace_characters'] and not edges['uncovered_span_isalnum_characters']
    selfcheck = {'passed': True, 'answers_checked': len(global_ids), 'raw_tokens_checked': total_tokens,
        'original_spans_checked': total_spans, 'first_boundary_crossing_tokens_preserved': total_boundary,
        'full_input_offset_reselection_exact': True, 'full_original_span_nonwhitespace_coverage': True,
        'all_span_isalnum_characters_covered': True, 'independent_character_token_oracle_exact': True,
        'independent_character_window_oracle_exact': True, 'fit_calibration_group_disjoint': True,
        'no_tokenizer_or_feature_values_used': True, 'no_test_or_withheld_answers_or_labels_read': True,
        'explicit_edge_selfchecks': toy, 'counts': counts}
    edge_report = {'version': VERSION, 'counts': {key: len(value) for key, value in edges.items()},
                   'cases': edges, 'policy': 'Only report boundary cases; do not modify spans, answer labels or eligibility.'}
    for name, value in [('gold_selfcheck.json', selfcheck), ('gold_edge_cases.json', edge_report)]:
        path = DATA / name; pending = path.with_suffix(path.suffix + '.pending')
        dump_json(pending, value); files.append((path, pending))
    output_hashes = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(pending),
                      'bytes': pending.stat().st_size,
                      **({'rows': file_line_counts[path.name]} if path.suffix == '.jsonl' else {})}
                     for path, pending in files]
    for path, pending in files:
        pending.replace(path)
    manifest = {'complete': True, 'version': VERSION, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'signature': signature, 'signature_sha256': digest_text(json.dumps(signature, sort_keys=True, separators=(',', ':'))),
        'source_paths': {key: str(path.resolve()) for key, path in paths.items()},
        'counts': counts, 'total_answers': len(global_ids), 'total_raw_tokens': total_tokens,
        'total_official_spans': total_spans, 'edge_case_counts': edge_report['counts'],
        'official_test_opened': False, 'model_or_probe_state_written': False,
        'all_quality_good_answers_eligible': True,
        'difference_from_r16': 'No automatic refusal exclusion: unlabeled quality-good refusals retain eligible negative lexical windows. Cross-dataset F1 differences are not evidence of an algorithmic gain.',
        'outputs': output_hashes, 'selfcheck_passed': True}
    pending = manifest_path.with_suffix('.json.pending'); dump_json(pending, manifest); pending.replace(manifest_path)
    print(json.dumps({'complete': True, 'counts': counts, 'edge_case_counts': edge_report['counts'],
                      'gold_manifest_sha256': sha(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Explicitly rebuild these new gold outputs after a reviewed implementation fix')
    build(parser.parse_args().overwrite)
