"""Read-only independent integrity and material-witness audit of FAVA staging."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import re
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    started = time.perf_counter()
    manifest = read(HERE / 'manifest.json')
    completed = read(HERE / 'complete.json')
    assert completed['manifest_sha256'] == sha(HERE / 'manifest.json')
    for name, expected in manifest['artifacts_sha256'].items():
        assert sha(HERE / name) == expected, name
    for path, expected in manifest['sources_sha256'].items():
        assert sha(path) == expected, path
    parser = module('audit_original_fava_parser', ROOT / 'additional_data/fava_training/inspect_training.py')
    rules = module('audit_existing_material_rules', ROOT / 'src/prepare_auxiliary_human.py')
    raw = read(ROOT / 'additional_data/fava_training/training.json')
    census = {row['raw_index']: row for row in rows(ROOT / 'additional_data/fava_training_full_alignment_v1/rows.jsonl')
              if row['exact_character_alignment'] and row['alignment_class'] == 'factual_only'}
    fixed = {row['raw_index']: row for row in rows(HERE / 'fixed_candidate_index.jsonl')}
    retained = {row['raw_index']: row for row in rows(HERE / 'candidate_fit.jsonl')}
    quarantine = {row['raw_index']: row for row in rows(HERE / 'quarantined_candidate_index.jsonl')}
    duplicates = rows(HERE / 'duplicate_index.jsonl')
    assert len(census) == len(fixed) == 7483 and set(census) == set(fixed)
    assert len(retained) == 7481 and len(quarantine) == 2 and not duplicates
    assert set(retained).isdisjoint(quarantine) and set(retained) | set(quarantine) == set(fixed)
    references = {}
    reference_members = defaultdict(set)
    reference_ordinals = defaultdict(list)
    span_types = Counter()
    for index, checked in census.items():
        original = raw[index]
        refs, answer = parser.prompt_parts(original['prompt'])
        assert len(refs) == 5
        assert digest(original['prompt']) == checked['prompt_sha256']
        assert digest(original['completion']) == checked['completion_sha256']
        assert digest(answer) == checked['answer_sha256']
        for ordinal, text in enumerate(refs, 1):
            key = digest(text)
            assert checked['reference_text_sha256'][ordinal - 1] == key
            references[key] = text
            reference_members[key].add(index)
            reference_ordinals[key].append([index, ordinal])
        if index not in retained:
            continue
        row = retained[index]
        assert row['original_response'] == answer and row['question'] == ''
        assert row['answer_sha256'] == digest(answer)
        assert row['prompt_sha256'] == row['original_prompt_sha256'] == digest(original['prompt'])
        assert row['completion_sha256'] == digest(original['completion'])
        assert row['reference_text_sha256'] == checked['reference_text_sha256']
        assert row['source_id'] == checked['reference_set_group_sha256']
        assert row['labels'] == checked['aligned_synthetic_spans']
        assert row['released_prompt_sha256'] == digest(row['released_prompt'])
        assert row['released_prompt'] + answer == original['prompt']
        start, end = row['evidence_original_prompt_range']
        assert row['released_prompt'][start:end] == row['retrieved_passages']
        assert re.findall(r'(?:^|\n)Reference \[(\d+)\]: ', row['retrieved_passages']) == ['1', '2', '3', '4', '5']
        assert row['synthetic_not_human_gold'] and row['unmarked_positions_silver_not_verified_negative']
        assert not row['edited_projection_used'] and not row['new_labels_generated']
        for label in row['labels']:
            assert label['type'] in {'entity', 'relation', 'invented', 'contradictory'}
            assert answer[label['start']:label['end']] == label['text']
            span_types[label['type']] += 1
    assert dict(span_types) == manifest['span_types']
    # Rebuild connected components as graph traversal instead of the staging union-find.
    adjacency = {index: set() for index in fixed}
    for members in reference_members.values():
        for index in members:
            adjacency[index].update(members - {index})
    components = []
    unseen = set(fixed)
    while unseen:
        stack = [min(unseen)]
        seen = set()
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            stack.extend(adjacency[index] - seen)
        unseen.difference_update(seen)
        components.append(sorted(seen))
    reported = rows(HERE / 'candidate_material_group_index.jsonl')
    assert sorted(components) == sorted(row['raw_indices'] for row in reported)
    for group in reported:
        expected_id = 'fava_material_' + digest('|'.join(f'fava_train_{i}' for i in group['raw_indices']))
        assert group['group_id'] == expected_id
        for index in group['raw_indices']:
            assert fixed[index]['group_id'] == expected_id
            if index in retained:
                assert retained[index]['group_id'] == expected_id
        assert group['quarantined'] == bool(set(group['raw_indices']) & set(quarantine))
        assert not group['quarantined'] or set(group['raw_indices']) <= set(quarantine)
    sources = {row['source_id']: row for row in rows(ROOT.parent / 'data/raw/source_info.jsonl')}
    identities = {row['source_id']: row for row in rows(HERE / 'rt_source_identity_index.jsonl')}
    blocked = {sid for sid, row in identities.items() if row['blocked_after_existing_edge_propagation']}
    assert len(sources) == len(identities) == 2965 and len(blocked) == 705
    matches = rows(HERE / 'material_match_index.jsonl')
    direct = set()
    empty_matches = 0
    for match in matches:
        sid = match['rt_source_id_witness']
        part = rules.evidence_parts(sources[sid])[match['rt_part_index']]
        ref = references[match['reference_text_sha256']]
        assert match['blocked'] == (sid in blocked)
        assert match['rt_material_group_id'] == identities[sid]['material_group_id']
        assert match['fava_members_raw_index_and_reference_ordinal'] == reference_ordinals[match['reference_text_sha256']]
        if match['reason'] == 'exact_evidence_part_SHA':
            assert part == ref
            empty_matches += int(ref == '')
        else:
            pw, rw = rules.norm(part), rules.norm(ref)
            p, r = match['rt_word_start'], match['fava_word_start']
            assert len(pw[p:p + 20]) == 20 and pw[p:p + 20] == rw[r:r + 20]
            assert digest(' '.join(pw[p:p + 20])) == match['shared20_sha256_witness']
        if match['blocked']:
            direct.update(reference_members[match['reference_text_sha256']])
    assert len(matches) == 21 and direct == {3705, 11525}
    assert direct == set(quarantine)
    # Fresh blocked-only matching check on every original reference. No response file is read.
    blocked_exact, blocked_grams = set(), set()
    for sid in blocked:
        for part in rules.evidence_parts(sources[sid]):
            blocked_exact.add(digest(part))
            words = rules.norm(part)
            blocked_grams.update(digest(' '.join(words[i:i + 20])) for i in range(len(words) - 19))
    direct_fresh = set()
    for key, ref in references.items():
        words = rules.norm(ref)
        hit = key in blocked_exact or any(digest(' '.join(words[i:i + 20])) in blocked_grams for i in range(len(words) - 19))
        if hit:
            direct_fresh.update(reference_members[key])
    assert direct_fresh == direct and not set(retained) & direct_fresh
    group_sizes = Counter(row['group_id'] for row in retained.values())
    assert len(group_sizes) == 7407 and max(group_sizes.values()) == 7
    for name, expected in manifest['artifacts_sha256'].items():
        assert sha(HERE / name) == expected
    result = {'passed': True, 'answers': len(retained), 'factual_spans': sum(span_types.values()),
              'span_types': dict(span_types), 'fixed_candidates': len(fixed), 'groups': len(group_sizes),
              'largest_material_group': max(group_sizes.values()), 'all_original_text_and_labels_exact': True,
              'all_original_five_references_preserved': True, 'group_graph_independently_rebuilt': True,
              'all_match_witnesses_verified': len(matches), 'blocked_only_matching_independently_rebuilt': True,
              'retained_blocked_material_matches': 0, 'quarantine_raw_indices': sorted(quarantine),
              'quarantine_notes': {'3705': 'Real consecutive 20-normalized-word evidence overlap.',
                                   '11525': 'Empty Reference 1 exact SHA matches empty RT parts. Conservative overblocking, not evidence of substantive leakage; frozen candidates unchanged.'},
              'empty_reference_match_records': empty_matches, 'source_identity_reconstruction_scope': 'Conservative confirmed-train complement, not complete original official split recovery.',
              'candidate_manifest_sha256': sha(HERE / 'manifest.json'), 'candidate_data_unchanged': True,
              'token_files_read_or_modified': False, 'raw_response_or_test_answers_read': False,
              'GPU_used': False, 'model_loaded': False, 'seconds': time.perf_counter() - started}
    (HERE / 'DATA_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'DATA_REPORT.md').write_text('''# FAVA factual synthetic auxiliary candidates

Data staging and its independent read-only audit completed. Fixed 7,483 factual-only, exactly aligned candidates became **7,481 answers in 7,407 material groups**, with 19,728 unchanged synthetic spans. Largest group: 7. No duplicate rows were removed.

The full original five Reference sections and corrupted answer are preserved. The question is empty. The stored released prompt is a FAVA checking-prompt prefix, not an original user question or native model trace. Corrections and edited projections are not model inputs. Unmarked positions are noisy silver negatives, not human-verified true statements.

Two rows were conservatively quarantined: raw 3,705 has actual consecutive-20-word evidence overlap; raw 11,525 has an empty first Reference whose exact hash matches empty RAGTruth material parts. The latter is overblocking, not proof of substantive leakage. Frozen candidate membership was retained.

All 7,481 original answers/labels/reference boundaries and all 21 recorded matching witnesses passed audit. An independent blocked-only material index found no retained overlap under the fixed exact/20-word rule. Source blocking uses 989 QA identities and the conservative complement of confirmed non-QA training identities; it does not claim recovery of every original official split. Shared-reference groups can connect distractors and do not prove event/entity independence. Paraphrase overlap remains outside this rule.

This stage loaded no tokenizer or model, used no GPU, performed no training, and read no official-test answers/labels. Root owns the separate tokenizer stage and its normalization handling. The original human QA training and evaluation files are unchanged.
''', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
