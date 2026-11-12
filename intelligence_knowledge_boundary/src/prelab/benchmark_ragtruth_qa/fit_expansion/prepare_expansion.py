"""CPU-only exact-text deduplication, fixed Llama coordinates and MiniCheck plans."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import importlib.util
import hashlib
import json
import shutil
from collections import Counter, defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
QA = HERE.parent
DATA = HERE/'data'
SEM = HERE/'minicheck'
OLDSEM = QA/'semantic_baseline/cuda_variant'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


inv = load('expansion_inventory', HERE/'inspect_fit_expansion.py')
gold = load('expansion_gold_helpers', QA/'src/build_gold.py')
layout = load('expansion_layout_helpers', QA/'src/feature_qa.py')
sem = load('expansion_minicheck_helpers', QA/'semantic_baseline/run_semantic.py')
sha, readl, writej, writel = inv.sha, inv.readl, inv.writej, inv.writel


def source_paths():
    files = [HERE/'manifest.json', HERE/'additional_good_candidates.jsonl', HERE/'fit_source_whitelist.jsonl',
             Path(__file__), HERE/'run_minicheck_expansion.py', HERE/'protocol.json',
             QA/'src/build_gold.py', QA/'src/feature_qa.py',
             QA/'semantic_baseline/run_semantic.py', QA/'semantic_baseline/run_semantic_cuda.py',
             QA/'semantic_baseline/download_manifest.json',
             OLDSEM/'claim_feature_manifest.json', OLDSEM/'inference_complete.json',
             OLDSEM/'protocol.json', OLDSEM/'plans.jsonl',
             QA/'data/fit.jsonl', QA/'data/gold_manifest.json', QA/'data/development_manifest.json']
    files += [QA/'data'/f'{p}_fit.jsonl' for p in ('answers', 'tokens', 'windows_k4', 'windows_excluded')]
    # Byte hashes only; calibration contents/labels are not parsed or changed.
    files += [QA/'data'/n for n in ('calibration.jsonl', 'answers_calibration.jsonl',
                                  'tokens_calibration.jsonl', 'windows_k4_calibration.jsonl')]
    return {str(p.resolve()): sha(p) for p in files}


def plan_claims(row, tok, punkt, stats):
    # Exactly the frozen MiniCheck text preparation; labels/model identity absent.
    text, doc = row['original_response'], row['retrieved_passages']
    segments = []
    for si, (a, b) in enumerate(sem.sentences(text, punkt)):
        pieces = sem.bounded(text, a, b, 96, tok)
        stats['long_claim_sentences_split'] += len(pieces) > 1
        for x, y in pieces:
            segments.append({'start': x, 'end': y, 'text': text[x:y], 'sentence_index': si,
                             'token_count': len(tok.encode(text[x:y], add_special_tokens=False))})
    raw_doc = []
    for a, b in sem.sentences(doc, punkt):
        pieces = sem.bounded(doc, a, b, 400, tok)
        stats['long_document_sentences_split'] += len(pieces) > 1
        raw_doc.extend(pieces)
    ds = []
    for a, b in raw_doc:
        if ds and len(tok.encode(doc[ds[-1][0]:b], add_special_tokens=False)) <= 400:
            ds[-1] = (ds[-1][0], b)
        else:
            ds.append((a, b))
    chunks = [{'start': a, 'end': b, 'text': doc[a:b],
               'token_count': len(tok.encode(doc[a:b], add_special_tokens=False))} for a, b in ds]
    assert segments and chunks
    assert sem.coverage(text, [(s['start'], s['end']) for s in segments]) == sem.coverage(doc, ds) == 0
    lengths = [len(tok.encode(d['text']+tok.eos_token+s['text'])) for s in segments for d in chunks]
    assert max(lengths) <= 512
    stats.update({'answers': 1, 'claims': len(segments), 'document_chunks': len(chunks), 'pairs': len(lengths)})
    stats['max_input_tokens'] = max(stats['max_input_tokens'], max(lengths))
    return {'response_id': row['response_id'], 'partition': 'fit', 'group_id': row['group_id'],
            'answer_sha256': row['answer_sha256'], 'document_sha256': hashlib.sha256(doc.encode()).hexdigest(),
            'claims': segments, 'document_chunks': chunks, 'pair_lengths': lengths,
            'uncovered_nonwhitespace_chars': 0, 'truncated_input_tokens': 0}


def record_gold(row, plan):
    original, covered = gold.check_fixed_layout(row, plan)
    text, labels = row['original_response'], row['labels']
    offsets = original['response_token_offsets']; n = len(offsets)
    chars, riskchars, lexical, risk = gold.map_characters(text, labels, offsets)
    gold.independent_label_check(text, labels, offsets, lexical, risk)
    ident = {k: row[k] for k in ('response_id', 'source_id', 'group_id', 'partition')}
    ident['answer_id'] = row['response_id']; rid = row['response_id']; flags = []
    span_tokens = []
    for i, s in enumerate(labels):
        a, b = s['start'], s['end']
        if a == b: flags.append('zero_length_span')
        if not any(chars[a:b]): flags.append('span_without_isalnum')
        assert all(covered[j] or text[j].isspace() for j in range(a, b))
        inds = [j for j, (x, y) in enumerate(offsets) if any(
            text[c].isalnum() for c in range(max(x, a), min(y, b)))]
        span_tokens.append({'span_index': i, 'original_start': a, 'original_end': b, 'risk_token_indices': inds})
    if labels and not any(risk): flags.append('answer_risk_without_risk_token')
    token = {**ident, 'original_response': text, 'answer_sha256': row['answer_sha256'],
        'original_labels': labels, 'token_count': n, 'token_ids': original['answer_token_ids'],
        'answer_token_positions': original['answer_token_positions'],
        'response_token_offsets_raw': original['response_token_offsets_raw'], 'response_token_offsets': offsets,
        'lexical_mask': lexical, 'risk_mask': risk, 'risk_character_spans': gold.character_spans(text, riskchars),
        'token_risk_character_spans': [gold.character_spans(text, riskchars, a, b) if risk[j] else []
                                     for j, (a, b) in enumerate(offsets)],
        'span_token_mapping': span_tokens, 'answer_risk': int(bool(labels)),
        'first_answer_token_preserved': True, 'edge_case_flags': sorted(set(flags))}
    windows, excluded = [], []
    for a, b in gold.windows_for_count(n):
        ids = list(range(a, b)); li = [j for j in ids if lexical[j]]; ri = [j for j in ids if risk[j]]
        intervals = gold.merge_intervals(offsets[a:b]); left = intervals[0][0]; right = intervals[-1][1]
        riskspans = gold.merge_intervals([(x['start'], x['end']) for j in ids
                                        for x in token['token_risk_character_spans'][j]])
        # Character-union oracle, independent of risk-mask OR.
        oracle = any(text[c].isalnum() and any(s['start'] <= c < s['end'] for s in labels)
                     for j in ids for c in range(*offsets[j]))
        assert bool(ri) == oracle
        w = {**ident, 'window_id': f'{rid}__k4_{a:05d}', 'k': 4, 'stride': 1,
            'token_start': a, 'token_end': b, 'token_indices': ids,
            'answer_token_positions': original['answer_token_positions'][a:b],
            'token_ids': original['answer_token_ids'][a:b], 'character_intervals': intervals,
            'char_start': left, 'char_end': right, 'bounding_text': text[left:right],
            'lexical_token_indices': li, 'risk_token_indices': ri,
            'risk_character_spans': [{'start': x, 'end': y, 'text': text[x:y]} for x, y in riskspans],
            'eligible': bool(li), 'label': int(bool(ri)) if li else None}
        if li: windows.append(w)
        else:
            w['exclusion_reason'] = 'no_lexical_token'; excluded.append(w)
    if not windows: flags.append('no_eligible_window')
    answer = {**ident, 'original_response': text, 'answer_sha256': row['answer_sha256'], 'original_labels': labels,
        'quality': 'good', 'eligible': True, 'label': int(bool(labels)), 'token_count': n,
        'lexical_token_count': sum(lexical), 'risk_token_count': sum(risk), 'official_span_count': len(labels),
        'candidate_window_count': len(windows)+len(excluded), 'eligible_window_count': len(windows),
        'positive_window_count': sum(w['label'] for w in windows), 'excluded_window_count': len(excluded),
        'edge_case_flags': sorted(set(flags))}
    return token, windows, excluded, answer


def prepare():
    assert not (DATA/'export_freeze.json').exists(), 'Export already frozen; do not overwrite'
    assert not layout.torch.cuda.is_initialized()
    snap = source_paths()
    inventory = sem.read(HERE/'manifest.json')
    for name, expected in inventory['files_sha256'].items(): assert sha(HERE/name) == expected
    old = readl(QA/'data/fit.jsonl'); new = readl(HERE/'additional_good_candidates.jsonl')
    assert len(old) == 634 and len(new) == 3078
    oldids = {r['response_id'] for r in old}; sources = {r['source_id']: r for r in old}
    groups = defaultdict(list)
    for r in old+new: groups[(r['source_id'], r['answer_sha256'])].append(r)
    aliases, conflicts, addition = [], [], []
    for (sid, ah), rs in sorted(groups.items()):
        assert len({r['original_response'] for r in rs}) == 1
        rs = sorted(rs, key=lambda r: (r['response_id'] not in oldids, int(r['response_id'])))
        chosen = rs[0]
        equal = all(r['labels'] == chosen['labels'] for r in rs)
        if not equal:
            conflicts.append({'source_id': sid, 'answer_sha256': ah, 'records': rs,
                              'action': 'Retain original fit record if present; additional aliases quarantined, no relabeling'})
            if chosen['response_id'] not in oldids: continue
        aliases.append({'canonical_response_id': chosen['response_id'], 'source_id': sid,
                        'group_id': sources[sid]['group_id'], 'answer_sha256': ah,
                        'full_labels_equal': equal, 'retained_original_fit': chosen['response_id'] in oldids,
                        'provenance': [{'response_id': r['response_id'], 'model': r['model'],
                                        'temperature': r['temperature'], 'quality': r['quality'],
                                        'labels_sha256': sem.digest(r['labels'])} for r in rs]})
        if chosen['response_id'] in oldids: continue
        orig = sources[sid]
        row = {k: orig[k] for k in ('source_id', 'group_id', 'partition', 'official_split', 'question',
                                    'retrieved_passages', 'released_prompt', 'prompt_sha256')}
        row.update({k: chosen[k] for k in ('response_id', 'model', 'temperature', 'quality', 'original_response',
                                          'labels', 'answer_sha256', 'span_integrity_issues', 'official_quality_eligible')})
        row.update({'annotation_origin': 'RAGTruth released human span annotation, unchanged',
                    'label_offsets': 'original response character offsets, end exclusive; no text normalization',
                    'feature_status': 'Llama tokenizer coordinates only; no Llama forward/native trajectory'})
        addition.append(row)
    DATA.mkdir(exist_ok=True); SEM.mkdir(exist_ok=True)
    writel(DATA/'label_conflicts.jsonl', conflicts)
    writel(DATA/'answer_provenance.jsonl', aliases)
    addition.sort(key=lambda r: (r['source_id'], int(r['response_id'])))
    writel(DATA/'new_fit.jsonl', addition)
    oldbytes = (QA/'data/fit.jsonl').read_bytes()
    assert oldbytes.endswith(b'\n')
    (DATA/'fit.jsonl').write_bytes(oldbytes+(DATA/'new_fit.jsonl').read_bytes())
    assert len(addition)+634 == len(aliases)
    # Always copy unchanged original gold first; new rows follow with the same schema.
    handles = {}
    for stem in ('answers', 'tokens', 'windows_k4', 'windows_excluded'):
        target = DATA/f'{stem}_fit.jsonl'
        target.write_bytes((QA/'data'/target.name).read_bytes())
        handles[stem] = target.open('a', encoding='utf-8', newline='\n')
    ll = layout.AutoTokenizer.from_pretrained(layout.MODEL, local_files_only=True, use_fast=True, trust_remote_code=False)
    tokenizer_sig = layout.tokenizer_signature(ll)
    import nltk
    from nltk.tokenize.punkt import PunktTokenizer
    nltk.data.path.insert(0, str(QA/'semantic_baseline/nltk_data'))
    punkt = PunktTokenizer('english'); mt = sem.tokenizer()
    for rel, expected in sem.read(QA/'semantic_baseline/download_manifest.json')['files_sha256'].items():
        assert sha(QA/'semantic_baseline'/rel) == expected
    counts = Counter(); stats = Counter(); miniplans = []; tokenplans = []
    try:
        for i, row in enumerate(addition):
            # Only visible text and IDs enter tokenization. Generator/labels do not.
            visible = {k: row[k] for k in ('response_id', 'source_id', 'group_id', 'partition', 'official_split',
                                           'released_prompt', 'original_response', 'retrieved_passages')}
            plan = layout.prepare_row(ll, visible, include_no_context=False)
            tokenplans.append(plan)
            token, windows, excluded, answer = record_gold(row, plan)
            for stem, records in [('tokens', [token]), ('answers', [answer]),
                                  ('windows_k4', windows), ('windows_excluded', excluded)]:
                for r in records: gold.dump_line(handles[stem], r)
            counts.update({'answers': 1, 'risk_answers': answer['label'], 'raw_tokens': answer['token_count'],
                           'lexical_tokens': answer['lexical_token_count'], 'risk_tokens': answer['risk_token_count'],
                           'official_spans': answer['official_span_count'], 'eligible_windows': len(windows),
                           'risk_windows': answer['positive_window_count'], 'excluded_nonlexical_windows': len(excluded)})
            miniplans.append(plan_claims(row, mt, punkt, stats))
            if (i+1) % 250 == 0: print('EXPANSION_CPU_PREPARED', i+1, len(addition), flush=True)
    finally:
        for h in handles.values(): h.close()
    writel(DATA/'new_token_plans.jsonl', tokenplans)
    writel(SEM/'new_plans.jsonl', miniplans)
    # Retokenize original fit only to verify complete-template coordinates are unchanged.
    oldtokens = {r['response_id']: r for r in readl(QA/'data/tokens_fit.jsonl')}
    for row in old:
        v = layout.prepare_row(ll, row, include_no_context=False)['original']; t = oldtokens[row['response_id']]
        assert v['answer_token_ids'] == t['token_ids'] and v['response_token_offsets'] == t['response_token_offsets']
        assert v['answer_token_positions'] == t['answer_token_positions']
    reuse = []
    cache_manifest = sem.read(OLDSEM/'claim_feature_manifest.json')
    cache_index = {r['response_id']: r for r in cache_manifest['records']}
    for row in old:
        rid = row['response_id']; meta = OLDSEM/'claim_features'/f'{rid}.json'
        array = meta.with_suffix('.npz'); score = OLDSEM/'scores'/f'{rid}.json'; ref = cache_index[rid]
        assert sha(meta) == ref['metadata_sha256'] and sha(array) == ref['npz_sha256']
        m = sem.read(meta); assert m['original_answer_sha256'] == row['answer_sha256']
        assert sha(score) == m['score_row_sha256']
        reuse.append({'response_id': rid, 'source_id': row['source_id'], 'group_id': row['group_id'],
                      'metadata_path': str(meta.resolve()), 'npz_path': str(array.resolve()),
                      'score_path': str(score.resolve()), 'metadata_sha256': sha(meta),
                      'npz_sha256': ref['npz_sha256'], 'score_sha256': sha(score), 'tokens': m['tokens']})
    writej(SEM/'reuse_manifest.json', {'status': 'reuse_original_fit_cache_only', 'records': reuse,
                                     'answers': 634, 'old_manifest_sha256': sha(OLDSEM/'claim_feature_manifest.json')})
    # Document choice is not known until the frozen checker's forward. Bound it
    # using every planned doc length, never labels or assumed winner documents.
    selected_min = selected_max = claim_tokens = 0
    for p in miniplans:
        lens = np.asarray(p['pair_lengths']).reshape(len(p['claims']), len(p['document_chunks']))
        selected_min += int(lens.min(1).sum()); selected_max += int(lens.max(1).sum())
        claim_tokens += sum(c['token_count'] for c in p['claims'])
    storage = {'selected_encoder22_valid_input_tokens_lower': selected_min,
               'selected_encoder22_valid_input_tokens_upper': selected_max,
               'encoder22_float32_hidden_bytes_lower': selected_min*1024*4,
               'encoder22_float32_hidden_bytes_upper': selected_max*1024*4,
               'claim_last_hidden_bytes_approx': claim_tokens*1024*4,
               'metadata_and_coordinates_additional': True,
               'disk_free_bytes_at_preparation': shutil.disk_usage(HERE).free,
               'choice': 'Bounds over each claim all-document input lengths; actual argmax remains unknown',
               'old_634_encoder22_not_included': True}
    assert storage['disk_free_bytes_at_preparation'] > (selected_max+claim_tokens)*1024*4*1.2
    writej(SEM/'storage_estimate.json', storage)
    writej(DATA/'preparation_statistics.json', {'new': dict(counts), 'minicheck_new': dict(stats),
        'original_answers_preserved': 634, 'combined_answers': 634+len(addition), 'sources': 634, 'groups': 615,
        'duplicate_excess_merged': 3712-len(aliases), 'full_label_conflict_groups': len(conflicts),
        'all_original_fit_token_coordinates_reproduced': True, 'truncated_tokens': 0,
        'new_sources_or_groups': 0, 'features_extracted': False, 'models_trained': False})
    writej(DATA/'tokenizer_signature.json', tokenizer_sig)
    assert not layout.torch.cuda.is_initialized()
    assert source_paths() == snap
    outputs = list(DATA.glob('*.jsonl'))+[DATA/'preparation_statistics.json', DATA/'tokenizer_signature.json',
                                        SEM/'new_plans.jsonl', SEM/'reuse_manifest.json', SEM/'storage_estimate.json']
    writej(DATA/'export_freeze.json', {'status': 'frozen_cpu_export_waiting_for_gpu_authorization',
        'source_files_sha256': snap, 'output_files_sha256': {str(p.resolve()): sha(p) for p in outputs},
        'new_answers': len(addition), 'retained_answers': 634, 'combined_answers': len(addition)+634,
        'calibration_unchanged_and_not_parsed': True, 'test_not_opened': True,
        'no_llama_forward': True, 'generator_identity_metadata_only': True,
        'duplicate_policy': 'Same source+exact text; preserve old fit ID first, otherwise numeric minimum response ID; all provenance retained; unequal FULL official labels quarantined',
        'minicheck_next_stage': 'New deduplicated answers only; same fixed float32 CUDA checker, no training'})
    print(json.dumps(sem.read(DATA/'preparation_statistics.json'), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
