"""Bounded CPU checks; one failed FAVA row, synthetic accents, eight prior rows."""
from pathlib import Path
import sys
import json
import time
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q
import fava_nfc_character_map as new
import tail_finetune as old
import feature_qa as replay
import build_gold as gold
from transformers import AutoTokenizer
from tokenizers import NormalizedString, normalizers
import torch


def mass_check(text, offsets, m):
    actual = np.bincount(m[0], weights=m[2], minlength=len(offsets))
    expected = np.asarray([any(not c.isspace() for c in text[a:b]) for a,b in offsets])
    assert np.allclose(actual[expected], 1., rtol=0, atol=2e-7)
    assert not actual[~expected].any()


def main():
    assert not torch.cuda.is_initialized()
    started = time.perf_counter()
    bert = AutoTokenizer.from_pretrained(ROOT.parent / 'models/ModernBERT-base', local_files_only=True)
    normalizer = bert.backend_tokenizer.normalizer
    assert json.loads(normalizer.__getstate__()) == {'type': 'NFC'}
    normalized_examples = []
    for text in ('s\u030c', 'c\u0301', 'e\u0301', 'a\u0302\u0301'):
        ns = NormalizedString(text); normalizer.normalize(ns)
        normalized_examples.append({'original': ns.original, 'normalized': ns.normalized,
            'single_normalized_character_source_slice': ns[0:1].original})
    candidate = ROOT / 'auxiliary_fava_v1/candidate_fit.jsonl'
    with candidate.open(encoding='utf-8') as stream:
        row = next(r for r in map(json.loads, stream) if r['response_id'] == 'fava_train_2892')
    original_row_hash = q.digest(row)
    text = row['original_response']
    assert '\u030c' in text and '\u0301' in text
    llama = AutoTokenizer.from_pretrained(replay.MODEL, local_files_only=True)
    view = replay.encode_view(llama, row['released_prompt'], text, row['evidence_original_prompt_range'])
    raw = view['response_token_offsets']
    prefix = row['retrieved_passages']+bert.sep_token+bert.sep_token
    encoded = bert(prefix+text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    eo = np.asarray(encoded['offset_mapping'], np.int64)
    inside = (eo[:,1] > len(prefix)) & (eo[:,0] < len(prefix)+len(text))
    start = np.where(inside, np.maximum(0, eo[:,0]-len(prefix)), -1)
    end = np.where(inside, np.minimum(len(text), eo[:,1]-len(prefix)), -1)
    try: old.character_map(text, raw, start, end)
    except AssertionError as e: legacy_failure = str(e)
    else: raise AssertionError('Expected original2892 failure did not reproduce')
    m, diagnostics = new.character_map(text, raw, start, end, normalizer=normalizer, return_diagnostics=True)
    assert diagnostics['repaired_character_count'] == 2
    assert [p['missing_character_index'] for p in diagnostics['repairs']] == [287,290]
    mass_check(text, raw, m)
    assert q.digest(row) == original_row_hash and gold.digest_text(text) == row['answer_sha256']
    proof = {'response_id': row['response_id'], 'input_tokens': len(encoded['input_ids']),
        'raw_tokens': len(raw), 'legacy_failure_reproduced': legacy_failure, 'diagnostics': diagnostics,
        'input_ids': encoded['input_ids'], 'answer_encoder_start': start.tolist(), 'answer_encoder_end': end.tolist(),
        'response_token_offsets': raw, 'mapping': [x.tolist() for x in m],
        'original_row_sha256': original_row_hash, 'candidate_file_sha256': q.sha(candidate),
        'answer_and_labels_unchanged': True, 'all_nonwhitespace_raw_mass_one': True}
    q.save(OUT / 'REAL_2892_PROOF.json', proof)
    synthetic = []
    for text in ('Cafe\u0301.', 'Pas\u030cic\u0301', 'A\u030A', 'a\u0302\u0301', 'x e\u0301 y',
                 'ordinary ASCII 123.', 'Already café / Čech.', '中文，123', 'one\n two\tthree'):
        enc = bert(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
        begin = [a if a < b else -1 for a,b in enc['offset_mapping']]
        finish = [b if a < b else -1 for a,b in enc['offset_mapping']]
        offsets = [(i,i+1) for i in range(len(text))]
        mm, detail = new.character_map(text, offsets, begin, finish, normalizer=normalizer, return_diagnostics=True)
        mass_check(text, offsets, mm)
        if detail['used_legacy_path']:
            baseline = old.character_map(text, offsets, begin, finish)
            assert all(np.array_equal(a,b) for a,b in zip(mm,baseline))
        synthetic.append({'text': text, **detail})
    duplicate, duplicate_detail = new.character_map('s\u030c', [(0,1),(1,2)], [0,0], [1,1],
        normalizer=normalizer, return_diagnostics=True)
    assert np.array_equal(duplicate[0], [0,0,1,1])
    assert np.array_equal(duplicate[1], [0,1,0,1])
    assert np.array_equal(duplicate[2], [.5,.5,.5,.5])
    rejected = []
    for name,text,start,end,norm in (
        ('missing_ordinary_letter', 'abc', [0],[2], normalizer),
        ('uncomposed_accent', 'q\u0301', [0],[1], normalizer),
        ('leading_accent_without_starter', '\u0301a', [1],[2], normalizer),
        ('non_NFC_normalizer', 'abc', [0],[3], normalizers.NFKC())):
        try: new.character_map(text, [(0,len(text))], start, end, normalizer=norm)
        except ValueError as exc: rejected.append({'case': name, 'reason': str(exc)})
        else: raise AssertionError('Unproven gap incorrectly accepted: '+name)
    exact = []
    # Only eight already-exported ordinary rows: no full dataset retokenization.
    with (ROOT / 'auxiliary_fava_v1/token_inputs.jsonl').open(encoding='utf-8') as stream:
        prior = [json.loads(next(stream)) for _ in range(8)]
    prior_ids = {r['response_id'] for r in prior}; texts = {}
    with candidate.open(encoding='utf-8') as stream:
        for r in map(json.loads, stream):
            if r['response_id'] in prior_ids: texts[r['response_id']] = r['original_response']
            if len(texts) == len(prior): break
    for r in prior:
        text = texts[r['response_id']]
        a,detail = new.character_map(text, r['response_token_offsets'], r['answer_encoder_start'],
            r['answer_encoder_end'], normalizer=normalizer, return_diagnostics=True)
        b = old.character_map(text, r['response_token_offsets'], r['answer_encoder_start'], r['answer_encoder_end'])
        assert detail['used_legacy_path'] and all(np.array_equal(x,y) for x,y in zip(a,b))
        assert all(np.array_equal(x,y) for x,y in zip(a,r['mapping']))
        exact.append(r['response_id'])
    assert not torch.cuda.is_initialized()
    report = {'passed': True, 'normalized_string_examples': normalized_examples,
        'real2892_exactly_two_proven_repairs': diagnostics,
        'synthetic_checks': synthetic, 'duplicate_owner_half_weights_passed': True,
        'unproven_cases_rejected': rejected, 'previous_eight_FAVA_rows_exact_legacy_and_saved': exact,
        'seconds': time.perf_counter()-started, 'GPU_used': False, 'model_loaded': False,
        'full_tokenization_run': False, 'old_QA_mapping_modified': False,
        'candidate_text_labels_modified': False,
        'source_sha256': {str(Path(p).resolve()): q.sha(p) for p in
            (Path(__file__), new.__file__, old.__file__, ROOT.parent / 'models/ModernBERT-base/tokenizer.json')}}
    q.save(OUT / 'CPU_CHECK.json', report)
    print('FAVA_NFC_CHECK_PASSED', diagnostics, 'old_exact_rows', len(exact), flush=True)


if __name__ == '__main__': main()
