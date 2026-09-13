"""CPU-only auxiliary human-label/input alignment; no fitting or test data."""
from collections import Counter
from pathlib import Path
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer
import feature_qa as replay
import build_gold as gold
import tail_finetune as mapping
import run_development as q

ROOT = q.ROOT
OUT = ROOT / 'auxiliary_human_v1'
MODEL = ROOT.parent / 'models/ModernBERT-base'


def main():
    assert not torch.cuda.is_initialized()
    target = OUT / 'token_input_preparation.json'
    assert not target.exists() and not (OUT / 'token_inputs.jsonl').exists()
    manifest = q.read(OUT / 'manifest.json')
    assert manifest['status'] == 'staged_not_used_in_training' and manifest['span_alignment_issues'] == 0
    assert q.sha(OUT / 'candidate_fit.jsonl') == manifest['artifacts_sha256']['candidate_fit.jsonl']
    rows = q.lines(OUT / 'candidate_fit.jsonl')
    llama = AutoTokenizer.from_pretrained(replay.MODEL, local_files_only=True)
    bert = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    stats = {task: Counter() for task in ('Summary', 'Data2txt')}
    lengths = {task: [] for task in stats}
    exceptions = []
    start_time = time.time()
    with (OUT / 'token_inputs.jsonl').open('w', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            task = row['task_type']
            text = row['original_response']
            view = replay.encode_view(llama, row['released_prompt'], text, row['evidence_original_prompt_range'])
            offsets = view['response_token_offsets']
            _, _, lexical, risk = gold.map_characters(text, row['labels'], offsets)
            prefix = row['retrieved_passages'] + bert.sep_token + row['question'] + bert.sep_token
            whole = prefix + text
            encoded = bert(whole, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
            eo = np.asarray(encoded['offset_mapping'], np.int64)
            inside = (eo[:, 1] > len(prefix)) & (eo[:, 0] < len(whole))
            begin = np.where(inside, np.maximum(0, eo[:, 0] - len(prefix)), -1)
            end = np.where(inside, np.minimum(len(text), eo[:, 1] - len(prefix)), -1)
            charmap = mapping.character_map(text, offsets, begin, end)
            n = len(offsets)
            sums = np.bincount(charmap[0], weights=charmap[2], minlength=n)
            nons = np.array([any(not c.isspace() for c in text[a:b]) for a, b in offsets])
            assert np.max(np.abs(sums[nons] - 1)) < 2e-7
            assert not sums[~nons].any()
            # Independent interval oracle on a fixed every97th response; data selection does not use gold.
            if i % 97 == 0:
                gold.independent_label_check(text, row['labels'], offsets, lexical, risk)
            if len(encoded['input_ids']) > bert.model_max_length or not any(lexical) or bool(row['labels']) != bool(any(risk)):
                exceptions.append({'response_id': row['response_id'], 'input_tokens': len(encoded['input_ids']),
                                   'lexical_tokens': sum(lexical), 'original_answer_risk': bool(row['labels']),
                                   'lexical_risk': bool(any(risk))})
            windows = gold.windows_for_count(n)
            eligible = [(a, b) for a, b in windows if any(lexical[a:b])]
            st = stats[task]
            st.update(answers=1, encoder_input_tokens=len(encoded['input_ids']), raw_answer_tokens=n,
                      lexical_answer_tokens=sum(lexical), risk_answer_tokens=sum(risk),
                      windows=len(eligible), risk_windows=sum(any(risk[a:b]) for a, b in eligible))
            lengths[task].append(len(encoded['input_ids']))
            stream.write(json.dumps({'response_id': row['response_id'], 'source_id': row['source_id'],
                'group_id': row['group_id'], 'task_type': task, 'answer_sha256': row['answer_sha256'],
                'input_ids': encoded['input_ids'], 'answer_encoder_start': begin.tolist(),
                'answer_encoder_end': end.tolist(), 'mapping': [x.tolist() for x in charmap],
                'response_token_ids': view['answer_token_ids'], 'response_token_offsets': offsets,
                'response_token_offsets_raw': view['response_token_offsets_raw'],
                'lexical_mask': lexical, 'risk_mask': risk, 'answer_risk': int(bool(row['labels']))},
                ensure_ascii=False) + '\n')
            if (i + 1) % 1000 == 0:
                print('AUX_INPUTS', i + 1, round(time.time() - start_time, 1), flush=True)
    for task, values in lengths.items():
        stats[task]['max_input_tokens'] = max(values)
        stats[task]['median_input_tokens'] = float(np.median(values))
    report = {'status': 'prepared_not_trained' if not exceptions else 'review_required_no_silent_filtering',
        'answers': len(rows), 'stats': {k: dict(v) for k, v in stats.items()},
        'exceptions': exceptions, 'no_truncation': True, 'GPU_used': False,
        'labels': 'Unchanged released human spans mapped using the exact original QA gold routine and Llama tokenizer/wrapper. Four-raw-BPE windows retained for descriptive counts.',
        'input': 'Entire original evidence + ModernBERT SEP + original task instruction + SEP + original answer; CLS/SEP supplied by tokenizer.',
        'purpose': 'Auxiliary candidate training preparation only. No change to QA fit/cal/test or selection; no model has used these rows yet.',
        'mapping': 'Same nonwhitespace character-average logit mapping as the prepared QA ModernBERT baseline.',
        'wall_seconds': time.time() - start_time,
        'sources_sha256': {str(p.resolve()): q.sha(p) for p in [OUT / 'manifest.json', OUT / 'candidate_fit.jsonl',
                             Path(__file__), Path(gold.__file__), Path(replay.__file__), Path(mapping.__file__)]},
        'tokenizer_signature': replay.tokenizer_signature(llama),
        'modernbert_tokenizer_sha256': q.sha(MODEL / 'tokenizer.json'),
        'token_inputs_sha256': q.sha(OUT / 'token_inputs.jsonl')}
    q.save(target, report)
    print(json.dumps({k: report[k] for k in ('status', 'answers', 'stats', 'exceptions', 'wall_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
