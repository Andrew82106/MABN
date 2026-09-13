"""Map isolated, exact-aligned FAVA silver training spans; no model training."""
from pathlib import Path
from collections import Counter
import argparse
import json
import time
import numpy as np
from transformers import AutoTokenizer
import feature_qa as replay
import build_gold as gold
import tail_finetune as mapping
import run_development as q

OUT = q.ROOT / 'auxiliary_fava_v1'
MODEL = q.ROOT.parent / 'models/ModernBERT-base'
FACT_TYPES = {'entity', 'relation', 'invented', 'contradictory'}


def protocol():
    return {'version': 'fava-silver-token-preparation-v1',
        'source': 'Only previously staged exact-aligned factual-only official FAVA TRAIN candidates after source quarantine; retain synthetic provenance.',
        'input': 'Five original references + ModernBERT SEP + empty question + SEP + original corrupted answer; no correction, markup, original-label type or generator ID enters model input.',
        'raw_tokenization': 'Use existing Llama chat-wrapper encode_view only for consistent raw-token/offset machinery, with the released FAVA checking-prompt prefix. This is auxiliary tokenization, not an observed native generation trace or a fabricated factual user question.',
        'labels': 'Original four factual-candidate type spans are silver positives; other lexical positions are silver negatives under the released synthetic annotation, not verified truth. No edited projection is used.',
        'mapping': 'Unchanged QA nonwhitespace character-average map from whole-input encoder positions to raw tokens. Main QA labels, tokenization, windows, train/cal lists remain unchanged.',
        'limits': 'No span repair, whitespace normalization, answer rewrite, silent truncation or dropped exception. Any exception leaves preparation requiring review, not ready for training.',
        'GPU_used': False, 'trained': False, 'official_test_answers_or_labels_read': False}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'token_design_freeze.json').exists()
    q.save(OUT/'token_protocol.json', protocol())
    q.save(OUT/'token_design_freeze.json', {'protocol_sha256': q.sha(OUT/'token_protocol.json'),
        'source_sha256': {str(p.resolve()): q.sha(p) for p in [Path(__file__), Path(replay.__file__), Path(gold.__file__), Path(mapping.__file__), MODEL/'tokenizer.json']},
        'new_training': False, 'GPU_used': False})
    print('FAVA_TOKEN_PROTOCOL_FROZEN_NO_DATA_OR_FIT', flush=True)


def run():
    frozen = q.read(OUT/'token_design_freeze.json')
    assert q.read(OUT/'token_protocol.json') == protocol()
    assert q.sha(OUT/'token_protocol.json') == frozen['protocol_sha256']
    for path, digest in frozen['source_sha256'].items(): assert q.sha(Path(path)) == digest
    assert not (OUT/'token_inputs.jsonl').exists()
    manifest = q.read(OUT/'manifest.json')
    source_path = OUT/'candidate_fit.jsonl'
    # The independent data stage must finish before this command can be invoked.
    complete = q.read(OUT/'complete.json')
    assert q.sha(OUT/'manifest.json') == complete['manifest_sha256']
    assert q.sha(source_path) == manifest['artifacts_sha256']['candidate_fit.jsonl']
    rows = q.lines(source_path)
    assert len(rows) == manifest['answers'] and rows
    llama = AutoTokenizer.from_pretrained(replay.MODEL, local_files_only=True)
    bert = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    counts = Counter(); lengths = []; exceptions = []; started = time.perf_counter()
    with (OUT/'token_inputs.jsonl').open('w', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            assert row['task_type'] == 'FAVA_synthetic' and row['question'] == ''
            assert row['labels'] and {label['type'] for label in row['labels']} <= FACT_TYPES
            text = row['original_response']
            assert gold.digest_text(text) == row['answer_sha256']
            view = replay.encode_view(llama, row['released_prompt'], text, row['evidence_original_prompt_range'])
            offsets = view['response_token_offsets']
            _, _, lexical, risk = gold.map_characters(text, row['labels'], offsets)
            prefix = row['retrieved_passages'] + bert.sep_token + bert.sep_token
            encoded = bert(prefix + text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
            eo = np.asarray(encoded['offset_mapping'], np.int64)
            inside = (eo[:, 1] > len(prefix)) & (eo[:, 0] < len(prefix) + len(text))
            begin = np.where(inside, np.maximum(0, eo[:, 0] - len(prefix)), -1)
            end = np.where(inside, np.minimum(len(text), eo[:, 1] - len(prefix)), -1)
            charmap = mapping.character_map(text, offsets, begin, end)
            n = len(offsets)
            sums = np.bincount(charmap[0], weights=charmap[2], minlength=n)
            nonspace = np.array([any(not c.isspace() for c in text[a:b]) for a, b in offsets])
            assert np.max(np.abs(sums[nonspace] - 1)) < 2e-7 and not sums[~nonspace].any()
            if i % 97 == 0: gold.independent_label_check(text, row['labels'], offsets, lexical, risk)
            if len(encoded['input_ids']) > 8192 or not any(lexical) or not any(risk):
                exceptions.append({'response_id': row['response_id'], 'input_tokens': len(encoded['input_ids']),
                    'lexical_tokens': sum(lexical), 'silver_risk_tokens': sum(risk)})
            eligible = [(a, b) for a, b in gold.windows_for_count(n) if any(lexical[a:b])]
            counts.update(answers=1, encoder_input_tokens=len(encoded['input_ids']), raw_answer_tokens=n,
                lexical_answer_tokens=sum(lexical), risk_answer_tokens=sum(risk), windows=len(eligible),
                risk_windows=sum(any(risk[a:b]) for a, b in eligible))
            lengths.append(len(encoded['input_ids']))
            stream.write(json.dumps({'response_id': row['response_id'], 'source_id': row['source_id'],
                'group_id': row['group_id'], 'task_type': 'FAVA_synthetic', 'answer_sha256': row['answer_sha256'],
                'input_ids': encoded['input_ids'], 'answer_encoder_start': begin.tolist(), 'answer_encoder_end': end.tolist(),
                'mapping': [x.tolist() for x in charmap], 'response_token_ids': view['answer_token_ids'],
                'response_token_offsets': offsets, 'response_token_offsets_raw': view['response_token_offsets_raw'],
                'lexical_mask': lexical, 'risk_mask': risk, 'answer_risk': 1,
                'synthetic_not_human_gold': True, 'unmarked_positions_silver_not_verified_negative': True}, ensure_ascii=False)+'\n')
            if (i+1) % 1000 == 0: print('FAVA_INPUTS', i+1, round(time.perf_counter()-started, 1), flush=True)
    counts.update(max_input_tokens=max(lengths))
    report = {'status': 'prepared_not_trained' if not exceptions else 'review_required_no_silent_filtering',
        'answers': len(rows), 'stats': dict(counts), 'median_input_tokens': float(np.median(lengths)),
        'exceptions': exceptions, 'no_truncation': True, 'GPU_used': False, 'trained': False,
        'manifest_sha256': q.sha(OUT/'manifest.json'), 'candidate_sha256': q.sha(source_path),
        'token_design_sha256': q.sha(OUT/'token_design_freeze.json'), 'token_inputs_sha256': q.sha(OUT/'token_inputs.jsonl'),
        'tokenizer_signature': replay.tokenizer_signature(llama), 'modernbert_tokenizer_sha256': q.sha(MODEL/'tokenizer.json'),
        'wall_seconds': time.perf_counter()-started, 'existing_QA_modified': False}
    q.save(OUT/'token_input_preparation.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run')); args = p.parse_args()
    {'prepare': prepare, 'run': run}[args.stage]()
