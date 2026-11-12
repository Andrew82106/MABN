"""Independent CPU audit of MiniCheck sentence coordinates and score adapter."""
from pathlib import Path
from collections import defaultdict
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
PROJECT = next(p for p in RUN.parents if p.name == 'intelligence_knowledge_boundary')
sys.path.insert(0, str(OUT))
from evaluate_extensions import sha, load, save, indexed


def main():
    sem = OUT.parent / 'semantic'
    prepared = sem / 'minicheck_run'
    receipt = load(prepared / 'PREPARED.json')
    for path, expected in receipt['source_input_sha256'].items():
        assert sha(path) == expected
    import nltk
    from transformers import AutoTokenizer
    nltk.data.path.insert(0, str(PROJECT / 'artifacts/dependencies/benchmark_ragtruth_qa/semantic_baseline/nltk_data'))
    punkt = nltk.tokenize.PunktTokenizer('english')
    tokenizer = AutoTokenizer.from_pretrained(sem / 'resources/MiniCheck-Flan-T5-Large', local_files_only=True)
    total = sentence_count = whitespace_tokens = max_pair_length = 0
    split_checks, score_checks = {}, {}
    for part in ['validation', 'test']:
        records = indexed(prepared / f'{part}_prepared.jsonl')
        inputs = indexed(RUN / 'data' / part / 'inputs.jsonl')
        geometry = indexed(RUN / 'data' / part / 'geometry.jsonl')
        assert records.keys() == inputs.keys() == geometry.keys()
        for rid, r in records.items():
            inp, geo = inputs[rid], geometry[rid]
            text = inp['original_response']
            assert r['documents'] == inp['retrieved_passages']
            assert r['question_not_used_by_native_method'] == inp['question']
            assert 'labels' not in r and 'risk_mask' not in r
            expected, position = [], 0
            for line in text.split('\n'):
                for a, b in punkt.span_tokenize(line):
                    if line[a:b].strip():
                        expected.append({'start': position + a, 'end': position + b, 'text': line[a:b]})
                position += len(line) + 1
            assert expected == r['sentences']
            # Independently reproduce upstream 500-whitespace-word chunks,
            # including its explicit newline sentence separators.
            doc_sentences = []
            for line in r['documents'].split('\n'):
                doc_sentences.extend(punkt.tokenize(line))
                doc_sentences.append('\n')
            doc_sentences = doc_sentences[:-1] or ['']
            chunk_list, pending, words = [], [], 0
            for sentence in doc_sentences:
                size = len(sentence.split())
                if words + size > 500:
                    chunk_list.append(' '.join(pending))
                    pending, words = [sentence], size
                else:
                    pending.append(sentence)
                    words += size
            if pending:
                chunk_list.append(' '.join(pending))
            chunks = [x.replace(' \n ', '\n').strip() for x in chunk_list]
            chunks = [x for x in chunks if x]
            assert chunks == r['source_chunks']
            for s in r['sentences']:
                assert text[s['start']:s['end']] == s['text']
                for doc in chunks:
                    length = len(tokenizer('predict: ' + doc + tokenizer.eos_token + s['text'], truncation=False)['input_ids'])
                    assert length <= 2048
                    max_pair_length = max(max_pair_length, length)
            ii, jj, weights = r['mapping']
            n = len(geo['lexical_mask'])
            assert n == r['original_token_count']
            assert len(ii) == len(jj) == len(weights)
            assert all(0 <= i < n and 0 <= j < len(r['sentences']) and w > 0 for i, j, w in zip(ii, jj, weights))
            assert np.allclose(np.bincount(ii, weights=weights, minlength=n), 1., rtol=0, atol=1e-12)
            actual = defaultdict(dict)
            for i, j, w in zip(ii, jj, weights):
                assert j not in actual[i]
                actual[i][j] = w
            owners = [[] for _ in text]
            for j, sentence in enumerate(r['sentences']):
                for k in range(sentence['start'], sentence['end']):
                    owners[k].append(j)
            assert all(owners[k] for k, char in enumerate(text) if not char.isspace())
            for i, (a, b) in enumerate(geo['response_token_offsets']):
                chars = [k for k in range(a, b) if not text[k].isspace()]
                if not chars:
                    whitespace_tokens += 1
                    continue
                expected_weights = defaultdict(float)
                for k in chars:
                    for j in owners[k]:
                        expected_weights[j] += 1 / (len(chars) * len(owners[k]))
                assert set(expected_weights) == set(actual[i])
                assert all(abs(w - actual[i][j]) <= 1e-12 for j, w in expected_weights.items())
            total += 1
            sentence_count += len(r['sentences'])
        split_checks[part] = {'answers': len(records), 'prepared_sha256': sha(prepared / f'{part}_prepared.jsonl')}
        prediction_path = prepared / f'{part}_predictions.jsonl'
        if prediction_path.exists():
            predictions = indexed(prediction_path)
            assert predictions.keys() == records.keys()
            for rid, pred in predictions.items():
                r = records[rid]
                assert pred['response_sha256'] == r['response_sha256']
                native = pred['native_sentence_scores']
                assert len(native) == len(r['sentences'])
                risks = []
                for s, original in zip(native, r['sentences']):
                    assert all(s[k] == original[k] for k in ['start', 'end', 'text'])
                    probabilities = s['support_probability_per_chunk']
                    assert len(probabilities) == len(r['source_chunks'])
                    assert all(0 <= p <= 1 for p in probabilities)
                    assert s['used_chunks'] == r['source_chunks']
                    assert s['support_probability'] == max(probabilities)
                    assert s['risk_probability'] == 1 - s['support_probability']
                    risks.append(s['risk_probability'])
                ii, jj, ww = r['mapping']
                recomputed = np.bincount(ii, weights=np.asarray(risks)[jj] * ww, minlength=r['original_token_count'])
                assert np.allclose(recomputed, pred['token_scores'], rtol=0, atol=1e-12)
            score_checks[part] = {'answers': len(predictions), 'prediction_sha256': sha(prediction_path),
                'all_chunk_maxima_score_direction_and_sentence_to_token_projections_recomputed': True}
    assert total == 550 and sentence_count == 2482
    report = {'status': 'PASS', 'answers': total, 'native_sentences': sentence_count,
        'max_native_doc_claim_pair_tokens': max_pair_length, 'truncated_pairs': 0,
        'all_source_chunks_independently_reconstructed': True,
        'all_nonwhitespace_characters_covered': True, 'all_original_tokens_mapping_mass_one': True,
        'nonwhitespace_mapping_independently_reconstructed': True,
        'whitespace_tokens_with_nonzero_native_mass': whitespace_tokens,
        'input': 'retrieved_passages + original sentence; question not used as evidence/claim',
        'native_granularity': 'sentence; common BPE mapping is not native token detection',
        'splits': split_checks, 'completed_prediction_checks': score_checks,
        'prepared_receipt_sha256': sha(prepared / 'PREPARED.json'),
        'code_sha256': {str(sem / n): sha(sem / n) for n in ['minicheck_official.py', 'common.py']},
        'GPU_used': False, 'stage': 'prepared inputs/mapping plus available actual native-score projections'}
    save(OUT / 'MINICHECK_INPUT_AUDIT.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
