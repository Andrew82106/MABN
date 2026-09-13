"""Independent bounded audit: all sparse maps/gold, six tokenizer replays."""
from pathlib import Path
from collections import Counter
from itertools import zip_longest
import sys
import json
import hashlib
import time
import unicodedata as ud
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = OUT.parent
ROOT = DATA.parent
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q
import feature_qa as replay
import fava_nfc_character_map as nfc
from transformers import AutoTokenizer
from tokenizers import NormalizedString
import torch


def readl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def sha_text(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


def audit():
    assert not torch.cuda.is_initialized()
    started = time.perf_counter()
    report = q.read(OUT / 'token_input_preparation.json')
    assert report['status'] == 'prepared_not_trained' and not report['exceptions']
    assert report['no_truncation'] and not report['trained'] and not report['GPU_used']
    files = {OUT / 'token_inputs.jsonl': report['token_inputs_sha256'],
        OUT / 'nfc_repairs.jsonl': report['nfc_repairs_sha256'],
        OUT / 'token_design_freeze.json': report['token_design_sha256'],
        DATA / 'manifest.json': report['manifest_sha256'], DATA / 'candidate_fit.jsonl': report['candidate_sha256']}
    frozen = q.read(OUT / 'token_design_freeze.json')
    files.update({Path(k): v for k,v in frozen['source_sha256'].items()})
    files[OUT / 'token_protocol.json'] = frozen['protocol_sha256']
    for p, digest in files.items(): assert q.sha(p) == digest, str(p)
    data_audit = q.read(DATA / 'DATA_AUDIT.json')
    assert data_audit['passed'] and data_audit['candidate_manifest_sha256'] == q.sha(DATA / 'manifest.json')
    previous = {r['response_id']: r for r in readl(ROOT / 'auxiliary_fava_v1/token_inputs.jsonl')}
    assert len(previous) == 692
    repair_ledger = {r['response_id']: r for r in readl(OUT / 'nfc_repairs.jsonl')}
    assert len(repair_ledger) == 6
    bert = AutoTokenizer.from_pretrained(ROOT.parent / 'models/ModernBERT-base', local_files_only=True)
    llama = AutoTokenizer.from_pretrained(replay.MODEL, local_files_only=True)
    bert_vocabulary_size, llama_vocabulary_size = len(bert), len(llama)
    normalizer = bert.backend_tokenizer.normalizer
    assert json.loads(normalizer.__getstate__()) == {'type': 'NFC'}
    counters = Counter(); label_types = Counter(); lengths = []; ids = set(); groups = set()
    exact_old = []; repaired = []; max_mass_error = max_weight_error = 0.
    for row, saved in zip_longest(readl(DATA / 'candidate_fit.jsonl'), readl(OUT / 'token_inputs.jsonl')):
        assert row is not None and saved is not None, 'Candidate/token row count mismatch'
        rid = row['response_id']; assert rid == saved['response_id'] and rid not in ids
        ids.add(rid); groups.add(row['group_id'])
        assert row['task_type'] == saved['task_type'] == 'FAVA_synthetic' and row['question'] == ''
        assert saved['synthetic_not_human_gold'] and saved['unmarked_positions_silver_not_verified_negative']
        for key in ('source_id', 'group_id', 'answer_sha256'): assert row[key] == saved[key], (rid,key)
        text = row['original_response']; assert sha_text(text) == row['answer_sha256']
        labels = row['labels']; assert labels and saved['answer_risk'] == 1
        for label in labels:
            assert label['type'] in {'entity', 'relation', 'invented', 'contradictory'}
            assert 0 <= label['start'] < label['end'] <= len(text)
            assert text[label['start']:label['end']] == label['text']
            label_types[label['type']] += 1
        offsets = saved['response_token_offsets']; raw_offsets = saved['response_token_offsets_raw']
        n = len(offsets); m = len(saved['input_ids']); assert n == len(saved['response_token_ids']) == len(raw_offsets)
        assert 0 < m <= 8192 and all(0 <= i < bert_vocabulary_size for i in saved['input_ids'])
        assert all(0 <= i < llama_vocabulary_size for i in saved['response_token_ids'])
        assert all(0 <= a < b <= len(text) for a,b in offsets)
        assert offsets == [[max(0,a),min(len(text),b)] for a,b in raw_offsets]
        covered = set(k for a,b in offsets for k in range(a,b))
        assert all(k in covered for k,c in enumerate(text) if not c.isspace())
        lexical = [int(any(text[k].isalnum() for k in range(a,b))) for a,b in offsets]
        risk = [int(any(text[k].isalnum() and any(l['start'] <= k < l['end'] for l in labels)
                         for k in range(a,b))) for a,b in offsets]
        assert lexical == saved['lexical_mask'] and risk == saved['risk_mask']
        assert any(lexical) and any(risk)
        starts, ends = saved['answer_encoder_start'], saved['answer_encoder_end']
        assert len(starts) == len(ends) == m
        owners = [set() for _ in text]
        for j,(a,b) in enumerate(zip(starts,ends)):
            if a < 0 or b < 0:
                assert a == b == -1; continue
            assert 0 <= a < b <= len(text)
            for k in range(a,b):
                if not text[k].isspace(): owners[k].add(j)
        missing = [k for k,c in enumerate(text) if not c.isspace() and not owners[k]]
        if missing:
            record = repair_ledger[rid]
            assert record['repaired_character_count'] == len(missing)
            assert [p['missing_character_index'] for p in record['repairs']] == missing
            for p in record['repairs']:
                c, a, b = p['missing_character_index'], p['cluster_start'], p['cluster_end']
                cluster = text[a:b]
                assert ud.combining(text[c]) > 0 and ud.category(text[c]).startswith('M')
                assert cluster == p['original_cluster'] and p['starter_index'] == a
                assert a < c < b and ud.combining(text[a]) == 0
                assert all(ud.combining(text[k]) > 0 for k in range(a+1,b))
                assert b == len(text) or ud.combining(text[b]) == 0
                composed = ud.normalize('NFC', cluster)
                assert len(composed) == 1 and composed == p['normalized_character']
                assert ud.normalize('NFD', composed) == ud.normalize('NFD', cluster)
                assert ud.normalize('NFC', text[a:c]+text[c+1:b]) != composed
                ns = NormalizedString(cluster); normalizer.normalize(ns)
                assert ns.normalized == composed and ns[0:1].original == text[a:a+1]
                assert sorted(owners[a]) == p['encoder_token_indices'] and owners[a]
                owners[c] = owners[a].copy()
            # Retokenize only these six, using unedited full references and answer.
            prefix = row['retrieved_passages']+bert.sep_token+bert.sep_token
            enc = bert(prefix+text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
            assert enc['input_ids'] == saved['input_ids']
            eo = np.asarray(enc['offset_mapping'], np.int64)
            inside = (eo[:,1] > len(prefix)) & (eo[:,0] < len(prefix)+len(text))
            a = np.where(inside, np.maximum(0, eo[:,0]-len(prefix)), -1)
            b = np.where(inside, np.minimum(len(text), eo[:,1]-len(prefix)), -1)
            assert a.tolist() == starts and b.tolist() == ends
            view = replay.encode_view(llama, row['released_prompt'], text, row['evidence_original_prompt_range'])
            for key in ('response_token_offsets', 'response_token_offsets_raw'): assert view[key] == saved[key]
            assert view['answer_token_ids'] == saved['response_token_ids']
            new_map, proof = nfc.character_map(text, offsets, starts, ends, normalizer=normalizer, return_diagnostics=True)
            assert dict(response_id=rid, **proof) == record
            assert all(np.array_equal(x,y) for x,y in zip(new_map,saved['mapping']))
            repaired.append({'response_id': rid, 'encoder_input_tokens': m, 'repaired_characters': len(missing),
                'evidence_sha256': sha_text(row['retrieved_passages']), 'answer_sha256': row['answer_sha256'],
                'labels_sha256': q.digest(labels), 'all_token_IDs_offsets_mapping_and_proof_replayed_exact': True,
                'independent_NFC_equivalence_and_original_starter_ownership_verified': True})
        else: assert rid not in repair_ledger
        assert all(owners[k] for k,c in enumerate(text) if not c.isspace())
        rr,cc,ww = map(np.asarray, saved['mapping'])
        assert len(rr) == len(cc) == len(ww) and np.isfinite(ww).all() and (ww > 0).all()
        assert (rr >= 0).all() and (rr < n).all() and (cc >= 0).all() and (cc < m).all()
        assert len(set(zip(rr,cc))) == len(rr)
        totals = np.bincount(rr.astype(int), weights=ww, minlength=n)
        nonspace = np.asarray([any(not c.isspace() for c in text[a:b]) for a,b in offsets])
        err = float(np.max(np.abs(totals[nonspace]-1),initial=0))
        max_mass_error = max(max_mass_error,err)
        assert err < 2e-7 and not totals[~nonspace].any()
        # Independent ownership-to-weight construction for every row, without
        # calling either original character_map or the new helper here.
        actual = {(int(i),int(j)):float(w) for i,j,w in zip(rr,cc,ww)}
        expected = {}
        for i,(a,b) in enumerate(offsets):
            chars = [k for k in range(a,b) if not text[k].isspace()]
            if not chars: continue
            cols = set().union(*(owners[k] for k in chars))
            for j in cols:
                expected[(i,j)] = sum(1/len(owners[k]) for k in chars if j in owners[k])/len(chars)
        assert actual.keys() == expected.keys(), rid
        weight_err = max((abs(actual[k]-np.float32(v)) for k,v in expected.items()),default=0.)
        max_weight_error = max(max_weight_error,float(weight_err))
        assert weight_err < 1e-7, (rid,weight_err)
        if rid in previous:
            assert saved == previous[rid], ('v1_previous_record_changed',rid)
            exact_old.append(rid)
        wc = wp = 0
        for i in range(max(1,n-3)):
            stop = min(i+4,n)
            if any(lexical[i:stop]): wc += 1; wp += int(any(risk[i:stop]))
        counters.update(answers=1, encoder_input_tokens=m, raw_answer_tokens=n,
            lexical_answer_tokens=sum(lexical), risk_answer_tokens=sum(risk), windows=wc, risk_windows=wp)
        lengths.append(m)
        if len(ids)%1500 == 0: print('FAVA_AUDIT_ROWS',len(ids),flush=True)
    counters.update(max_input_tokens=max(lengths), nfc_repaired_answers=len(repaired),
        nfc_repaired_characters=sum(r['repaired_characters'] for r in repaired))
    assert counters == Counter(report['stats'])
    assert len(ids) == report['answers'] == report['expected_answers'] == 7482
    assert len(exact_old) == len(previous) == 692
    assert {r['response_id'] for r in repaired} == repair_ledger.keys()
    assert sum(r['repaired_characters'] for r in repaired) == 8
    assert dict(label_types) == data_audit['span_types'] and len(groups) == data_audit['groups']
    assert float(np.median(lengths)) == report['median_input_tokens']
    for p,digest in files.items(): assert q.sha(p) == digest
    assert not torch.cuda.is_initialized()
    result = {'passed': True, 'status': 'passed_ready_for_synthetic_aux_training_preparation',
        'counts_independently_recomputed': dict(counters), 'groups': len(groups), 'span_types': dict(label_types),
        'previous_v1_records_exact': len(exact_old), 'previous_v1_response_ids': exact_old,
        'all_candidate_IDs_order_and_original_labels_match': True,
        'all_raw_offsets_encoder_column_ranges_and_character_coverage_verified': True,
        'all_sparse_weights_independently_reconstructed': True,
        'maximum_raw_nonspace_mass_error': max_mass_error, 'maximum_sparse_weight_difference': max_weight_error,
        'six_original_input_tokenizer_replays': repaired,
        'audit_seconds': time.perf_counter()-started, 'GPU_used': False, 'new_model_or_training': False,
        'calibration_or_test_read': False, 'frozen_artifacts_changed': False,
        'material_isolation_rebuilt': False, 'existing_DATA_AUDIT_sha256': q.sha(DATA/'DATA_AUDIT.json'),
        'token_preparation_sha256': q.sha(OUT/'token_input_preparation.json'),
        'token_inputs_sha256': report['token_inputs_sha256'], 'audit_code_sha256': q.sha(Path(__file__)),
        'limits': 'Released FAVA synthetic factual annotations, not human gold. This audit checks unchanged provenance/mapping; it does not establish factual label correctness or model performance.'}
    q.save(OUT / 'AUDIT.json', result)
    text = ('# FAVA tokenization v2独立审计\n\n'
        '通过，可交合成辅助训练准备；本次未训练、未用GPU。\n\n'
        f'- 全7482答、{len(groups)}材料组；原候选ID/顺序、标签、坐标及映射覆盖均一致。\n'
        '- v1已完成的692条同ID记录完整字段exact。\n'
        '- 6答8个NFC组合符已用原完整资料和回答重放：两种分词ID、offset、稀疏map及proof均exact。\n'
        f'- 独立复算：969140原BPE、854011 lexical、215922风险BPE；943938可评4BPE窗口、278300风险窗口。最大输入1481，无截断。\n'
        f'- 全量稀疏权重独立重建通过；原token质量最大误差{max_mass_error:.3g}。\n\n'
        '保留FAVA合成标签身份；审核的是原标签和映射一致性，不是新增人工事实裁决。原数据隔离沿用已通过的DATA_AUDIT，未重建或修改。\n')
    (OUT / 'AUDIT_REPORT.md').write_text(text,encoding='utf-8')
    print('FAVA_TOKEN_AUDIT_PASSED',dict(counters),'old_exact',len(exact_old),flush=True)


if __name__ == '__main__':
    try: audit()
    except BaseException as exc:
        q.save(OUT / f'AUDIT_FAILURE_{time.time_ns()}.json', {'error':repr(exc),'frozen_artifacts_changed':False})
        raise
