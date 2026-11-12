"""Fixed fit-only NLL diagnosis controlling word starts, character and position."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import run_development as q

OUT = q.ROOT / 'results/first_risk_token_matched_v1'
OLD = q.ROOT / 'results/first_risk_token_diagnosis_v1'


def protocol():
    return {'version': 'first-risk-wordstart-matched-v1',
        'scope': 'Only original634fit tokens and fixed existing NLL. No calibration/test, GPU, model fitting, relabeling or deployable feature creation.',
        'first_risk': 'For each original human span, first raw BPE overlapping at least one alphanumeric risk character; deduplicate identical first token indices within an answer, as original diagnosis.',
        'normal': 'Original lexical_mask true and risk_mask false; no model-score filtering.',
        'word_start': 'Find first Python str.isalnum character within original raw token offset. Word start iff its preceding answer character is absent or not isalnum. This is a character-boundary diagnostic, not a language-aware word tokenizer.',
        'character_class': 'First alphanumeric character within raw offset: digit using isdigit, else upper using isupper, else lower using islower, else other. Unicode Python character rules, order fixed.',
        'position': 'Decile=min(9,floor(10*raw_token_index/raw_token_count)); raw positions include punctuation and whitespace. No learned bins.',
        'pooled': ['all first risk versus all normal word-start tokens', 'word-start first risk versus normal word-start tokens', 'word-start continuation risk versus normal word-start tokens'],
        'matching': 'Same answer, same word_start Boolean, same character_class and same position decile. Use ALL eligible normal tokens in that exact bucket. No cross-answer fallback, nearest-neighbor search, or score-based matching.',
        'percentile': '(count(normal NLL < first NLL)+0.5*count(equal))/number of matched normal tokens. HigherNLL direction fixed. Each distinct first risk token equal weight; also report answer-equal mean and word-start-only subset.',
        'missing': 'No matching normal bucket means unmatched, retained with reason and all strata. No imputation or exclusion from reported coverage denominator.',
        'review': 'Original implementation minimum overlapping lexical BPE and unique-first/remaining-risk/normal masks verified; independent rerun must reproduce its pooled counts and AUROCs.',
        'limits': 'Gold-known diagnostic, not deployment F1. Controls are coarse; residual lexical/content, citation, answer and boundary differences remain. Does not establish a causal word-start mechanism or validate new detector features.',
        'bootstrap': 'None', 'trained': False, 'GPU_used': False}


def token_class(text, begin, end, index, count):
    first = next((i for i in range(int(begin), int(end)) if text[i].isalnum()), None)
    if first is None: return None
    c = text[first]
    kind = 'digit' if c.isdigit() else 'upper' if c.isupper() else 'lower' if c.islower() else 'other'
    return {'first_alnum_character_offset': first, 'first_alnum_character': c,
        'word_start': first == 0 or not text[first-1].isalnum(), 'character_class': kind,
        'position_decile': min(9, (10*index)//count)}


def bucket(info):
    return info['word_start'], info['character_class'], info['position_decile']


def percentile(value, normals):
    a = np.asarray(normals, dtype=np.float64)
    assert len(a)
    return float((np.count_nonzero(a < value)+.5*np.count_nonzero(a == value))/len(a))


def prepare():
    assert not (OUT / 'design_freeze.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    # Fixed toy cases: whitespace, continuation BPE, digits, uncased Unicode, ties.
    text = 'Abc 12 中'
    assert token_class(text, 1, 3, 1, 10)['word_start'] is False
    assert token_class(text, 3, 5, 2, 10)['character_class'] == 'digit'
    assert token_class(text, 3, 5, 2, 10)['word_start'] is True
    assert token_class(text, 7, 8, 9, 10)['character_class'] == 'other'
    assert token_class(text, 3, 4, 0, 10) is None
    assert percentile(2., [1., 2., 2., 3.]) == .5
    assert token_class('A', 0, 1, 0, 1)['word_start'] is True
    q.save(OUT / 'protocol.json', protocol())
    bindings = q.read(OLD / 'feature_bindings.json'); assert len(bindings) == 634
    files = [Path(__file__), q.DATA / 'tokens_fit.jsonl', q.DATA / 'gold_manifest.json',
        OLD / 'complete.json', OLD / 'summary.json', OLD / 'span_records.jsonl',
        OLD / 'feature_bindings.json', q.ROOT / 'src/analyze_first_risk_token.py', OUT / 'protocol.json']
    for rec in bindings:
        p = q.DATA / 'features' / (rec['response_id']+'.npz')
        assert q.sha(p) == rec['feature_sha256']
        files.extend((p, p.with_suffix('.json')))
    q.save(OUT / 'CPU_SELFCHECK.json', {'passed': True, 'character_boundaries': True,
        'unicode_case_classes': True, 'raw_position_bins': True, 'half_credit_ties': True,
        'trained': False, 'GPU_used': False})
    q.save(OUT / 'design_freeze.json', {'status': 'fixed_before_analysis',
        'files_sha256': {str(p.resolve()): q.sha(p) for p in files},
        'toy_check_sha256': q.sha(OUT / 'CPU_SELFCHECK.json'), 'calibration_read': False, 'test_read': False})
    print('FIRST_RISK_MATCH_PROTOCOL_FROZEN', flush=True)


def auc(positive, negative):
    if not len(positive) or not len(negative): return None
    return float(roc_auc_score([1]*len(positive)+[0]*len(negative), list(positive)+list(negative)))


def matched_summary(records):
    matched = [r for r in records if r['matched']]
    per_answer = defaultdict(list)
    for r in matched: per_answer[r['response_id']].append(r['percentile'])
    return {'first_risk_tokens': len(records), 'matched': len(matched), 'unmatched': len(records)-len(matched),
        'coverage_fraction': len(matched)/len(records) if records else None,
        'answers_with_first_risk': len({r['response_id'] for r in records}),
        'answers_with_match': len(per_answer),
        'mean_percentile': float(np.mean([r['percentile'] for r in matched])) if matched else None,
        'median_percentile': float(np.median([r['percentile'] for r in matched])) if matched else None,
        'answer_equal_mean_percentile': float(np.mean([np.mean(v) for v in per_answer.values()])) if per_answer else None,
        'total_matched_normal_comparisons': sum(r['normal_count'] for r in matched),
        'above_half_count': sum(r['percentile'] > .5 for r in matched),
        'equal_half_count': sum(r['percentile'] == .5 for r in matched)}


def run():
    assert not (OUT / 'started.json').exists()
    freeze = q.read(OUT / 'design_freeze.json')
    assert q.read(OUT / 'protocol.json') == protocol()
    for p, expected in freeze['files_sha256'].items(): assert q.sha(Path(p)) == expected, p
    assert q.sha(OUT / 'CPU_SELFCHECK.json') == freeze['toy_check_sha256']
    q.save(OUT / 'started.json', {'time': time.time(), 'design_freeze_sha256': q.sha(OUT / 'design_freeze.json')})
    started = time.perf_counter()
    rows = q.lines(q.DATA / 'tokens_fit.jsonl'); assert len(rows) == 634
    pooled = defaultdict(list); stats = Counter(); first_records = []; strata = defaultdict(list)
    original_spans = {(r['response_id'], r['span_index']): r for r in q.lines(OLD / 'span_records.jsonl')}
    used_spans = 0
    for row in rows:
        assert row['partition'] == 'fit'
        rid, text = row['response_id'], row['original_response']
        path = q.DATA / 'features' / (rid+'.npz'); side = q.read(path.with_suffix('.json'))
        assert side['complete'] and side['partition'] == 'fit'
        assert q.sha(path) == side['npz_sha256']
        with np.load(path, allow_pickle=False) as saved:
            nll = saved['nll'].astype(np.float64); offsets = saved['response_token_offsets'].copy()
            assert np.array_equal(saved['token_ids'], row['token_ids'])
        assert np.array_equal(offsets, row['response_token_offsets'])
        assert len(nll) == row['token_count'] and np.isfinite(nll).all() and (nll >= 0).all()
        lex = np.asarray(row['lexical_mask'], bool); risk = np.asarray(row['risk_mask'], bool)
        info = [token_class(text, a, b, i, len(nll)) for i, (a, b) in enumerate(offsets)]
        assert np.array_equal(lex, np.asarray([r is not None for r in info]))
        starts = defaultdict(list)
        for j, label in enumerate(row['original_labels']):
            assert text[label['start']:label['end']] == label['text']
            indices = [i for i, (a,b) in enumerate(offsets) if any(text[k].isalnum()
                for k in range(max(int(a), label['start']), min(int(b), label['end'])))]
            original = original_spans[(rid, j)]; used_spans += 1
            assert original['lexical_token_count'] == len(indices)
            if indices:
                assert risk[indices].all() and indices[0] == original['first_raw_token']
                assert float(nll[indices[0]]) == original['first_nll']
                starts[indices[0]].append(j)
        masks = {'first': np.asarray([i in starts for i in range(len(nll))]),
                 'normal': lex & ~risk}
        masks['continuation'] = lex & risk & ~masks['first']
        word = np.asarray([bool(r and r['word_start']) for r in info])
        normal_buckets = defaultdict(list)
        for i in np.flatnonzero(masks['normal']): normal_buckets[bucket(info[i])].append(int(i))
        for name, mask in masks.items():
            pooled[name].extend(nll[mask].tolist()); pooled[name+'_word_start'].extend(nll[mask & word].tolist())
            for i in np.flatnonzero(mask):
                stats[(name, info[i]['word_start'], info[i]['character_class'])] += 1
        for i, span_indices in sorted(starts.items()):
            controls = normal_buckets[bucket(info[i])]
            record = {'response_id': rid, 'group_id': row['group_id'], 'raw_token_index': i,
                'raw_token_count': len(nll), 'token_start': int(offsets[i,0]), 'token_end': int(offsets[i,1]),
                'token_text': text[int(offsets[i,0]):int(offsets[i,1])], 'span_indices': span_indices,
                **info[i], 'first_nll': float(nll[i]), 'matched': bool(controls), 'normal_count': len(controls),
                'normal_raw_token_indices': controls, 'normal_nll': nll[controls].tolist(),
                'percentile': percentile(nll[i], nll[controls]) if controls else None,
                'unmatched_reason': None if controls else 'No normal in same answer/word-start-state/character-class/position-decile'}
            first_records.append(record)
            strata[(record['word_start'], record['character_class'], record['position_decile'])].append(record)
    old = q.read(OLD / 'summary.json')['all_lexical']
    for name in ('first', 'continuation', 'normal'):
        assert len(pooled[name]) == old[name]['n']
    old_first_auc = auc(pooled['first'], pooled['normal'])
    old_cont_auc = auc(pooled['continuation'], pooled['normal'])
    assert old_first_auc == old['first_versus_normal_AUROC']
    assert old_cont_auc == old['continuation_versus_normal_AUROC']
    assert used_spans == len(original_spans)
    comparisons = {'all_first_vs_normal_word_start': auc(pooled['first'], pooled['normal_word_start']),
        'word_start_first_vs_normal_word_start': auc(pooled['first_word_start'], pooled['normal_word_start']),
        'word_start_continuation_vs_normal_word_start': auc(pooled['continuation_word_start'], pooled['normal_word_start'])}
    summary = {'fit_answers': 634, 'human_spans': used_spans,
        'original_implementation_review': {'passed': True, 'all_span_first_indices_exact': True,
            'original_first_vs_all_normal_AUROC_exact': old_first_auc,
            'original_continuation_vs_all_normal_AUROC_exact': old_cont_auc},
        'pooled_counts': {k: len(v) for k,v in pooled.items()}, 'word_start_AUROC': comparisons,
        'matched_all_first': matched_summary(first_records),
        'matched_word_start_first': matched_summary([r for r in first_records if r['word_start']]),
        'matched_non_word_start_first': matched_summary([r for r in first_records if not r['word_start']]),
        'by_first_character': {kind: matched_summary([r for r in first_records if r['character_class'] == kind])
            for kind in ('digit','upper','lower','other')},
        'group_counts': [{'token_group': k[0], 'word_start': k[1], 'character_class': k[2], 'n': n} for k,n in sorted(stats.items())],
        'matching_strata': [{'word_start': k[0], 'character_class': k[1], 'position_decile': k[2], **matched_summary(v)}
                            for k,v in sorted(strata.items())],
        'seconds': time.perf_counter()-started, 'trained': False, 'GPU_used': False,
        'calibration_read': False, 'official_test_read': False, 'gold_conditioned_diagnostic_not_deployment_F1': True}
    q.savel(OUT / 'matched_first_tokens.jsonl', first_records)
    q.save(OUT / 'summary.json', summary)
    m, ws = summary['matched_all_first'], summary['matched_word_start_first']
    lines = ['# 首风险词元：词首与同回答匹配诊断', '',
        '仅原634fit；原标签、阈值及结果未改。原实现的646处span首BPE索引与两项AUROC已独立精确重现。', '',
        f"原首风险对全部normal AUROC {old_first_auc:.6f}；改用normal词首：{comparisons['all_first_vs_normal_word_start']:.6f}；双方都限词首：{comparisons['word_start_first_vs_normal_word_start']:.6f}。",
        f"原续风险对全部normal {old_cont_auc:.6f}；双方都限词首：{comparisons['word_start_continuation_vs_normal_word_start']:.6f}。", '',
        f"同回答、相同词首状态/首字符类/原BPE位置十分档匹配：{m['matched']}/{m['first_risk_tokens']}处可匹配，{m['unmatched']}处不可匹配。平均NLL百分位{m['mean_percentile']:.6f}，按回答等权为{m['answer_equal_mean_percentile']:.6f}。",
        f"其中词首子集可匹配{ws['matched']}/{ws['first_risk_tokens']}，平均百分位{ws['mean_percentile']:.6f}。", '',
        '百分位0.5表示与匹配normal无平均排序优势；并列计半分。每个首风险使用桶内全部normal，无匹配不回退、不补值；逐项记录包含桶内原BPE索引与NLL。', '',
        '这些位置由人工gold事后确定，是条件诊断，不是部署F1，也不能证明词首造成原差异。仍未完全控制词义、引用、具体内容和标注边界；匹配不足的覆盖也限制解释。没有据此构造最终特征或训练模型。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'status': 'complete_fit_only_diagnosis',
        'design_freeze_sha256': q.sha(OUT / 'design_freeze.json'),
        'files_sha256': {n: q.sha(OUT / n) for n in ('summary.json', 'matched_first_tokens.jsonl', 'REPORT.md')},
        'trained': False, 'GPU_used': False, 'calibration_read': False, 'official_test_read': False})
    print('MATCHED_FIRST_RISK', comparisons, m, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare','run'))
    stage = parser.parse_args().stage
    try: {'prepare': prepare, 'run': run}[stage]()
    except BaseException as exc:
        q.save(OUT / f'FAILURE_{stage}_{time.time_ns()}.json', {'error': repr(exc)})
        raise
