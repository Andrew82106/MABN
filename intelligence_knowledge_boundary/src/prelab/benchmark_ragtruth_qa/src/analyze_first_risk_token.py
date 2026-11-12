"""Fit-only positional NLL diagnosis; gold positions are never detector inputs."""
from pathlib import Path
from collections import defaultdict
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import run_development as q

OUT = q.ROOT/'results/first_risk_token_diagnosis_v1'
PAPER = 'https://arxiv.org/html/2507.20836v4'


def main():
    assert not (OUT/'started.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'scope': 'Original634fit only; no calibration body, official test, new extraction or model fitting.',
        'reference': PAPER, 'publication_status': 'Author preprint consulted; main-conference acceptance not established here.',
        'question': 'Is existing reconstructed pre-token NLL higher on the first lexical token of a human risk span than later lexical tokens?',
        'token_groups': 'Per unchanged human span, lexical indices overlapping risk characters. First is its minimum index. For global pooled groups B=unique first indices, I=other risk lexical indices excluding B, O=all non-risk lexical indices.',
        'signal': 'Existing frozen fixed-Llama NF4 NLL only. No entropy cache exists; do not call NLL entropy.',
        'statistics': 'Unsigned original direction higherNLL=higherRisk. Global B-vs-O and I-vs-O AUROC; descriptive sensitivity excluding absolute raw-token0 from both sides. Per original type and all types, paired first minus mean remaining NLL for spans with>=2 lexical tokens; retain single-token counts.',
        'no_selection': 'No C, threshold, epoch, feature selection or new training. Gold-conditioned subgroup AUC is not deployable full-window F1.',
        'limits': 'BPE/lexical first positions and QA-only scope differ from the paper. Span boundaries and vocabulary may explain effects; no causal claim. Original four-BPE evaluation and all labels remain unchanged.'}
    q.save(OUT/'protocol.json', protocol)
    files = [Path(__file__), q.DATA/'tokens_fit.jsonl', q.DATA/'gold_manifest.json']
    q.save(OUT/'started.json', {'time': time.time(), 'sources_sha256': {str(p.resolve()): q.sha(p) for p in files}})
    rows = q.lines(q.DATA/'tokens_fit.jsonl'); assert len(rows) == 634
    pooled = defaultdict(list); no_first = defaultdict(list); by_type = defaultdict(list); spans_out = []; bindings = []
    started = time.perf_counter()
    for row in rows:
        assert row['partition'] == 'fit'
        rid = row['response_id']; path = q.DATA/'features'/(rid+'.npz'); side = q.read(path.with_suffix('.json'))
        assert side['complete'] and side['partition'] == 'fit' and q.sha(path) == side['npz_sha256']
        with np.load(path) as z:
            nll = z['nll'].copy(); offsets = z['response_token_offsets'].copy()
            assert np.array_equal(z['token_ids'], row['token_ids'])
        assert np.array_equal(offsets, row['response_token_offsets']) and len(nll) == row['token_count']
        assert np.isfinite(nll).all() and (nll >= 0).all()
        text = row['original_response']; lex = np.asarray(row['lexical_mask'], bool); risk = np.asarray(row['risk_mask'], bool)
        starts = set()
        for j, label in enumerate(row['original_labels']):
            assert text[label['start']:label['end']] == label['text']
            indices = [i for i, (a, b) in enumerate(offsets) if any(text[k].isalnum()
                for k in range(max(int(a), label['start']), min(int(b), label['end'])))]
            record = {'response_id': rid, 'group_id': row['group_id'], 'span_index': j,
                'type': label['label_type'], 'start': label['start'], 'end': label['end'], 'lexical_token_count': len(indices)}
            if indices:
                assert risk[indices].all(); first = indices[0]; starts.add(first)
                record.update(first_raw_token=first, first_nll=float(nll[first]),
                    continuation_mean_nll=float(nll[indices[1:]].mean()) if len(indices)>1 else None)
                if len(indices)>1:
                    record['paired_difference'] = record['first_nll']-record['continuation_mean_nll']
            spans_out.append(record); by_type[label['label_type']].append(record)
        bmask = np.zeros(len(nll), bool); bmask[list(starts)] = True
        masks = {'first': bmask, 'continuation': risk & lex & ~bmask, 'normal': lex & ~risk}
        for name, mask in masks.items():
            pooled[name].extend(nll[mask].astype(float).tolist())
            mask = mask.copy(); mask[0] = False
            no_first[name].extend(nll[mask].astype(float).tolist())
        bindings.append({'response_id': rid, 'feature_sha256': side['npz_sha256']})
    def auc_report(groups):
        result = {name: {'n': len(values), 'mean_nll': float(np.mean(values)), 'median_nll': float(np.median(values))}
            for name, values in groups.items()}
        for positive in ('first','continuation'):
            p = groups[positive]; n = groups['normal']
            result[positive+'_versus_normal_AUROC'] = float(roc_auc_score([1]*len(p)+[0]*len(n), p+n))
        return result
    def paired(records):
        differences = [r['paired_difference'] for r in records if 'paired_difference' in r]
        return {'spans': len(records), 'paired_spans': len(differences),
            'single_lexical_token_spans': sum(r['lexical_token_count']==1 for r in records),
            'no_lexical_token_spans': sum(r['lexical_token_count']==0 for r in records),
            'mean_difference': float(np.mean(differences)) if differences else None,
            'median_difference': float(np.median(differences)) if differences else None,
            'first_higher_count': sum(d>0 for d in differences)}
    report = {'fit_answers': len(rows), 'human_spans': len(spans_out), 'all_lexical': auc_report(pooled),
        'exclude_absolute_raw_token0_sensitivity': auc_report(no_first),
        'paired_all': paired(spans_out), 'paired_by_original_type': {k: paired(v) for k,v in by_type.items()},
        'seconds': time.perf_counter()-started, 'trained': False, 'GPU_used': False, 'calibration_read': False, 'official_test_read': False}
    q.savel(OUT/'span_records.jsonl', spans_out); q.save(OUT/'feature_bindings.json', bindings); q.save(OUT/'summary.json', report)
    lines = ['# 首个错误词元的训练集诊断', '', f'依据[作者论文]({PAPER})提出的位置差异，只分析原634fit的已有NLL。不是完整论文复现，也不是新增检测器。', '',
        '| 分组 | 词元数 | NLL均值 |', '|---|---:|---:|']
    for name in ('first','continuation','normal'):
        e = report['all_lexical'][name]; lines.append(f"| {name} | {e['n']} | {e['mean_nll']:.6f} |")
    lines += ['', f"首词元对正常词元AUROC：{report['all_lexical']['first_versus_normal_AUROC']:.6f}；其余风险词元对正常词元：{report['all_lexical']['continuation_versus_normal_AUROC']:.6f}。",
        '', '这些分组使用金标位置作诊断，实际检测时不知道这些位置；不能把子集AUROC当作整体F1或宣称已实现定位。原QA标签、窗口和阈值均未改。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    q.save(OUT/'complete.json', {'summary_sha256': q.sha(OUT/'summary.json'), 'trained': False, 'GPU_used': False})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
