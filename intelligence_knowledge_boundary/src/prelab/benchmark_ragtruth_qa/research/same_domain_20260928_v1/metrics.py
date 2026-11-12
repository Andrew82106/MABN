"""Common evaluation for all methods in the 2026-09-28 in-domain run."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def readl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s]


def metrics_at(labels, scores, threshold):
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    assert y.shape == p.shape and len(y) and np.isfinite(p).all()
    prediction = p >= threshold
    tp, fp = int((prediction & y).sum()), int((prediction & ~y).sum())
    fn, tn = int((~prediction & y).sum()), int((~prediction & ~y).sum())
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'auroc': float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            'average_precision': float(average_precision_score(y, p)) if y.any() else None,
            'threshold': float(threshold)}


def select_threshold(labels, scores):
    y = np.asarray(labels, bool)
    p = np.asarray(scores, np.float64)
    assert len(y) and y.shape == p.shape and np.isfinite(p).all()
    order = np.argsort(-p, kind='stable')
    ranked = p[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    tp = np.cumsum(y[order])[ends]
    predicted = ends + 1
    f1 = 2 * tp / np.maximum(predicted + y.sum(), 1)
    precision = tp / predicted
    candidates = list(zip(f1.tolist(), precision.tolist(), ranked[ends].tolist()))
    candidates.append((0., 0., float(np.nextafter(p.max(), np.inf))))
    return float(max(candidates)[2])


def pool_windows(token_scores, lexical_mask, pooling='max'):
    p = np.asarray(token_scores, np.float64)
    lexical = np.asarray(lexical_mask, bool)
    assert p.shape == lexical.shape and np.isfinite(p).all()
    intervals = [(0, len(p))] if len(p) < 4 else [(i, i + 4) for i in range(len(p) - 3)]
    result = []
    for a, b in intervals:
        if not lexical[a:b].any():
            continue
        # Native mean adapters preserve their full native interval; our max
        # detector excludes punctuation-only token scores from the maximum.
        if pooling == 'max':
            value = p[a:b][lexical[a:b]].max()
        elif pooling == 'mean':
            value = p[a:b].mean()
        else:
            raise ValueError(pooling)
        result.append(float(value))
    return np.asarray(result, np.float64)


def evaluate_partition(partition, scores_by_response, pooling='max', thresholds=None,
                       *, scores_are_windows=False, allow_test=False):
    """Tune only on validation no-hint, then apply to both reported cohorts.

    Each score array is aligned to original Llama BPE or precomputed eligible
    four-BPE windows. Callers must supply every response, without filtering gold.
    Test evaluation additionally requires explicit release by the orchestrator.
    """
    if partition == 'test':
        if not allow_test or thresholds is None or not (HERE / 'TEST_RELEASE.json').exists():
            raise RuntimeError('Test labels are sealed until predictions are frozen.')
        labels = {x['response_id']: x for x in readl(DATA / 'test/sealed/labels.jsonl')}
        members = [x for x in readl(DATA / 'split_assignments.jsonl') if x['split'] == 'test']
        answers = [{'response_id': x['response_id'], 'group_id': x['group_id'],
                    'unanswerability_hint_setting': x['unanswerability_hint_setting'],
                    'answer_risk': labels[x['response_id']]['answer_label']} for x in members]
        sequences = {x['response_id']: dict(x, risk_mask=labels[x['response_id']]['risk_mask'])
                     for x in readl(DATA / 'test/geometry.jsonl')}
    elif partition in ('fit', 'validation'):
        label_root = DATA / partition
        answers = readl(label_root / 'answers.jsonl')
        sequences = {x['response_id']: x for x in readl(label_root / 'sequences.jsonl')}
    else:
        raise ValueError(partition)
    assert set(scores_by_response) == {x['response_id'] for x in answers}, 'Missing or extra responses'
    rows = []
    for answer in answers:
        rid = answer['response_id']
        sequence = sequences[rid]
        lexical = np.asarray(sequence['lexical_mask'], bool)
        risk = np.asarray(sequence['risk_mask'], bool)
        assert lexical.shape == risk.shape
        intervals = ([(0, len(risk))] if len(risk) < 4 else
                     [(i, i + 4) for i in range(len(risk) - 3)])
        window_labels = np.asarray([bool((risk[a:b] & lexical[a:b]).any())
                             for a, b in intervals if lexical[a:b].any()], bool)
        if partition == 'test':
            published = [x for x in labels[rid]['window_labels'] if x is not None]
            assert np.array_equal(window_labels, published), rid
        scores = (np.asarray(scores_by_response[rid], np.float64) if scores_are_windows else
                  pool_windows(scores_by_response[rid], lexical, pooling))
        assert scores.shape == window_labels.shape and len(scores) and np.isfinite(scores).all(), rid
        rows.append({'response_id': rid, 'group_id': answer['group_id'],
                     'no_hint': not answer['unanswerability_hint_setting'],
                     'answer_label': bool(answer['answer_risk']),
                     'answer_score': float(scores.max()), 'window_labels': window_labels,
                     'window_scores': scores})
    if thresholds is None:
        if partition != 'validation':
            raise ValueError('Threshold selection is validation-only.')
        selected = [r for r in rows if r['no_hint']]
        assert selected
        thresholds = {
            'answer': select_threshold([r['answer_label'] for r in selected],
                                       [r['answer_score'] for r in selected]),
            'window': select_threshold(np.concatenate([r['window_labels'] for r in selected]),
                                       np.concatenate([r['window_scores'] for r in selected]))}
    cohorts = {}
    for cohort in ['all', 'no_hint']:
        selected = [r for r in rows if cohort == 'all' or r['no_hint']]
        assert selected
        cohorts[cohort] = {
            'answer': metrics_at([r['answer_label'] for r in selected],
                                 [r['answer_score'] for r in selected], thresholds['answer']),
            'window': metrics_at(np.concatenate([r['window_labels'] for r in selected]),
                                 np.concatenate([r['window_scores'] for r in selected]), thresholds['window']),
            'groups': len({r['group_id'] for r in selected})}
    w, a = cohorts['no_hint']['window'], cohorts['no_hint']['answer']
    return {'partition': partition, 'thresholds': thresholds, 'cohorts': cohorts,
            'selection_key': [min(w['f1'], a['f1']), w['f1'], w['precision']],
            'primary_cohort': 'no_hint', 'pooling': pooling,
            'annotation': 'released automatic span annotations; not human gold'}
