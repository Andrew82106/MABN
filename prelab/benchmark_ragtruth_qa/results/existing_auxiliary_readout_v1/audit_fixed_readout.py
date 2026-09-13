"""Independent fixed-epoch score/count audit; no model forward, training or GPU."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
SOURCE = QA/'results/semantic_multitask_v1'
METHOD = 'semantic_tcn_aux_types_w32'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def threshold(y, s):
    y = np.asarray(y, int); s = np.asarray(s, np.float64)
    unique, inverse = np.unique(s, return_inverse=True)
    positive = np.bincount(inverse, weights=y, minlength=len(unique)).astype(np.int64)
    size = np.bincount(inverse, minlength=len(unique))
    tp = positive[::-1].cumsum()[::-1]; n = size[::-1].cumsum()[::-1]
    score = 2*tp/(n+y.sum()); precision = tp/n
    candidates = [(float(score[j]), float(precision[j]), float(unique[j])) for j in range(len(unique))]
    candidates.append((0.,0.,float(np.nextafter(s.max(),np.inf))))
    f, p, t = max(candidates)
    return {'threshold':t,'f1':f,'precision':p,'rows':len(y),'positive':int(y.sum())}


def metric(y, s, t):
    y = np.asarray(y, int); pred = s >= t
    tp = int(((y == 1)&pred).sum()); fp = int(((y == 0)&pred).sum())
    fn = int(((y == 1)&~pred).sum()); tn = int(((y == 0)&~pred).sum())
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'auroc':float(roc_auc_score(y,s)), 'average_precision':float(average_precision_score(y,s))}


def run():
    start = time.perf_counter()
    protocol = read(OUT/'protocol.json'); complete = read(OUT/'complete.json'); summary = read(OUT/'summary.json')
    assert complete['summary_sha256'] == sha(OUT/'summary.json')
    assert protocol['source_sha256'] == sha(QA/'src/run_existing_auxiliary_readout.py')
    old_done = read(SOURCE/'complete.json'); old = read(SOURCE/'summary.json')
    assert old_done['files_sha256']['summary.json'] == sha(SOURCE/'summary.json')
    assert not old_done['test_opened']
    entry = old['selected'][METHOD]
    assert entry['epoch'] == protocol['selected_epoch'] == 3 and entry == summary['control']
    assert len(old['all_epochs'][METHOD]) == 30
    assert entry == max(old['all_epochs'][METHOD],key=lambda x:x['selection_key'])
    path = SOURCE/METHOD/'epoch_003_scores.npz'
    assert sha(path) == entry['scores_sha256'] == protocol['score_sha256'] == summary['auxiliary_token_logits_sha256']
    index_path = QA/'results/sequence_v1/token_index.json'
    assert sha(index_path) == protocol['index_sha256']
    index = read(index_path)['answers']
    answers, tokens, windows = [], [], []
    sources = [path,index_path,OUT/'protocol.json',OUT/'summary.json',OUT/'complete.json',OUT/'scores.npz',
               SOURCE/'summary.json',SOURCE/'complete.json',QA/'src/run_existing_auxiliary_readout.py']
    for part in ('fit','calibration'):
        for dest,name in ((answers,'answers'),(tokens,'tokens'),(windows,'windows_k4')):
            p = QA/f'data/{name}_{part}.jsonl'; dest.extend(lines(p)); sources.append(p)
    assert len(answers) == len(tokens) == len(index) == 793 and len(windows) == 210364
    assert [a['response_id'] for a in answers] == [t['response_id'] for t in tokens] == [i['response_id'] for i in index]
    assert all(a['partition'] == ('fit' if j < 634 else 'calibration') for j,a in enumerate(answers))
    offsets, aw, slots = {}, {a['response_id']:[] for a in answers}, []
    tok = {t['response_id']:t for t in tokens}
    cursor = 0
    for a,i in zip(answers,index):
        assert i['left'] == cursor and i['right']-i['left'] == tok[a['response_id']]['token_count']
        offsets[a['response_id']] = cursor; cursor = i['right']
    assert cursor == 213159
    for j,w in enumerate(windows):
        assert w['eligible'] and w['partition'] == ('fit' if j < 168123 else 'calibration')
        assert w['token_indices'] == list(range(w['token_start'],w['token_end']))
        assert 0 < len(w['token_indices']) <= 4
        local = [i for i in w['token_indices'] if tok[w['response_id']]['lexical_mask'][i]]
        assert local == w['lexical_token_indices'] and local
        assert bool(w['label']) == any(tok[w['response_id']]['risk_mask'][i] for i in local)
        slots.append(np.asarray(local)+offsets[w['response_id']]); aw[w['response_id']].append(j)
    groups = [{a['group_id'] for a in answers if a['partition'] == p} for p in ('fit','calibration')]
    assert len(groups[0]) == 615 and len(groups[1]) == 154 and not groups[0]&groups[1]
    snapshot = {str(p.resolve()):sha(p) for p in sources}
    readout_threads = torch.get_num_threads()
    with np.load(path,allow_pickle=False) as z:
        main = z['token_scores'].copy(); aux = z['auxiliary_token_logits'].copy()
        oldw, olda = z['window_scores'].copy(),z['answer_scores'].copy()
        # Original trainer explicitly set4; the separate fixed readout uses default.
        torch.set_num_threads(4)
        assert np.array_equal(torch.sigmoid(torch.from_numpy(z['token_logits'])).numpy(),main)
        torch.set_num_threads(readout_threads)
    assert aux.shape == (213159,2) and np.isfinite(aux).all()
    # The same CPU primitive/dtype is intentional; no neural model is loaded.
    dual = torch.sigmoid(torch.from_numpy(aux)).numpy()
    new = np.maximum(dual[:,0],dual[:,1])
    assert sha(OUT/'scores.npz') == summary['scores_sha256']
    saved = np.load(OUT/'scores.npz',allow_pickle=False)
    assert np.array_equal(new,saved['token_scores'])
    all_metrics = {}
    for name,pred,target in (('control',main,summary['control']),('aux_max',new,summary['aux_max'])):
        ws = np.asarray([pred[ix].max() for ix in slots],np.float64)
        ass = np.asarray([ws[aw[a['response_id']]].max() for a in answers],np.float64)
        if name == 'control': assert np.array_equal(ws,oldw) and np.array_equal(ass,olda)
        else: assert np.array_equal(ws,saved['window_scores']) and np.array_equal(ass,saved['answer_scores'])
        ts = {'window':threshold([w['label'] for w in windows[168123:]],ws[168123:]),
              'answer':threshold([a['label'] for a in answers[634:]],ass[634:])}
        assert ts == target['thresholds']
        metrics = {}
        for part,wl,wh,al,ah in (('fit',0,168123,0,634),('calibration',168123,210364,634,793)):
            metrics[part] = {'windows':metric([w['label'] for w in windows[wl:wh]],ws[wl:wh],ts['window']['threshold']),
                'answers':metric([a['label'] for a in answers[al:ah]],ass[al:ah],ts['answer']['threshold'])}
        assert metrics == target['metrics']
        all_metrics[name] = metrics
    assert snapshot == {str(p.resolve()):sha(p) for p in sources}
    assert not torch.cuda.is_initialized()
    result = {'status':'passed','selected_epoch':3,'epoch_reselected':False,'source_snapshot':snapshot,
        'token_scores_maxdiff':0.,'window_scores_maxdiff':0.,'answer_scores_maxdiff':0.,
        'old_primary_control_exact':True,'fixed_two_aux_sigmoid_max_exact':True,
        'all_210364_window_geometry_lexical_and_gold_checked':True,
        'independent_threshold_and_metrics_exact':True,'metrics':all_metrics,
        'new_fits':0,'neural_forward':False,'GPU_used':False,'official_test_opened':False,
        'original_primary_sigmoid_CPU_threads':4,'fixed_aux_readout_CPU_threads':readout_threads,
        'audit_context_correction':'Each sigmoid replays its original CPU thread context; see AUDIT_FAILURE_01. Saved source probabilities/thresholds unchanged.',
        'seconds':time.perf_counter()-start}
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'INDEPENDENT_AUDIT.md').write_text(
        '固定旧 epoch3 的两个辅助 logits 经 sigmoid→max，原213159词元、210364窗口、793整答分数均精确重现；旧主头控制也完全一致。独立重算原fit/cal混淆、排序指标、双阈值均通过。没有新轮次选择、模型前向、训练或GPU。\n\n'
        '原主头 cal 窗口/整答 0.640745 / 0.857143，辅助 max 为 0.645510 / 0.861244。该小收益只属于原634fit旧模型的固定读出，不是3680fit双LR训练的成绩，也未超过当前强组合。\n',encoding='utf-8')
    print('EXISTING_AUXILIARY_READOUT_INDEPENDENT_AUDIT_PASSED',flush=True)


if __name__ == '__main__': run()
