"""Independent arithmetic audit of the fixed36 FAVA convex candidates."""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
UPSTREAM = QA/'results/full_context_fava_transfer_v1/transfer'
LR = QA/'results/citation_alignment_lr_v1'
TREE = QA/'results/completed_score_combiner_v1'
ALPHA = (0.,.2,.4,.6,.8,1.)
PEERS = ('lookback','harp_claim','semantic_claim')
spec = importlib.util.spec_from_file_location('_independent_count_primitives',
    QA/'results/existing_auxiliary_readout_v1/audit_fixed_readout.py')
ind = importlib.util.module_from_spec(spec); spec.loader.exec_module(ind)
read, lines, sha = ind.read, ind.lines, ind.sha


def key(ts, alpha):
    return [min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-alpha]


def run():
    start = time.perf_counter()
    done = read(OUT/'complete.json'); summary = read(OUT/'summary.json'); prep = read(OUT/'preparation_complete.json')
    assert done['target'] == summary['target'] == 'fava' and done['candidate_count'] == 36
    assert done['summary_sha256'] == sha(OUT/'summary.json') and done['new_fits'] == 0
    for path,digest in prep['source_sha256'].items(): assert sha(Path(path)) == digest
    assert prep['protocol_sha256'] == sha(OUT/'protocol.json')
    assert prep['CPU_check_sha256'] == sha(OUT/'CPU_SELFCHECK.json')
    upstream = read(UPSTREAM/'complete.json'); started = read(OUT/'started.json')
    assert [e['epoch'] for e in upstream['all_QA_epochs']] == [1,2,3]
    target = upstream['selected']
    assert target['epoch'] == started['selected_epoch'] and target['epoch'] in (1,2,3)
    assert target == max(upstream['all_QA_epochs'],key=lambda e:e['selection_key'])
    assert target == summary['target_selected']
    assert started['upstream_complete_sha256'] == sha(UPSTREAM/'complete.json')
    path = UPSTREAM/f"qa_epoch_{target['epoch']:02d}_token_predictions.npz"
    assert sha(path) == started['upstream_token_predictions_sha256'] == target['artifacts_sha256']['_token_predictions.npz']
    assert not upstream['official_test_opened'] and not done['official_test_opened']
    answers, tokens, windows = [],[],[]
    gold_paths = []
    for part in ('fit','calibration'):
        for dest,name in ((answers,'answers'),(tokens,'tokens'),(windows,'windows_k4')):
            p = QA/f'data/{name}_{part}.jsonl'; gold_paths.append(p);dest.extend(lines(p))
    assert len(answers) == len(tokens) == 793 and len(windows) == 210364
    assert [a['response_id'] for a in answers] == [t['response_id'] for t in tokens]
    ai = {a['response_id']:i for i,a in enumerate(answers)}
    tb = {t['response_id']:t for t in tokens}
    wind_to_answer = np.asarray([ai[w['response_id']] for w in windows],int)
    with np.load(path,allow_pickle=False) as z:
        assert len(z.files) == 3839
        probs = {rid:z[rid].copy() for rid in ai}
    for rid,p in probs.items():
        assert p.shape == (tb[rid]['token_count'],) and np.isfinite(p).all()
    detector = []
    for j,w in enumerate(windows):
        assert w['partition'] == ('fit' if j < 168123 else 'calibration') and w['eligible']
        assert w['token_indices'] == list(range(w['token_start'],w['token_end']))
        assert 0 < len(w['token_indices']) <= 4
        lex = [k for k in w['token_indices'] if tb[w['response_id']]['lexical_mask'][k]]
        assert lex == w['lexical_token_indices'] and lex
        detector.append(float(probs[w['response_id']][lex].max()))
    detector = np.asarray(detector,np.float64)
    assert np.array_equal(detector,np.load(OUT/'detector_window_probability.npy'))
    assert len({a['group_id'] for a in answers[:634]}) == 615 and len({a['group_id'] for a in answers[634:]}) == 154
    assert not {a['group_id'] for a in answers[:634]} & {a['group_id'] for a in answers[634:]}
    wy = np.asarray([w['label'] for w in windows],int); ay = np.asarray([a['label'] for a in answers],int)
    assert wy[168123:].sum() == 5984 and ay[634:].sum() == 100
    def answermax(score):
        result = np.full(len(answers),-np.inf)
        np.maximum.at(result,wind_to_answer,score)
        assert np.isfinite(result).all()
        return result
    def thresholds(score,ans):
        return {'window':ind.threshold(wy[168123:],score[168123:]),'answer':ind.threshold(ay[634:],ans[634:])}
    def metrics(score,ans,ts):
        return {part:{'windows':ind.metric(wy[wl:wh],score[wl:wh],ts['window']['threshold']),
            'answers':ind.metric(ay[al:ah],ans[al:ah],ts['answer']['threshold'])}
            for part,wl,wh,al,ah in (('fit',0,168123,0,634),('calibration',168123,210364,634,793))}
    target_ans = answermax(detector)
    assert thresholds(detector,target_ans) == target['thresholds']
    assert metrics(detector,target_ans,target['thresholds'])['calibration'] == target['calibration']
    lr,tree = read(LR/'summary.json'),read(TREE/'summary.json')
    controls=[]
    for peer in PEERS:
        e = lr['selected'][peer+'__two_scores_and_citation']
        controls.append((peer+'__old_lr',e,LR/(e['candidate']+'_scores.npz')))
        e = tree['methods'][peer]
        controls.append((peer+'__old_tree',e,TREE/(peer+'_scores.npz')))
    assert set(summary['all_candidates']) == {name for name,_,_ in controls}
    snapshot = {str(p.resolve()):sha(p) for p in [*gold_paths,path,OUT/'complete.json',OUT/'summary.json',
        OUT/'preparation_complete.json',UPSTREAM/'complete.json',LR/'summary.json',TREE/'summary.json']}
    selected,checks,endpoints = {},[],[]
    for family,control,cpath in controls:
        assert control == summary['controls'][family] and sha(cpath) == control['scores_sha256']
        assert str(cpath.resolve()) in prep['source_sha256']
        with np.load(cpath,allow_pickle=False) as z:
            baseline = z['window_scores'].copy(); ba = z['answer_scores'].copy()
        assert np.array_equal(answermax(baseline),ba)
        assert metrics(baseline,ba,control['thresholds']) == control['metrics']
        entries = summary['all_candidates'][family]
        assert [e['target_weight'] for e in entries] == list(ALPHA)
        for alpha,e in zip(ALPHA,entries):
            assert e['target_epoch'] == target['epoch'] and e['target'] == 'fava'
            assert e == read(OUT/(e['candidate']+'.json'))
            cscore = OUT/(e['candidate']+'_scores.npz')
            assert sha(cscore) == e['scores_sha256']
            score = (1-alpha)*baseline + alpha*detector
            ans = answermax(score)
            with np.load(cscore,allow_pickle=False) as z:
                assert np.array_equal(score,z['window_scores'])
                assert np.array_equal(ans,z['answer_scores'])
            ts = thresholds(score,ans); mm = metrics(score,ans,ts)
            assert ts == e['thresholds'] and mm == e['metrics'] and key(ts,alpha) == e['selection_key']
            if alpha == 0:
                assert np.array_equal(score,baseline) and np.array_equal(ans,ba)
                assert ts == control['thresholds'] and mm == control['metrics']
                endpoints.append({'family':family,'alpha':0,'exact':True})
            if alpha == 1:
                assert np.array_equal(score,detector) and np.array_equal(ans,target_ans)
                assert ts == target['thresholds'] and mm['calibration'] == target['calibration']
                endpoints.append({'family':family,'alpha':1,'exact':True})
            checks.append({'candidate':e['candidate'],'scores_maxdiff':0.,'answermax_exact':True,
                'independent_threshold_and_fit_cal_metrics_exact':True})
        best = max(entries,key=lambda e:key(e['thresholds'],e['target_weight']))
        assert best == summary['selected'][family]
        selected[family] = {'candidate':best['candidate'],'alpha':best['target_weight'],
            'window_f1':best['metrics']['calibration']['windows']['f1'],
            'answer_f1':best['metrics']['calibration']['answers']['f1']}
    assert len(checks) == 36 and len(endpoints) == 12 and endpoints == summary['endpoint_checks']
    assert snapshot == {str(Path(p).resolve()):sha(Path(p)) for p in snapshot}
    report = {'status':'passed','candidate_count':36,'endpoint_count':12,'frozen_upstream_epoch':target['epoch'],
        'selected':selected,'checks':checks,'endpoints':endpoints,'source_sha256':snapshot,
        'native_fit_answers':634,'calibration_answers':159,'windows':210364,'calibration_windows':42241,
        'no_large_combination_used_as_base':True,'no_new_epoch_selection':True,'new_fits':0,
        'GPU_used':False,'official_test_opened':False,'seconds':time.perf_counter()-start}
    (OUT/'ARITHMETIC_REPLAY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=['# FAVA 固定36组合算术复算','',
        '通过：统一使用完整辅助训练及QA三轮后既有选中的QA轮次。全部36个概率凸组合、12个端点、原210364窗口与793答max、独立双阈值/fit-cal计数和六家选择均精确一致。没有使用large组合当新底座，没有训练或GPU。','',
        '| 原底座 | FAVA权重 | 窗口F1 | 整答F1 |','|---|---:|---:|---:|']
    for family,e in selected.items():text.append(f"| {family} | {e['alpha']:g} | {e['window_f1']:.6f} | {e['answer_f1']:.6f} |")
    text += ['', 'FAVA为自动合成辅助数据迁移；所有候选在反复使用的开发校准集比较，不新增独立测试或显著性声明。最优候选变化另记入总报告。']
    (OUT/'ARITHMETIC_REPLAY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('FAVA_FIXED36_ARITHMETIC_REPLAY_PASSED',flush=True)


if __name__ == '__main__': run()
