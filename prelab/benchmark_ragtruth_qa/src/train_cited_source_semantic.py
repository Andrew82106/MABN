"""Matched source-agnostic versus cited-source semantic features on original QA."""
from pathlib import Path
import argparse
import pickle
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run_development as q
import run_completed_score_fusion as upstream

OUT = q.ROOT / 'results/cited_source_semantic_lr_v1'
LEXICAL = q.ROOT / 'results/citation_alignment_v1'
SEMANTIC = q.ROOT / 'results/cited_source_semantic_v1'
CS = (.001, .01, .1)
MODES = ('source_agnostic_control', 'cited_source_gap')
SEMANTIC_NAMES = ['valid_ref_claim_fraction', 'any_source_support_mean',
                  'cited_support_gap_mean', 'citation_overlap_x_gap']


def protocol():
    return {
        'version': 'cited-source-semantic-matched-lr-v1',
        'peers': list(upstream.PEERS), 'C': list(CS), 'modes': list(MODES), 'fits': 18,
        'fit': 'Original634 fit answers /168123 unchanged 4BPE windows, original group/class loss weights.',
        'common_inputs': 'Same frozen peer/tail2 two scores, same eight lexical/citation features, valid-reference claim fraction and any-source MiniCheck support from new independent three-source forwards.',
        'additional_inputs': 'Only two: mean(any-source support minus max valid-cited-source support), and the same gap times citation overlap. No gold or raw source identity is a predictor.',
        'matched_control': 'Both modes receive identical new MiniCheck pair outputs and their source-agnostic signal. This separates citation correspondence from merely changing document splitting or adding model calls.',
        'standardization': 'Same fit-only weighted StandardScaler, original base weights; each mode fitted separately.',
        'optimizer': {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 2000, 'seed': 20261010},
        'calibration': 'Same159 answers /42241 windows, original independent F1 thresholds and q.selection_key per family. No cal gradients.',
        'aggregation': 'Predictions remain at original4BPE windows; whole-answer score is max across every eligible original window. No citation-only evaluation.',
        'limits': ['Claim text is unchanged, including reference wording; source-specific support is a learned cue, not proof.',
                   'Multiple cited sources use max individual support; joint multi-source entailment is not evaluated.',
                   'Unrecognized/invalid-only references give no new semantic citation features, not automatic risk labels.',
                   'Underlying models and last-stage inputs are not cross-fitted; calibration has been repeatedly used for development.',
                   'Extra MiniCheck calls make this an offline semantic-plus-generation detector, not a pure generator white-box probe.'],
        'GPU_used_by_training': False, 'official_test_opened': False,
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists()
    q.save(OUT / 'protocol.json', protocol())
    sources = [Path(__file__), Path(q.__file__), Path(upstream.__file__),
               LEXICAL / 'complete.json', upstream.OUT / 'complete.json', q.DATA / 'gold_manifest.json']
    q.save(OUT / 'design_freeze.json', {
        'protocol_sha256': q.sha(OUT / 'protocol.json'),
        'sources_sha256': {str(p.resolve()): q.sha(p) for p in sources},
        'feature_contract': 'window_features.npy float32[210364,4], geometry.json.window_order_sha256, feature_names.json; complete manifest files_sha256',
        'trained': False, 'official_test_opened': False,
    })
    print('CITED_SOURCE_MATCHED_LR_DESIGN_FROZEN_NO_FIT', flush=True)


def load_features(meta):
    for directory, name in ((LEXICAL, 'complete.json'), (SEMANTIC, 'features_complete.json')):
        manifest = q.read(directory / name)
        assert not manifest.get('official_test_opened', False)
        if directory == SEMANTIC:
            assert manifest['status'] == 'complete'
            assert manifest['no_test'] is True and manifest['trained'] is False
            assert q.read(directory / 'feature_names.json') == SEMANTIC_NAMES
        for relative, expected in manifest['files_sha256'].items():
            assert q.sha(directory / relative) == expected, relative
    order = q.digest([w['window_id'] for w in meta['windows']])
    assert q.read(SEMANTIC / 'geometry.json')['window_order_sha256'] == order
    assert q.read(LEXICAL / 'geometry.json')['window_order_sha256'] == order
    lexical = np.load(LEXICAL / 'window_features.npy', allow_pickle=False)
    semantic = np.load(SEMANTIC / 'window_features.npy', allow_pickle=False)
    assert lexical.shape == (210364, 8) and semantic.shape == (210364, 4)
    assert lexical.dtype == semantic.dtype == np.float32
    assert np.isfinite(lexical).all() and np.isfinite(semantic).all()
    assert ((semantic >= 0) & (semantic <= 1)).all()
    return lexical, semantic


def run():
    frozen = q.read(OUT / 'design_freeze.json')
    assert q.read(OUT / 'protocol.json') == protocol()
    assert q.sha(OUT / 'protocol.json') == frozen['protocol_sha256']
    for path, expected in frozen['sources_sha256'].items():
        assert q.sha(Path(path)) == expected, path
    if not (SEMANTIC / 'features_complete.json').exists():
        print('WAIT_COMPLETE_INDEPENDENT_SOURCE_FEATURES_NO_FIT', flush=True)
        return
    assert not (OUT / 'started.json').exists()
    meta = q.metadata()
    lexical, semantic = load_features(meta)
    bw, lw, _, y = q.base_weights(meta)
    assert len(y) == 168123
    cm = q.read(upstream.OUT / 'complete.json')
    assert q.sha(upstream.OUT / 'summary.json') == cm['summary_sha256']
    sources = q.read(upstream.OUT / 'summary.json')
    q.save(OUT / 'started.json', {'time': time.time(),
        'design_freeze_sha256': q.sha(OUT / 'design_freeze.json'),
        'semantic_features_complete_sha256': q.sha(SEMANTIC / 'features_complete.json'),
        'semantic_names': q.read(SEMANTIC / 'feature_names.json'), 'official_test_opened': False})
    candidates, selected = {}, {}
    start = time.perf_counter()
    for peer in upstream.PEERS:
        pair = []
        for alpha in (0., 1.):
            e = next(e for e in sources['all_candidates'][peer] if e['tail_weight'] == alpha)
            path = upstream.OUT / (e['candidate'] + '_scores.npz')
            assert q.sha(path) == e['scores_sha256']
            with np.load(path, allow_pickle=False) as z:
                pair.append(z['window_scores'].copy())
        common = np.column_stack((*pair, lexical, semantic[:, :2]))
        assert common.shape == (210364, 12)
        for mode in MODES:
            x = common if mode == MODES[0] else np.column_stack((common, semantic[:, 2:]))
            assert np.array_equal(x[:, :12], common)
            family = peer + '__' + mode
            scaler = StandardScaler().fit(x[:len(y)], sample_weight=bw)
            scaled = scaler.transform(x)
            scaler_path = OUT / (family + '_scaler.pkl')
            scaler_path.write_bytes(pickle.dumps(scaler, protocol=5))
            entries = []
            for c in CS:
                tick = time.perf_counter()
                model = LogisticRegression(C=c, solver='liblinear', penalty='l2', max_iter=2000, random_state=20261010)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(scaled[:len(y)], y, sample_weight=lw)
                score = model.predict_proba(scaled)[:, 1]
                answer = q.answer_scores(meta, score)
                lo, hi = meta['bounds']['calibration']
                thresholds = {'window': q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], score[lo:hi]),
                              'answer': q.choose_threshold([a['label'] for a in meta['answers'][634:]], answer[634:])}
                metrics = q.metrics(meta, score, thresholds)
                candidate = family + f'__C{c:g}'
                model_path, scores_path = OUT / (candidate + '.pkl'), OUT / (candidate + '_scores.npz')
                model_path.write_bytes(pickle.dumps(model, protocol=5))
                np.savez_compressed(scores_path, window_scores=score, answer_scores=answer)
                entry = {'candidate': candidate, 'peer': peer, 'mode': mode, 'C': c,
                    'thresholds': thresholds, 'metrics': metrics, 'selection_key': list(q.selection_key(thresholds, c)),
                    'input_width': x.shape[1], 'iterations': model.n_iter_.tolist(), 'seconds': time.perf_counter() - tick,
                    'model_sha256': q.sha(model_path), 'scaler_sha256': q.sha(scaler_path), 'scores_sha256': q.sha(scores_path)}
                q.save(OUT / (candidate + '.json'), entry)
                entries.append(entry)
                print('CITED_SOURCE_LR_COMPLETE', candidate, metrics['calibration']['windows']['f1'], metrics['calibration']['answers']['f1'], flush=True)
            candidates[family] = entries
            selected[family] = max(entries, key=lambda e: e['selection_key'])
    assert sum(map(len, candidates.values())) == 18
    q.save(OUT / 'summary.json', {'selected': selected, 'all_candidates': candidates, 'fits_completed': 18,
        'seconds': time.perf_counter() - start, 'official_test_opened': False, 'GPU_used': False})
    report = ['# 被引来源语义分数的匹配对照', '',
        '两模式都使用相同的新三来源推理结果、原8词面特征和相同两分数；区别仅为是否加入被引来源支持度差及其引用位置交互。', '',
        '| 底层对象 | 模式 | C | 窗口F1 | 整答F1 |', '|---|---|---:|---:|---:|']
    for e in selected.values():
        m = e['metrics']['calibration']
        report.append(f"| {e['peer']} | {e['mode']} | {e['C']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    report += ['', '全部793答及原210364个4BPE窗口保留；634fit只训练、159cal选阈值与C。不是只评有引用的句子，不是新人工标签。',
        '引用可能只是提及来源，单源支持也不等于多源联合支持；这些是输入信号，不直接判标签。',
        '这是已有模型训练内分数上的组合，非交叉拟合；反复开发校准成绩不能作为独立测试或SOTA证据。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'summary_sha256': q.sha(OUT / 'summary.json'), 'fits_completed': 18,
                                'official_test_opened': False, 'GPU_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
