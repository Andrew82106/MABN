"""CPU-only gate tests; no full-cohort fit or real feature preparation."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/'src'))
import run_ghost_matched_lr_v1 as run


def main():
    assert not torch.cuda.is_initialized()
    assert not (run.GHOST/'features_complete.json').exists()
    assert not (OUT/'fit_started.json').exists()
    assert not (OUT/'feature_preparation_started.json').exists()
    checks = []
    with patch.object(run.LogisticRegression, 'fit') as fitting, patch.object(run.np.lib.format, 'open_memmap') as matrix:
        for func in (run.prepare_features, run.fit):
            try:
                func()
            except RuntimeError as exc:
                assert str(exc).startswith('Complete3839 GHOST features missing;')
                checks.append(func.__name__ + ': missing completion rejected')
            else:
                raise AssertionError('Missing completion was accepted')
        with tempfile.TemporaryDirectory(prefix='partial_cohort_check_', dir=OUT) as folder:
            fake = Path(folder)
            run.q.save(fake/'features_complete.json', {'status':'complete','records':3838,'raw_answer_tokens':708505})
            with patch.object(run, 'GHOST', fake):
                try:
                    run.prepare_features()
                except AssertionError:
                    checks.append('3838 partial cohort rejected')
                else:
                    raise AssertionError('Partial cohort was accepted')
        fitting.assert_not_called()
        matrix.assert_not_called()
    assert not (OUT/'fit_started.json').exists()
    assert not (OUT/'feature_preparation_started.json').exists()
    assert not torch.cuda.is_initialized()
    report = {'status':'passed','checks':checks,'real_new_fits':0,'matrix_creation_calls':0,
              'GPU_used':False,'test_opened':False,
              'source_sha256':{str(Path(run.__file__).resolve()):run.q.sha(run.__file__),
                               str(Path(__file__).resolve()):run.q.sha(__file__)}}
    run.freeze(OUT/'MISSING_FEATURE_GATE_CHECK.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
