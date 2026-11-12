"""One-shot continuation: finish every model, bind predictions, then release test."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'final_evaluation_pipeline'
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    temporary = Path(path).with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def state(stage, **extra):
    save(OUT / 'STATUS.json', dict(stage=stage, pid=os.getpid(), updated_at=time.time(),
         test_released=(HERE / 'TEST_RELEASE.json').exists(), **extra))


def run(stage, *arguments):
    command = [sys.executable, '-u', '-B', '-X', 'utf8', *map(str, arguments)]
    state(stage, command=command)
    with (OUT / (stage + '.log')).open('w', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    assert not (HERE / 'TEST_RELEASE.json').exists(), 'Do not restart after test release'
    state('waiting_for_all_frozen_model_predictions')
    needed = [HERE / 'RAG_PIPELINE_COMPLETE.json', HERE / 'prediction_pipeline/CPU_COMPLETE.json',
              HERE / 'fusion/nonlinear_window_v1/test_predictions.receipt.json']
    while not all(path.exists() for path in needed):
        for path in [HERE / 'baseline_audit/ragognizer_pipeline/STATUS.json',
                     HERE / 'prediction_pipeline/CPU_STATUS.json']:
            if path.exists() and read(path)['stage'] == 'failed':
                raise RuntimeError('Upstream failed: ' + str(path))
        time.sleep(50)
    rag = read(HERE / 'RAG_PIPELINE_COMPLETE.json')
    assert rag['status'] == 'complete' and rag['all_gpu_children_exited'] and rag['GPU_released']
    assert not rag['test_labels_read']
    run('assemble', HERE / 'prepare_final_comparison.py')
    run('freeze', HERE / 'evaluate_final.py', 'freeze', '--config', HERE / 'FINAL_COMPARISON.json')
    run('evaluate', HERE / 'evaluate_final.py', 'evaluate', '--release')
    result = read(HERE / 'results/TEST_RESULTS.json')
    assert len(result['methods']) == 12 and result['test_answers_all'] == 279
    save(OUT / 'COMPLETE.json', {'status': 'complete', 'all_children_exited': True,
         'frozen_predictions_sha256': result['frozen_predictions_sha256'],
         'primary_candidate': read(HERE / 'FINAL_COMPARISON.json')['primary_candidate'],
         'finished_at': time.time(), 'test_answers_primary': result['test_answers_primary']})
    state('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        state('failed', traceback=traceback.format_exc())
        raise
