"""Produce fixed-model predictions for fusion and baselines; never score test."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'prediction_pipeline'
OUT.mkdir(exist_ok=True)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    temp = Path(path).with_suffix('.pending')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def state(stage, **extra):
    save(OUT / 'CPU_STATUS.json', dict(stage=stage, pid=os.getpid(), updated_at=time.time(), test_labels_read=False, **extra))


def run(name, args, python=sys.executable):
    command = [str(python), '-u', '-B', '-X', 'utf8', *map(str, args)]
    state(name, command=command)
    with (OUT / (name + '.log')).open('w', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    state('waiting_for_frozen_models_and_test_signals')
    needed = [HERE / 'fusion/PIPELINE_COMPLETE.json', HERE / 'GPU_READY_FOR_RAGOGNIZER.json',
              HERE / 'baseline_audit/lookback_run_v2/COMPLETE.json']
    while not all(p.exists() for p in needed):
        for path in [HERE / 'fusion/PIPELINE_STATUS.json', OUT / 'GPU_STATUS.json']:
            if path.exists() and read(path)['stage'] == 'failed': raise RuntimeError('Upstream failed: ' + str(path))
        time.sleep(50)
    run('whitebox_test', [HERE / 'train_whitebox.py', 'predict', '--out', HERE / 'whitebox_model', '--predict-partition', 'test'])
    chosen = read(HERE / 'model_design/SEMANTIC_SELECTION.json')['chosen']
    semantic = HERE / f'model_design/modernbert_{chosen}_test_v1/predictions.jsonl'
    whitebox = HERE / 'whitebox_model/token_predictions_test.jsonl'
    for kind in ['token', 'window']:
        directory = HERE / f'fusion/{kind}_v1'
        run(kind + '_fusion_test', [HERE / 'train_fusion.py', 'predict', '--out', directory,
            '--semantic', semantic, '--whitebox', whitebox, '--destination', directory / 'test_predictions.jsonl'])
    for variant, directory in [('default', HERE / 'baseline_audit/lookback_run_v2'),
                               ('tuned', HERE / 'baseline_audit/lookback_run_v2/tuned')]:
        for split in ['validation', 'test']:
            run('lookback_' + variant + '_' + split,
                [HERE / 'baseline_audit/predict_frozen_baselines.py', 'predict', '--method', 'lookback',
                 '--partition', split, '--run-dir', directory, '--out', HERE / f'baseline_audit/predictions/lookback_{variant}_{split}'])
    save(OUT / 'CPU_COMPLETE.json', {'selected_semantic': chosen, 'test_labels_read': False,
                                   'all_cpu_children_exited': True})
    state('complete')


if __name__ == '__main__':
    try: main()
    except BaseException:
        state('failed', traceback=traceback.format_exc())
        raise
