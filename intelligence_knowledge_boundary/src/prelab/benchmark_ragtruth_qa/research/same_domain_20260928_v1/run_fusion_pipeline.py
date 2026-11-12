"""Current-run CPU continuation once the two honest cross-fits finish."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fusion'
OUT.mkdir(exist_ok=True)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    target = OUT / name
    temp = target.with_suffix('.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(target)


def state(stage, **extra):
    save('PIPELINE_STATUS.json', dict(stage=stage, pid=os.getpid(), updated_at=time.time(), **extra))


def main():
    semantic_ready = HERE / 'model_design/PIPELINE_COMPLETE.json'
    whitebox_ready = HERE / 'whitebox_model/OOF_READY.json'
    state('waiting_for_both_oof_branches')
    while not (semantic_ready.exists() and whitebox_ready.exists()):
        for path, field in [(HERE / 'model_design/PIPELINE_STATUS.json', 'stage'),
                            (HERE / 'whitebox_model/PIPELINE_STATE.json', 'stage')]:
            if path.exists() and read(path).get(field) == 'failed':
                raise RuntimeError('Upstream failed: ' + str(path))
        time.sleep(50)
    semantic, whitebox = read(semantic_ready), read(whitebox_ready)
    chosen = semantic['selection']['chosen']
    folds = []
    for entry in sorted(semantic['fold_manifests'], key=lambda e: e['fold']):
        manifest = read(entry['manifest'])
        folds.append({'training_ids': manifest['fit_ids']['path'], 'predictions': manifest['predictions']['path'],
                      'manifest': entry['manifest']})
    config = {'semantic': {'folds': folds, 'validation': str(HERE / f'model_design/modernbert_{chosen}_v1/best_validation_predictions.jsonl')},
              'whitebox': {'folds': whitebox['folds'], 'validation': whitebox['validation']}}
    save('OOF_CONFIG.json', config)
    selected = []
    for granularity in ['token', 'window']:
        folder = OUT / (granularity + '_v1')
        state('training_fusion', granularity=granularity)
        command = [sys.executable, '-u', '-B', str(HERE / 'train_fusion.py'), 'train',
                   '--config', str(OUT / 'OOF_CONFIG.json'), '--out', str(folder), '--granularity', granularity]
        with (OUT / (granularity + '_training.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        best = read(folder / 'SELECTED.json')
        selected.append({'granularity': granularity, 'out': str(folder), 'name': best['name'],
                         'selection_key': best['report']['selection_key'], 'thresholds': best['thresholds'],
                         'validation': best['report']})
    # Exact tie keeps the directly trained window method, matching the task.
    winner = max(selected, key=lambda e: tuple(e['selection_key']) + (int(e['granularity'] == 'window'),))
    save('PIPELINE_COMPLETE.json', {'selected_fusion': winner, 'both_versions': selected,
         'fixed_average_is_reported_separately': True, 'semantic_backbone': chosen,
         'test_labels_read': False, 'test_predictions_created': False})
    state('complete', selected=winner)
    print(json.dumps({'complete': True, 'selected_granularity': winner['granularity'],
                      'validation_key': winner['selection_key']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException:
        state('failed', traceback=traceback.format_exc())
        raise
