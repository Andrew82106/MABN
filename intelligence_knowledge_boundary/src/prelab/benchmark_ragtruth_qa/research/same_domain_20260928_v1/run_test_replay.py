"""After semantic training, generate test predictions without opening labels."""
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
    path = Path(path); temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def status(stage, **extra):
    save(OUT / 'GPU_STATUS.json', dict(stage=stage, pid=os.getpid(), updated_at=time.time(), test_labels_read=False, **extra))


def execute(name, args):
    command = [sys.executable, '-u', '-B', '-X', 'utf8', *map(str, args)]
    status(name, command=command)
    with (OUT / (name + '.log')).open('w', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    status('waiting_for_semantic_gpu_release')
    while not (HERE / 'model_design/PIPELINE_COMPLETE.json').exists():
        state = read(HERE / 'model_design/PIPELINE_STATUS.json')
        if state['stage'] == 'failed': raise RuntimeError('Semantic pipeline failed')
        time.sleep(50)
    semantic = read(HERE / 'model_design/PIPELINE_COMPLETE.json')
    assert semantic['GPU_released'] and semantic['all_child_processes_exited']
    predictor = HERE / 'model_design/predict_full_context.py'
    for name in ['base', 'large']:
        checkpoint = HERE / f'model_design/modernbert_{name}_v1'
        output = HERE / f'model_design/modernbert_{name}_test_v1'
        common = ['--checkpoint-dir', checkpoint, '--out', output]
        if not (output / 'prepared.json').exists(): execute(name + '_prepare', [predictor, 'prepare', *common])
        if not (output / 'cpu_check.json').exists(): execute(name + '_cpu_check', [predictor, 'cpu-check', *common])
        execute(name + '_predict', [predictor, 'predict', *common])
    execute('test_whitebox_signals', [HERE / 'whitebox_signals.py', 'extract', '--partitions', 'test'])
    done = read(HERE / 'signals/COMPLETE_test.json')
    assert done['count'] == done['expected_count'] == 279
    save(HERE / 'GPU_READY_FOR_RAGOGNIZER.json', {'test_signals': done,
         'all_gpu_children_exited': True, 'test_labels_read': False, 'created_at': time.time()})
    status('complete_gpu_released')


if __name__ == '__main__':
    try: main()
    except BaseException:
        status('failed', traceback=traceback.format_exc())
        raise
