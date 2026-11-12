"""Write the explicit four-method extension config; no models/labels are loaded."""
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent
EXT = OUT.parent


def absolute(relative):
    return str((EXT / relative).resolve())


def main():
    methods = []
    for folder, name, description in [
        ('redeep', 'ReDeEP-paper-FP16', 'paper ECS/JSD; fixed k_head=1,k_layer=10,alpha=1,beta=.2; fit-token Pearson and fit-only MinMax; FP16 CPU offload'),
        ('lumina', 'LUMINA-NF4', 'official score formula; exact shared original prompt, fit-only label-blind donor; NF4 BF16 compute, FP32 reductions')]:
        base = 'whitebox/' + folder
        artifacts = ['whitebox/run_whitebox.py', 'whitebox/CONFIG.json', 'whitebox/plans.jsonl',
                     'whitebox/CPU_CHECK.json', base + '/RUNTIME.json', base + '/EXTRACTION_COMPLETE.json',
                     base + '/PREDICTIONS_COMPLETE.json', base + '/validation_predictions.receipt.json',
                     base + '/test_predictions.receipt.json']
        if folder == 'redeep':
            artifacts.append(base + '/FIT.json')
        methods.append({'name': name, 'role': 'baseline',
            'validation_predictions': absolute(base + '/validation_predictions.jsonl'),
            'test_predictions': absolute(base + '/test_predictions.jsonl'),
            'score_field': 'token_scores', 'scores_are_windows': False, 'pooling_candidates': ['mean', 'max'],
            'native_granularity': 'original Llama BPE token',
            'training_data': 'fit 1292 source-group-separated answers; validation thresholds only' if folder == 'redeep' else 'no detector fitting; validation thresholds only',
            'adaptation': description, 'model_artifacts': list(map(absolute, artifacts))})
    for folder, name, description, artifacts in [
        ('lettuce_v1_run', 'LettuceDetect-v1-retrained-BF16-forward',
         'unmodified upstream v0.1.6 dataset/two-logit CE/AdamW; ModernBERT-large generic initialization; FP32 trainable state, BF16 forward, batch8 via native-token-weighted accumulation',
         ['semantic/lettuce_v1.py', 'semantic/common.py', 'semantic/LETTUCE_V1_SOURCE_MANIFEST.json',
          'semantic/lettuce_v1_run/best.pt', 'semantic/lettuce_v1_run/SELECTED.json',
          'semantic/lettuce_v1_run/TRAIN_COMPLETE.json', 'semantic/lettuce_v1_run/HISTORY.json',
          'semantic/lettuce_v1_run/VALIDATION_REPLAY_AUDIT.json']),
        ('minicheck_run', 'MiniCheck-FlanT5-FP32-sentence',
         'unmodified official doc+claim inference/chunking; frozen released FlanT5-large FP32 checkpoint; original response sentences mapped to common BPE, no token detector training',
         ['semantic/minicheck_official.py', 'semantic/common.py', 'semantic/SOURCE_MANIFEST.json',
          'semantic/MODEL_MANIFEST.json', 'semantic/MODEL_REVISION.json'])]:
        base = 'semantic/' + folder
        artifacts += [base + '/PREPARED.json', base + '/CPU_CHECK.json', base + '/GPU_SMOKE.json',
                      base + '/validation_prepared.jsonl', base + '/test_prepared.jsonl',
                      base + '/validation_PREDICTION_RECEIPT.json', base + '/test_PREDICTION_RECEIPT.json']
        if folder == 'lettuce_v1_run':
            artifacts.append(base + '/fit_prepared.jsonl')
        methods.append({'name': name, 'role': 'baseline',
            'validation_predictions': absolute(base + '/validation_predictions.jsonl'),
            'test_predictions': absolute(base + '/test_predictions.jsonl'),
            'score_field': 'token_scores', 'scores_are_windows': False, 'pooling_candidates': ['mean', 'max'],
            'native_granularity': 'ModernBERT token' if folder == 'lettuce_v1_run' else 'sentence (BPE adapter only)',
            'training_data': 'fit 1292 only; checkpoints and thresholds selected on validation' if folder == 'lettuce_v1_run' else 'official frozen external checkpoint; validation thresholds only',
            'adaptation': description, 'model_artifacts': list(map(absolute, artifacts))})
    # Bind actual pretrained assets, in addition to their verified manifests,
    # when download has completed. Re-running this builder before freeze adds
    # the new weight files without changing any algorithm/selection definition.
    white_config = EXT / 'whitebox/CONFIG.json'
    if white_config.exists():
        config = json.loads(white_config.read_text(encoding='utf-8-sig'))
        model_dir = Path(config['model'])
        files = sorted(model_dir.glob('*.safetensors'))
        for method in methods[:2]:
            method['model_artifacts'] += [str(p.resolve()) for p in files]
            method['model_artifacts'] += list(config['code_sha256'])
    lettuce_sources = json.loads((EXT / 'semantic/LETTUCE_V1_SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    methods[2]['model_artifacts'] += [absolute('semantic/official/LettuceDetect_v1/' + f) for f in lettuce_sources['files']]
    semantic_sources = json.loads((EXT / 'semantic/SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    methods[3]['model_artifacts'] += [absolute('semantic/' + f) for f in semantic_sources
                                    if f.replace('\\', '/').startswith('official/MiniCheck/')]
    mini_manifest = EXT / 'semantic/MODEL_MANIFEST.json'
    if mini_manifest.exists():
        manifest = json.loads(mini_manifest.read_text(encoding='utf-8-sig'))
        methods[-1]['model_artifacts'] += [absolute('semantic/resources/MiniCheck-Flan-T5-Large/' + f)
                                            for f in manifest['files_sha256']]
    for method in methods:
        method['model_artifacts'] = list(dict.fromkeys(method['model_artifacts']))
    output = {'comparison': 'six independent baseline families; original Lookback/RAGognizer/HGB supplied by evaluator',
              'primary_candidate': 'Fusion-window-HGB', 'methods': methods}
    destination = OUT / 'new_methods.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    missing = [p for m in methods for p in [m['validation_predictions'], m['test_predictions'], *m['model_artifacts']] if not Path(p).exists()]
    print(json.dumps({'configuration': str(destination), 'ready_for_freeze': not missing, 'missing_count': len(missing)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
