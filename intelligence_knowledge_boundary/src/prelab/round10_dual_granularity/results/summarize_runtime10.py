"""Summarize recorded timings; no model, gold or score access."""
from pathlib import Path
import json
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def summary(rows):
    times = [float(r['seconds']) for r in rows]
    return {'rows': len(rows), 'sum_recorded_seconds': sum(times),
            'mean_seconds': sum(times)/len(times) if times else None,
            'median_seconds': median(times) if times else None,
            'max_peak_allocated_gib': max((r['peak_allocated_gib'] for r in rows), default=None)}


def main():
    inputs = [json.loads(s) for s in (ROOT/'data/inputs.jsonl').read_text('utf-8').splitlines()]
    out = {}
    for split in ('train', 'validation', 'test'):
        rows = [r for r in inputs if r['split'] == split]
        for stage, folder in [('generate','generation_records'), ('core','features'), ('soft','soft_features')]:
            if split != 'test' and stage != 'soft':
                continue
            out[split+'__'+stage] = summary([json.loads((ROOT/'data'/folder/(r['row_id']+'.json')).read_text('utf-8')) for r in rows])
    out['scope'] = 'Recorded per-row GPU stages only; excludes model loading, source curation, annotation, audit and reuse of old train/validation generation/core. Prototype core and soft are two separate replay passes, not a measured speed comparison between detector methods.'
    for name, keys in [('freeze10.json', ['fit_wall_seconds']), ('metrics_test.json', ['load_seconds','cpu_score_seconds'])]:
        path = ROOT/'results'/name
        if path.exists():
            result = json.loads(path.read_text('utf-8'))
            out[name] = {k:result[k] for k in keys}
    (ROOT/'results/runtime10.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n', 'utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
