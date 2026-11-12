"""Post-hoc descriptive checks of frozen OOF scores; no fitting or thresholds changed."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
def read(name): return json.loads((OUT / name).read_text(encoding='utf-8'))
def readl(name): return [json.loads(x) for x in (OUT / name).read_text(encoding='utf-8').splitlines() if x]

summary = read('summary.json')
answers = readl('answer_scores_oof.jsonl')
by_item = defaultdict(list)
for window in readl('window_scores_oof.jsonl'):
    by_item[window['item_ids'][0]].append(window)

result = {'scope': 'Post-hoc description of fixed training-source OOF predictions; no fitting, selection or threshold changes', 'methods': {}}
for name in summary['methods']:
    risk = [a for a in answers if a['main_eligible'] and a['gold'] == 1]
    misses = [a for a in risk if not a['predictions'][name]]
    def all_peak_risk(answer):
        windows = by_item[answer['item_id']]
        peak = max(w['scores'][name] for w in windows)
        return all(w['gold'] == 1 for w in windows if w['scores'][name] == peak)
    result['methods'][name] = {
        'risk_answers': len(risk), 'answer_false_negatives': len(misses),
        'risk_answers_all_peak_windows_risky': sum(all_peak_risk(a) for a in risk),
        'false_negatives_all_peak_windows_risky': sum(all_peak_risk(a) for a in misses),
        'false_negatives_with_any_window_alert': sum(any(w['predictions'][name] for w in by_item[a['item_id']]) for a in misses),
        'fold_window_thresholds': [summary['folds'][str(f)][name]['thresholds']['window']['threshold'] for f in range(5)],
        'fold_answer_thresholds': [summary['folds'][str(f)][name]['thresholds']['answer']['threshold'] for f in range(5)]}
result['all_methods_answer_false_negative_ids'] = [a['item_id'] for a in answers if a['main_eligible'] and a['gold'] == 1 and not any(a['predictions'].values())]
(OUT / 'score_diagnostics18.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
