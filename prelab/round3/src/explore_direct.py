"""Post-test uncertainty description of the validation-leading short-check method.

No fitting, threshold changes or new model calls. These comparisons were added
after viewing the test results and are explicitly exploratory, not confirmatory.
"""
import json
import numpy as np
from shared import ROOT,readl,sha
from evaluate_round3 import measure,bootstrap

def main():
    dest=ROOT/'results';test=readl(dest/'test_predictions.jsonl');val=readl(dest/'validation_predictions.jsonl')
    ann={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')}
    selection=json.loads((dest/'fusion_selection.json').read_text())
    winner=max(selection['methods'],key=lambda r:r['validation_global_auc_mean_policies'])['kind']
    assert winner=='direct_probs'
    truth=[np.array([int(any(lo<s['end'] and hi>s['start'] for s in ann[r['id']]['spans'])) for lo,hi in r['offsets']]) for r in test]
    names=['fusion_direct_probs_active','fusion_selection_only_active','fusion_baseline_active','linear_hybrid']
    cases={};predictions={};metrics={}
    for name in names:
        predictions[name]=[np.array(r['predictions'][name]) for r in test]
        metrics[name],cases[name]=measure(test,truth,predictions[name],val,[np.array(r['predictions'][name]) for r in val])
    comparisons=[(names[0],n) for n in names[1:]]
    result=bootstrap(cases,comparisons,truth,predictions)
    result.update(status='post-test exploratory comparisons, not preregistered confirmation',
        rationale='direct_probs had the highest mean validation AUROC among supplementary methods; inspect its uncertainty without refitting or changing thresholds',
        model_calls=0,refits=0,metrics=metrics,
        zero_of_40_false_alarms_two_sided_95pct_exact_upper=1-.025**(1/40),
        original_test_predictions_sha256=sha(dest/'test_predictions.jsonl'),
        original_selection_sha256=sha(dest/'fusion_selection.json'))
    (dest/'exploratory_direct_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result['paired_differences'],indent=2))

if __name__=='__main__':main()
