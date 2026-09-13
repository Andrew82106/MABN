"""Read-only live upstream gate check; never calls prepare/train or reads arrays."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_llama_harp_tcn as t

t.cpu_only()
assert not (t.producer.FEATURE/'feature_manifest.json').exists()
assert not (t.producer.OUT/'preparation_complete.json').exists()
assert t.upstream(False) is None
try:
    t.upstream(True)
except AssertionError as error:
    assert str(error)=='Complete3046 Llama manifest and producer numerical preparation required'
else:
    raise AssertionError('Missing upstream was accepted')
for name in ('matrices','preparation_started.json','preparation_manifest.json','started.json','complete.json'):
    assert not (t.OUT/name).exists()
t.save(t.OUT/'MISSING_UPSTREAM_GATE.json',{
    'passed':True,'utc':t.now(),'code_sha256':t.sha(t.__file__),
    'protocol_sha256':t.sha(t.OUT/'protocol.json'),
    'freeze_sha256':t.sha(t.OUT/'freeze.json'),
    'helper_sha256':t.sha(__file__),
    'actual_complete3046_manifest_missing':True,
    'actual_producer_numerical_preparation_missing':True,
    'optional_gate_returned_none':True,'required_gate_rejected':True,
    'formal_prepare_called':False,'formal_training_called':False,
    'numerical_matrices_created':False,'GPU_used':False,'official_test_opened':False})
print('LIVE_MISSING_UPSTREAM_GATE_PASSED_NO_PREP_NO_TRAIN',flush=True)
