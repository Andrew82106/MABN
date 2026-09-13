"""Immutable data/method contracts. Does not open any raw test content."""
from pathlib import Path
import hashlib,json,math
from datetime import datetime

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
IDENTITY=ROOT/'data/test_release_identity_manifest.json'
AUTHORIZATION=ROOT/'data/FINAL_TEST_RELEASE_AUTHORIZATION.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def text_sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),('Refuse to overwrite frozen output',str(p))
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def save_lines(p,rows):
    p=Path(p);assert not p.exists()
    p.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows),encoding='utf-8')
def check_hashes(values):
    assert values and isinstance(values,dict)
    for p,h in values.items():assert sha(p)==h,('Frozen file changed',p)

def validate_methods(freeze):
    assert freeze['complete'] and freeze['status']=='frozen_before_test_release'
    assert freeze['selection_data']=='fit_and_calibration_only' and not freeze['test_content_used_for_selection']
    assert freeze['methods'] and len({m['method_id'] for m in freeze['methods']})==len(freeze['methods'])
    assert any(m['primary'] for m in freeze['methods'])
    check_hashes(freeze['development_files_sha256'])
    for m in freeze['methods']:
        assert isinstance(m['method_id'],str) and m['method_id'] and isinstance(m['primary'],bool)
        assert m['score_direction']=='larger_is_risk' and m['answer_aggregation']=='max_all_eligible_windows'
        assert m['window_score_interface']=='all_eligible_original_k4_windows'
        assert m['prediction_unit'] in ('raw_token','raw_window','claim','answer_conditioned_window')
        assert all(isinstance(m[k],(int,float)) and not isinstance(m[k],bool) and math.isfinite(m[k]) for k in ('window_threshold','answer_threshold'))
        check_hashes(m['artifacts_sha256'])
        assert m['selection_record_sha256'] in m['artifacts_sha256'].values()
    b=freeze['group_bootstrap']
    assert b['unit']=='group_id' and b['algorithm']=='pooled_confusion_percentile' and b['quantile_method']=='linear'
    assert isinstance(b['replicates'],int) and b['replicates']>0 and isinstance(b['seed'],int)
    assert 0<b['confidence_level']<1 and b['zero_denominator_value']==0
    return freeze

def require_release(development_freeze_path):
    """Must run before raw-file hashes, quality, response or label access."""
    assert AUTHORIZATION.exists(),'No explicit root release authorization; test remains sealed'
    auth=read(AUTHORIZATION);fp=Path(development_freeze_path)
    assert auth['authorized'] is True and auth['authorization']=='root_explicit_release'
    assert auth['scope']=='ragtruth_qa_official_test_150_llama2_only'
    assert auth['development_freeze_sha256']==sha(fp)
    assert auth['identity_manifest_sha256']==sha(IDENTITY)
    assert auth['tools_protocol_sha256']==sha(HERE/'protocol.json')
    assert auth['authorization_utc'] and auth['development_freeze_path']==str(fp.resolve())
    f=validate_methods(read(fp))
    assert f['purpose']=='official_test_release' and f['identity_manifest_sha256']==sha(IDENTITY)
    assert f['tools_protocol_sha256']==sha(HERE/'protocol.json')
    assert datetime.fromisoformat(f['frozen_at_utc'])<=datetime.fromisoformat(auth['authorization_utc'])
    identity=read(IDENTITY);idx=lines(ROOT/'data/source_index.jsonl')
    assert identity['source_index_sha256']==sha(ROOT/'data/source_index.jsonl')
    assert identity['annotation_protocol_sha256']==sha(ROOT/'ANNOTATION_PROTOCOL.md')
    assert identity['gold_manifest_sha256']==sha(ROOT/'data/gold_manifest.json')
    assert identity['release_plan_sha256']==sha(ROOT/'TEST_RELEASE_PLAN.md')
    assert len(identity['identities'])==150 and len({r['source_id'] for r in identity['identities']})==150
    assert len({r['group_id'] for r in identity['identities']})==147
    expected=[{k:r[k] for k in ('source_id','group_id','response_id','official_split','native_model')} for r in idx if r['partition']=='sealed_official_test']
    assert identity['identities']==expected
    dg={r['group_id'] for r in idx if r['partition'] in ('fit','calibration')}
    assert not dg&{r['group_id'] for r in expected}
    assert all(r['official_split']=='test' and r['native_model']=='llama-2-7b-chat' for r in expected)
    p=read(HERE/'protocol.json');check_hashes(p['tool_files_sha256'])
    return f,identity,{'development_freeze_sha256':sha(fp),'identity_manifest_sha256':sha(IDENTITY),'authorization_sha256':sha(AUTHORIZATION),'tools_protocol_sha256':sha(HERE/'protocol.json')}

def check_bundle(path):
    p=Path(path);m=read(p)
    assert m['complete'] and m['selfcheck_passed']
    for n,h in m['outputs_sha256'].items():assert sha(p.parent/n)==h,n
    return m
