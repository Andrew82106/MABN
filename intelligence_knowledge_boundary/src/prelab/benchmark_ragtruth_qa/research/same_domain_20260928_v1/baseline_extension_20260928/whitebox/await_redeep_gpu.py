"""Wait for the semantic queue to release GPU, then run frozen ReDeEP stages.

No model/precision/formula changes. Status polling is 50 seconds. Child stages
are recorded by the root run_stage.py wrapper, using the verified environment.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

HERE=Path(__file__).resolve().parent
EXT=HERE.parent
WORKSPACE=next(p for p in HERE.parents if p.name=='Multi_Agent_Graph_Analysis')
PYTHON=WORKSPACE/'intelligence_knowledge_boundary/src/prelab/.venv/Scripts/python.exe'
RUNNER=HERE/'run_whitebox.py'
SEMANTIC=EXT/'semantic'
SOURCE_COMPLETE=SEMANTIC/'GPU_QUEUE_COMPLETE.json'
STATUS=HERE/'REDEEP_GPU_QUEUE_STATUS.json'
COMPLETE=HERE/'REDEEP_GPU_QUEUE_COMPLETE.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pending.replace(path)


def running_entries(entries):
    found=[]
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            command=p.info['cmdline'] or []
            if any(any(str(entry).replace('\\','/').lower()==str(arg).replace('\\','/').lower()
                           for arg in command) for entry in entries):
                found.append({'pid':p.pid,'created':p.info['create_time'],'command':command})
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found


def latest_semantic_failures():
    failures=[]
    for path in (EXT/'execution').glob('*/LATEST.json'):
        if not path.parent.name.startswith(('lettuce_','minicheck_')):continue
        value=read(path)
        if value.get('status')=='failed':failures.append({'receipt':str(path),'value':value})
    for path in [SEMANTIC/'GPU_QUEUE_STATUS.json',SEMANTIC/'GPU_QUEUE_COMPLETE.json']:
        if path.exists():
            value=read(path)
            if value.get('status')=='failed':failures.append({'receipt':str(path),'value':value})
    return failures


def run_stage(name,*args):
    command=[str(PYTHON),'-u','-B','-X','utf8',str(EXT/'run_stage.py'),name,str(RUNNER),*args]
    save(STATUS,{'status':'running','stage':name,'pid':os.getpid(),'command':command,'at':time.time()})
    log_path=HERE/(name+'_queue_driver.log')
    with log_path.open('a',encoding='utf-8') as out:
        process=subprocess.Popen(command,cwd=WORKSPACE,stdout=out,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        save(STATUS,{'status':'running','stage':name,'pid':os.getpid(),'launcher_pid':process.pid,
                     'command':command,'at':time.time()})
        code=process.wait()
    if code!=0:raise RuntimeError(f'{name} exited {code}; inspect {log_path}')
    receipt=read(EXT/'execution'/name/'LATEST.json')
    assert receipt['status']=='complete' and receipt['exit_code']==0
    assert receipt['entry_sha256']==receipt['entry_sha256_after']==sha(RUNNER)
    assert not running_entries([RUNNER]), f'ReDeEP child still running after {name}'
    return receipt


def main():
    started=time.time()
    assert PYTHON.exists() and RUNNER.exists()
    assert not COMPLETE.exists(),'ReDeEP GPU queue already completed; do not restart'
    source=None
    while True:
        failures=latest_semantic_failures()
        if failures:
            save(HERE/'SEMANTIC_PREDECESSOR_FAILURE.json',{'failures':failures,'at':time.time()})
            raise RuntimeError('Semantic predecessor has a failed latest stage; no GPU work started')
        if SOURCE_COMPLETE.exists():
            source=read(SOURCE_COMPLETE)
            assert source.get('status')=='complete',source
            assert source.get('all_gpu_children_exited') is True and source.get('GPU_released') is True,source
            running=running_entries([SEMANTIC/'lettuce_v1.py',SEMANTIC/'minicheck_official.py'])
            if not running:break
        else:running=[]
        save(STATUS,{'status':'waiting','stage':'semantic_gpu_queue','source':str(SOURCE_COMPLETE),
            'poll_seconds':50,'semantic_processes':running,'pid':os.getpid(),'at':time.time()})
        time.sleep(50)
    save(HERE/'SEMANTIC_GPU_HANDOFF.json',{'source_complete_sha256':sha(SOURCE_COMPLETE),'source':source,
        'semantic_process_scan_empty':True,'at':time.time()})
    receipts=[]
    receipts.append(run_stage('redeep_smoke','extract','--method','redeep','--limit','2'))
    receipts.append(run_stage('redeep_extract','extract','--method','redeep'))
    receipts.append(run_stage('redeep_score','score','--method','redeep'))
    extraction=read(HERE/'redeep/EXTRACTION_COMPLETE.json')
    predictions=read(HERE/'redeep/PREDICTIONS_COMPLETE.json')
    assert extraction['status']=='complete' and extraction['records']==1842
    assert predictions['status']=='complete'
    counts={}
    prediction_receipts={}
    for partition,required in [('validation',271),('test',279)]:
        path=HERE/'redeep'/f'{partition}_predictions.jsonl'
        values=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]
        assert len(values)==len({r['response_id'] for r in values})==required
        receipt=read(path.with_suffix('.receipt.json'))
        assert receipt['predictions_sha256']==sha(path) and receipt['test_labels_read'] is False
        counts[partition]=required;prediction_receipts[partition]=receipt
    raw_counts={p:len(list((HERE/'redeep/features'/p).glob('*.npz'))) for p in ['fit','validation','test']}
    assert raw_counts=={'fit':1292,'validation':271,'test':279}
    assert not running_entries([RUNNER])
    value={'status':'complete','raw_cache_counts':raw_counts,'prediction_counts':counts,'prediction_total':550,
        'prediction_receipts':prediction_receipts,'stage_receipts':receipts,
        'all_gpu_children_exited':True,'GPU_released':True,'test_labels_read':False,
        'source_complete_sha256':sha(SOURCE_COMPLETE),'elapsed_seconds':time.time()-started,'at':time.time()}
    save(COMPLETE,value);save(STATUS,{'status':'complete','stage':'finished','at':time.time()})
    print(json.dumps({'status':'complete','raw_cache_counts':raw_counts,'prediction_counts':counts,'GPU_released':True}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc()
        save(STATUS,{'status':'failed','error':error,'at':time.time(),'pid':os.getpid()})
        print(error,flush=True)
        raise
