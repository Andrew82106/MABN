"""Root-authorized exclusive GPU queue, after the already-running training exits."""
import os,subprocess,sys,time
from pathlib import Path
import psutil
import common as c

EXT=c.HERE.parent
PYTHON=c.PROJECT/'src/prelab/.venv/Scripts/python.exe'
STAGES=[('lettuce_validation','lettuce_v1.py',['predict','--split','validation']),
        ('lettuce_test','lettuce_v1.py',['predict','--split','test']),
        ('minicheck_smoke','minicheck_official.py',['smoke']),
        ('minicheck_validation','minicheck_official.py',['predict','--split','validation']),
        ('minicheck_test','minicheck_official.py',['predict','--split','test'])]

def same_child_running(receipt):
    try:
        p=psutil.Process(receipt['child_pid'])
        return p.create_time()>=receipt['started_at']-3 and p.create_time()<receipt.get('completed_at',time.time())+3
    except psutil.NoSuchProcess:return False

def main():
    start=time.time();completed=[]
    try:
        while True:
            train=c.read(EXT/'execution/lettuce_train/LATEST.json')
            if train['status']=='failed':raise RuntimeError('Lettuce training failed; downstream queue not started')
            if train['status']=='complete' and not same_child_running(train):break
            progress=c.read(c.HERE/'lettuce_v1_run/STATUS.json')
            c.save(c.HERE/'GPU_QUEUE_STATUS.json',{'status':'waiting_for_training','observed_training':train['status'],'progress':progress,'queue_pid':os.getpid()})
            print('WAIT_TRAIN',progress.get('epoch'),progress.get('answers'),flush=True)
            time.sleep(50)
        assert train['exit_code']==0 and train['entry_sha256']==train['entry_sha256_after']
        for name,script,args in STAGES:
            c.save(c.HERE/'GPU_QUEUE_STATUS.json',{'status':'running','stage':name,'completed':completed,'queue_pid':os.getpid()})
            receipt_path=EXT/'execution'/name/'LATEST.json'
            if receipt_path.exists():
                old=c.read(receipt_path)
                if old['status']=='running' and same_child_running(old):raise RuntimeError('Refusing to duplicate active stage '+name)
                if old['status']=='complete' and old.get('exit_code')==0 and old['entry_sha256_after']==c.sha(c.HERE/script):
                    completed.append(name);continue
            command=[str(PYTHON),'-u','-B','-X','utf8',str(EXT/'run_stage.py'),name,str(c.HERE/script),*args]
            print('START_GPU_STAGE',name,flush=True)
            process=subprocess.Popen(command,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            code=process.wait();receipt=c.read(receipt_path)
            assert code==0 and receipt['status']=='complete' and receipt['exit_code']==0,(name,code,receipt)
            assert not same_child_running(receipt), name+' child still running'
            assert receipt['entry_sha256']==receipt['entry_sha256_after'], name+' code changed while running'
            completed.append(name);print('DONE_GPU_STAGE',name,flush=True)
        outputs={str(p.relative_to(c.HERE)):c.sha(p) for folder in ['lettuce_v1_run','minicheck_run']
                 for p in (c.HERE/folder).glob('*predictions.jsonl')}
        c.save(c.HERE/'GPU_QUEUE_COMPLETE.json',{'status':'complete','stages':completed,'outputs_sha256':outputs,
          'all_gpu_children_exited':True,'GPU_released':True,'test_labels_read':False,'elapsed_seconds':time.time()-start})
        c.save(c.HERE/'GPU_QUEUE_STATUS.json',{'status':'complete','completed':completed,'GPU_released':True})
        print('SEMANTIC_GPU_QUEUE_COMPLETE',flush=True)
    except Exception as e:
        c.save(c.HERE/'GPU_QUEUE_STATUS.json',{'status':'failed','completed':completed,'error':repr(e),'queue_pid':os.getpid()})
        raise

if __name__=='__main__':main()
