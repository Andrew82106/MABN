"""Resumable HTTP range fallback, official fixed revision and LFS SHA verified."""
from pathlib import Path
import concurrent.futures,hashlib,json,time,urllib.request
import common as c
REV='96eafd01cee2d16cf81aaa2fb226b14f422a37b3'
SIZE=3132786242
SHA='41291881e13c6235ed47149cec903bee9493e45d9d7325587a9fa2e266c526c0'
ROOT=c.HERE/'resources/MiniCheck-Flan-T5-Large'
PARTS=ROOT/'.http_parts';PARTS.mkdir(exist_ok=True)
STEP=16*1024*1024
URL=f'https://huggingface.co/lytang/MiniCheck-Flan-T5-Large/resolve/{REV}/pytorch_model.bin'

def fetch(i):
    a=i*STEP;b=min(SIZE,a+STEP)-1;p=PARTS/f'{i:04d}.part'
    if p.exists() and p.stat().st_size==b-a+1:return p
    for attempt in range(5):
        try:
            # Distinct query prevents intermediate caches from serving another range.
            req=urllib.request.Request(URL+f'?download=true&range_part={i}',headers={'Range':f'bytes={a}-{b}'})
            with urllib.request.urlopen(req,timeout=90) as r:
                assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{SIZE}', (r.status,r.headers.get('Content-Range'))
                t=p.with_suffix('.pending')
                with t.open('wb') as f:
                    while chunk:=r.read(1024*1024):f.write(chunk)
            assert t.stat().st_size==b-a+1;t.replace(p);return p
        except Exception as e:
            print('RETRY',i,attempt,type(e).__name__,str(e)[:150],flush=True)
            if attempt==4:raise
            time.sleep(2)

def main():
    start=time.time();n=(SIZE+STEP-1)//STEP;done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for p in pool.map(fetch,range(n)):
            done+=1
            if done%5==0:
                status={'parts':done,'total_parts':n,'bytes_completed':sum(x.stat().st_size for x in PARTS.glob('*.part')),'elapsed_seconds':time.time()-start}
                c.save(c.HERE/'DOWNLOAD_STATUS.json',status);print(status,flush=True)
    target=ROOT/'pytorch_model.bin';temp=ROOT/'pytorch_model.bin.pending'
    h=hashlib.sha256()
    with temp.open('wb') as f:
        for i in range(n):
            with (PARTS/f'{i:04d}.part').open('rb') as r:
                for block in iter(lambda:r.read(8*1024*1024),b''):f.write(block);h.update(block)
    assert temp.stat().st_size==SIZE and h.hexdigest()==SHA
    temp.replace(target)
    files={str(p.relative_to(ROOT)):c.sha(p) for p in ROOT.iterdir() if p.is_file()}
    c.save(c.HERE/'MODEL_MANIFEST.json',{'repo':'lytang/MiniCheck-Flan-T5-Large','revision':REV,'files_sha256':files,'official_lfs_sha256_verified':SHA,'download':'HTTP range fallback'})
    print('MODEL_DOWNLOAD_COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
