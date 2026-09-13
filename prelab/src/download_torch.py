"""Resumable ranged download from the official PyTorch wheel host."""
import concurrent.futures
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
URL='https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl'
SIZE=2532350702
CHUNK=32*1024*1024
DEST=ROOT/'models/wheels'
DEST.mkdir(parents=True,exist_ok=True)
def part(i):
    start=i*CHUNK; end=min(SIZE,start+CHUNK)-1
    path=DEST/f'torch.part{i:03d}'
    if path.exists() and path.stat().st_size==end-start+1: return path
    for attempt in range(5):
        try:
            req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=60) as r:
                assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),r.headers
                with path.open('wb') as f:
                    while block:=r.read(1024*1024): f.write(block)
            assert path.stat().st_size==end-start+1
            print(f'part {i+1}/{(SIZE+CHUNK-1)//CHUNK}',flush=True)
            return path
        except Exception as e:
            print(f'retry {i} {type(e).__name__}',flush=True)
            time.sleep(2)
    raise RuntimeError(f'part {i} failed')
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    parts=list(pool.map(part,range((SIZE+CHUNK-1)//CHUNK)))
out=DEST/'torch-2.6.0+cu124-cp311-cp311-win_amd64.whl'
with out.open('wb') as f:
    for p in parts:
        with p.open('rb') as r:
            while block:=r.read(8*1024*1024): f.write(block)
assert out.stat().st_size==SIZE
print(str(out),flush=True)
