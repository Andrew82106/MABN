"""Pinned, chunk-resumable assets with upstream size and SHA256 verification."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
PRELAB=ROOT.parent
REPO='unsloth/Qwen2.5-7B-Instruct-bnb-4bit'
REV='bdd404162d94997f390efbfa660eb3f21cbbc81d'
SIZE=5547254622
HASH='99b85155b7bb40344d8c3938c8ec1147b27028804bca5d2e5559cb8e6533747d'
DEST=PRELAB/'models/Qwen2.5-7B-Instruct-bnb-4bit'
CHUNK=32*1024*1024

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024): h.update(block)
    return h.hexdigest()

def part(i):
    start=i*CHUNK; end=min(SIZE,start+CHUNK)-1
    path=DEST/'chunks'/f'{i:04d}.part'
    for attempt in range(12):
        have=path.stat().st_size if path.exists() else 0
        if have==end-start+1: return path
        lo=start+have
        host='https://hf-mirror.com' if attempt<8 else 'https://huggingface.co'
        url=f'{host}/{REPO}/resolve/{REV}/model.safetensors?chunk={i}&offset={lo}'
        try:
            with requests.get(url,headers={'Range':f'bytes={lo}-{end}'},timeout=(30,90),stream=True) as r:
                r.raise_for_status()
                expected=f'bytes {lo}-{end}/{SIZE}'
                if r.status_code!=206 or r.headers.get('Content-Range')!=expected:
                    raise RuntimeError(f'Range mismatch: {r.status_code} {r.headers.get("Content-Range")}')
                with path.open('ab') as f:
                    for b in r.iter_content(1024*1024):
                        if b: f.write(b)
            if path.stat().st_size==end-start+1:
                print(f'chunk {i+1}/{(SIZE+CHUNK-1)//CHUNK} verified size',flush=True)
                return path
        except Exception as e: print(f'retry chunk={i} attempt={attempt} {type(e).__name__}: {str(e)[:140]}',flush=True)
        time.sleep(2)
    raise RuntimeError(f'chunk {i} incomplete')

def main():
    DEST.mkdir(parents=True,exist_ok=True); (DEST/'chunks').mkdir(exist_ok=True)
    metadata=requests.get(f'https://hf-mirror.com/api/models/{REPO}/revision/{REV}?blobs=true',timeout=30).json()
    (ROOT/'results/model_upstream.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    for item in metadata['siblings']:
        name=item['rfilename']
        if name=='model.safetensors' or name=='.gitattributes': continue
        path=DEST/name
        if not path.exists() or path.stat().st_size!=item['size']:
            r=requests.get(f'https://hf-mirror.com/{REPO}/resolve/{REV}/{name}',timeout=90); r.raise_for_status()
            assert len(r.content)==item['size'],name; path.write_bytes(r.content)
        if item.get('lfs'): assert sha(path)==item['lfs']['sha256']
    model=DEST/'model.safetensors'
    if not (model.exists() and model.stat().st_size==SIZE and sha(model)==HASH):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            paths=list(pool.map(part,range((SIZE+CHUNK-1)//CHUNK)))
        with model.open('wb') as f:
            for path in paths:
                with path.open('rb') as r:
                    while b:=r.read(8*1024*1024): f.write(b)
        assert model.stat().st_size==SIZE and sha(model)==HASH,'Upstream hash mismatch'
    manifest={'repo':REPO,'revision':REV,'quantization':'third-party Unsloth bitsandbytes NF4 quantization of Qwen2.5-7B-Instruct; not official full-precision weights',
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in DEST.iterdir() if p.is_file()}}
    (ROOT/'results/model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('MODEL COMPLETE AND VERIFIED',flush=True)

if __name__=='__main__': main()
