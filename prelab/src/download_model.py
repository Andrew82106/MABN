"""Pinned model download, official host with mirror fallback; no remote code."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = '7ae557604adf67be50417f59c2c2f167def9a775'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DEST = ROOT / 'models' / 'Qwen2.5-0.5B-Instruct'
FILES = ['config.json', 'generation_config.json', 'merges.txt', 'model.safetensors',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'LICENSE', 'README.md']
MODEL_SIZE=988097824
MODEL_SHA='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'

def model_download():
    path=DEST/'model.safetensors'
    # The mirror may close a large response early; verify advertised upstream LFS size/hash.
    for attempt in range(10):
        size=path.stat().st_size if path.exists() else 0
        if size==MODEL_SIZE:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=MODEL_SHA: raise RuntimeError('Model SHA256 differs from upstream LFS')
            return {'file':path.name,'bytes':size,'sha256':digest,'upstream_lfs_verified':True}
        if size>MODEL_SIZE: raise RuntimeError('Oversized model download')
        print(f'Resume model at {size}/{MODEL_SIZE}',flush=True)
        host='https://hf-mirror.com' if attempt<5 else 'https://huggingface.co'
        url=f'{host}/{MODEL}/resolve/{REV}/model.safetensors?resume={size}'
        headers={'Range':f'bytes={size}-'} if size else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as r:
                if size and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {size}-')):
                    raise RuntimeError('Server did not honor resume offset')
                with path.open('ab' if size else 'wb') as f:
                    while block:=r.read(1024*1024): f.write(block)
        except Exception as e: print(type(e).__name__,str(e)[:150],flush=True)
    raise RuntimeError('Model incomplete after retries')

def download(name):
    if name=='model.safetensors': return model_download()
    path = DEST / name
    if not path.exists():
        for host in ['https://hf-mirror.com', 'https://huggingface.co']:
            url = f'{host}/{MODEL}/resolve/{REV}/{name}'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'prelab-research/1.0'})
                with urllib.request.urlopen(req, timeout=90) as r, path.with_suffix(path.suffix+'.part').open('wb') as f:
                    size = 0
                    while chunk := r.read(4*1024*1024):
                        f.write(chunk)
                        size += len(chunk)
                        if size % (100*1024*1024) == 0:
                            print(f'{name}: {size/1024**2:.0f} MB', flush=True)
                path.with_suffix(path.suffix+'.part').replace(path)
                break
            except Exception as e:
                print(f'{name}: {host}: {type(e).__name__}: {e}', flush=True)
        else:
            raise RuntimeError(f'Download failed: {name}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'OK {name} {path.stat().st_size}', flush=True)
    return {'file': name, 'bytes': path.stat().st_size, 'sha256': digest}

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, FILES))
    (ROOT/'results'/'model_manifest.json').write_text(json.dumps({'model':MODEL, 'revision':REV,'files':files},indent=2),encoding='utf-8')
