"""Download pinned public source and MiniCheck weights. Never initializes CUDA."""
from pathlib import Path
import os
os.environ['HF_HUB_DISABLE_XET']='1'
import hashlib, json, shutil, urllib.request
from huggingface_hub import HfApi, snapshot_download

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[6]
LETTUCE_REV='2096ed28f3b662a62b4406795da3d6fbc5490063'
MINICHECK_REV='b58b9fa69acbd1015ec970fa65dd752413a053d2'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    manifest={}
    sources={
      'LettuceDetect':('KRLabsOrg/LettuceDetect',LETTUCE_REV,[
        'README.md','LICENSE','scripts/train.py','lettucedetect/models/trainer.py',
        'lettucedetect/models/evaluator.py','lettucedetect/datasets/hallucination_dataset.py',
        'lettucedetect/models/inference.py','lettucedetect/preprocess/preprocess_ragtruth.py']),
      'MiniCheck':('Liyan06/MiniCheck',MINICHECK_REV,[
        'README.md','LICENSE','minicheck/inference.py','minicheck/minicheck.py','minicheck/utils.py'])}
    for name,(repo,revision,files) in sources.items():
        for f in files:
            p=HERE/'official'/name/f; p.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://raw.githubusercontent.com/{repo}/{revision}/{f}'
            if not p.exists():
                with urllib.request.urlopen(url,timeout=90) as r: p.write_bytes(r.read())
            manifest[str(p.relative_to(HERE))]={'url':url,'sha256':sha(p)}
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['MiniCheck/minicheck','LettuceDetect/lettucedetect','LettuceDetect/lettucedetect/datasets']:
        (HERE/'official'/name/'__init__.py').touch(exist_ok=True)
    repo='lytang/MiniCheck-Flan-T5-Large'
    dest=HERE/'resources'/'MiniCheck-Flan-T5-Large'
    lock=HERE/'MODEL_REVISION.json'
    if lock.exists(): revision=json.loads(lock.read_text())['revision']
    else:
        revision=HfApi().model_info(repo).sha
        lock.write_text(json.dumps({'repo':repo,'revision':revision},indent=2)+'\n')
    snapshot_download(repo,revision=revision,local_dir=dest,
        allow_patterns=['*.json','*.model','*.safetensors','pytorch_model.bin','README.md'],max_workers=2)
    files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and '.cache' not in p.parts}
    (HERE/'MODEL_MANIFEST.json').write_text(json.dumps({'repo':repo,'revision':revision,'files_sha256':files},indent=2)+'\n')
    print('RESOURCES_READY',len(manifest),len(files),flush=True)

if __name__=='__main__': main()
