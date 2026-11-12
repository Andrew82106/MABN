from pathlib import Path
import hashlib,json,urllib.request
HERE=Path(__file__).resolve().parent
REV='d96e413aaf39f97739207759e8f393df1ecc38af' # official 0.1.6, 2025-02-27
FILES=['README.md','LICENSE','scripts/train.py','lettucedetect/models/trainer.py',
       'lettucedetect/models/evaluator.py','lettucedetect/datasets/ragtruth.py',
       'lettucedetect/models/inference.py','lettucedetect/preprocess/preprocess_ragtruth.py']
m={}
for f in FILES:
    p=HERE/'official'/'LettuceDetect_v1'/f;p.parent.mkdir(parents=True,exist_ok=True)
    u=f'https://raw.githubusercontent.com/KRLabsOrg/LettuceDetect/{REV}/{f}'
    with urllib.request.urlopen(u,timeout=90) as r: p.write_bytes(r.read())
    m[f]={'url':u,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for d in ['lettucedetect','lettucedetect/models','lettucedetect/datasets']:
    (HERE/'official'/'LettuceDetect_v1'/d/'__init__.py').touch(exist_ok=True)
(HERE/'LETTUCE_V1_SOURCE_MANIFEST.json').write_text(json.dumps({'revision':REV,'tag':'0.1.6','files':m},indent=2)+'\n')
print('LETTUCE_V1_READY')
