import hashlib
import json
import re
import string
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRELAB=ROOT.parent
def readl(path): return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]
def writel(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()
def norm(s):
    s=s.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def token_f1(a,b):
    from collections import Counter
    aa=norm(a).split(); bb=norm(b).split()
    n=sum((Counter(aa)&Counter(bb)).values())
    return 2*n/(len(aa)+len(bb)) if aa or bb else 1.
