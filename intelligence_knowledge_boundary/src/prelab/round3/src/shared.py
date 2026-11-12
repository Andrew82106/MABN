import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT.parent; OLD=PRE/'round2'
sys.path.insert(0,str(OLD/'src'))
def readl(p): return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
def writel(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def data():
    rows=readl(ROOT/'data/rows.jsonl')
    return {s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
def features(r):
    return Path(r['feature_path'])
