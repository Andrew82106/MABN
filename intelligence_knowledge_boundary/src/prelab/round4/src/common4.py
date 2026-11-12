import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT.parent
sys.path.append(str(PRE/'round2/src'))
def readl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines()]
def writel(p,rows):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
