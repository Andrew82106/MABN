"""Fetch released data and reject changed contents instead of silently changing the run."""
import hashlib
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'response.jsonl':'e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073',
          'source_info.jsonl':'0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b'}
dest=ROOT/'data/raw'; dest.mkdir(parents=True,exist_ok=True)
for name,digest in EXPECTED.items():
    path=dest/name
    if not path.exists():
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{name}',timeout=90) as r:
            data=r.read()
        assert hashlib.sha256(data).hexdigest()==digest,f'Upstream changed: {name}'
        path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    print('Verified',name)
license_path=dest/'LICENSE'
if not license_path.exists():
    license_path.write_bytes(urllib.request.urlopen('https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/LICENSE',timeout=30).read())
