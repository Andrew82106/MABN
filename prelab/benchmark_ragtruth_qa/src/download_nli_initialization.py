"""Download only the explicitly pinned public NLI initialization, never execute remote code."""
from pathlib import Path
import hashlib
import json
import time
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / 'models/ModernBERT-base-nli'
REPO = 'tasksource/ModernBERT-base-nli'
REVISION = 'de4ab7e77845098b7fab7f6ab9d370ddff27b19c'
NAMES = ('README.md', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    remote = {s.rfilename: s for s in info.siblings}
    started = time.perf_counter(); records = []
    for name in NAMES:
        item = remote[name]
        path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=MODEL))
        assert path.stat().st_size == item.size
        digest = sha(path)
        if item.lfs:
            assert digest == item.lfs.sha256
            verification = 'Official LFS SHA256 and local size/SHA256 exact'
        else:
            data = path.read_bytes()
            git_digest = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            assert git_digest == item.blob_id
            verification = 'Official Git blob SHA1 exact plus local SHA256'
        records.append({'file': name, 'bytes': item.size, 'sha256': digest, 'official_blob_id': item.blob_id,
            'official_lfs_sha256': item.lfs.sha256 if item.lfs else None, 'verification': verification})
        print('NLI_ASSET_VERIFIED', name, item.size, flush=True)
    manifest = {'repository': REPO, 'revision': REVISION, 'files': records,
        'files_sha256': {r['file']: r['sha256'] for r in records}, 'seconds': time.perf_counter()-started,
        'GPU_used': False, 'remote_code_executed': False, 'model_loaded': False,
        'purpose': 'Exploratory NLI encoder initialization; upstream provenance incompletely traced, not claimed contamination-free.'}
    (MODEL / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('NLI_PINNED_DOWNLOAD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
