"""Pinned generic ModernBERT-large, resumable anonymous download; never uses GPU."""
from pathlib import Path
import requests
import download_checkpoint as transport

REPO = 'answerdotai/ModernBERT-large'
REVISION = '45bb4654a4d5aaff24dd11d4781fa46d39bf8c13'
MODEL = Path(__file__).resolve().parents[2] / 'models/ModernBERT-large'
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'README.md', 'model.safetensors']


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    transport.TARGET = MODEL
    transport.MANIFEST = MODEL / 'download_manifest.json'
    transport.REPO = REPO
    transport.REVISION = REVISION
    session = requests.Session()
    session.trust_env = False
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    response = session.get(url, timeout=(20, 45))
    response.raise_for_status()
    info = response.json()
    assert info['sha'] == REVISION and info['gated'] is False
    siblings = {s['rfilename']: s for s in info['siblings']}
    records = []
    for name in FILES:
        f = siblings[name]
        records.append({'filename': name, 'expected_bytes': f['size'],
                        'expected_git_blob_sha1': f['blobId'],
                        'expected_lfs_sha256': (f.get('lfs') or {}).get('sha256'),
                        'status': 'pending', 'downloaded_bytes': 0})
    state = {'schema_version': 1, 'repo_id': REPO, 'revision': REVISION,
             'target_directory': str(MODEL), 'status': 'starting',
             'started_at_utc': transport.now(), 'anonymous_access': True,
             'license': info.get('cardData', {}).get('license'),
             'account_authorization_modified': False, 'model_loaded': False,
             'GPU_used': False, 'ragtruth_finetuned_checkpoint': False,
             'download_format': 'safetensors', 'source_metadata_url': url,
             'total_expected_bytes': sum(r['expected_bytes'] for r in records), 'files': records}
    transport.save(state)
    print('START', state['total_expected_bytes'], REVISION, flush=True)
    for rec in records:
        transport.download(session, rec, state)
    state.update(status='complete', current_file=None, completed_at_utc=transport.now(),
                 all_files_source_hash_matched=all(r.get('source_hash_match') for r in records))
    transport.save(state)
    print('COMPLETE', transport.MANIFEST, flush=True)


if __name__ == '__main__':
    main()
