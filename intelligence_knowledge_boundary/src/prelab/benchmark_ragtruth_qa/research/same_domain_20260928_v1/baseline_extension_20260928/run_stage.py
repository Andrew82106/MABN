"""Run one root-scheduled stage and retain its command, log and exit status."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    name, script, *arguments = sys.argv[1:]
    assert name and all(c.isalnum() or c in '_-' for c in name)
    script = Path(script).resolve(strict=True)
    out = HERE / 'execution' / name
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    command = [sys.executable, '-u', '-B', '-X', 'utf8', str(script), *arguments]
    log_path = out / (stamp + '.log')
    receipt = {'name': name, 'command': command, 'cwd': str(Path.cwd()),
               'entry_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
               'started_at': time.time(), 'log': str(log_path), 'launcher_pid': os.getpid()}
    receipt_path = out / (stamp + '.json')
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace',
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        receipt.update(child_pid=process.pid, status='running')
        save(receipt_path, receipt)
        save(out / 'LATEST.json', receipt)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line.rstrip(), flush=True)
        code = process.wait()
    receipt.update(exit_code=code, status='complete' if code == 0 else 'failed',
                   completed_at=time.time(), elapsed_seconds=time.time() - receipt['started_at'],
                   entry_sha256_after=hashlib.sha256(script.read_bytes()).hexdigest())
    save(receipt_path, receipt)
    save(out / 'LATEST.json', receipt)
    print('STAGE_COMPLETE', name, code, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
