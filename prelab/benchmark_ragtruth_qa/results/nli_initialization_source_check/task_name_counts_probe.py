"""Read only pinned Parquet metadata/task_name column via bounded HTTP Range."""
import io
import json
import hashlib
import threading
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pyarrow.parquet as pq

REV='1e009645b2943106614107b06107b1ee85ac1161'
URL=f'https://huggingface.co/datasets/MoritzLaurer/dataset_train_nli/resolve/{REV}/data/train-00000-of-00001.parquet'
SIZE=206032209
OUT=Path(__file__).resolve().parent
EXCLUDED={'mnli','anli','fevernli','wanli','lingnli'}


class Ranges(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.url=URL;self.cache=[];self.bytes=0;self.requests=0
        self.lock=threading.Lock();self.local=threading.local();self.remote_allowed=True
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        assert 0<=self.pos<=SIZE
        return self.pos
    def fetch(self,a,n):
        assert self.remote_allowed and 0<n<=2_000_000
        if not hasattr(self.local,'session'):self.local.session=requests.Session()
        for attempt in range(3):
            try:
                r=self.local.session.get(self.url,headers={'Range':f'bytes={a}-{a+n-1}'},timeout=30,stream=True)
                if r.status_code!=206:
                    status=r.status_code;r.close();raise RuntimeError(f'Range required; HTTP {status}, full body not downloaded')
                cr=r.headers.get('Content-Range','')
                assert cr==f'bytes {a}-{a+n-1}/{SIZE}',cr
                data=r.raw.read(n+1);final_url=r.url;r.close();assert len(data)==n
                with self.lock:
                    assert self.bytes+len(data)<=4_000_000,'STOP partial-transfer budget'
                    self.bytes+=len(data);self.requests+=1;self.cache.append((a,data));self.url=final_url
                return data
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(.5)
    def read(self,n=-1):
        if n<0:n=SIZE-self.pos
        n=min(n,SIZE-self.pos)
        if not n:return b''
        a=self.pos
        for start,data in self.cache:
            if start<=a and a+n<=start+len(data):
                self.pos+=n;return data[a-start:a-start+n]
        data=self.fetch(a,n);self.pos+=len(data);return data


def main():
    assert not (OUT/'TASK_NAME_COUNTS.json').exists(),'Do not silently replace a completed count'
    tick=time.perf_counter();remote=Ranges();pf=pq.ParquetFile(remote,pre_buffer=False,buffer_size=0)
    metadata=pf.metadata;ci=pf.schema.names.index('task_name');chunks=[]
    assert metadata.num_rows==1018733
    for i in range(metadata.num_row_groups):
        c=metadata.row_group(i).column(ci)
        start=min(x for x in (c.dictionary_page_offset,c.data_page_offset) if x is not None and x>=0)
        chunks.append((start,c.total_compressed_size))
    assert sum(n for _,n in chunks)==1166665
    with ThreadPoolExecutor(max_workers=12) as pool:
        work=[pool.submit(remote.fetch,a,n) for a,n in chunks]
        for i,future in enumerate(as_completed(work),1):
            future.result()
            if i%200==0:print('TASK_NAME_COLUMN_RANGES',i,len(chunks),flush=True)
    remote.remote_allowed=False
    # PyArrow can now access only cached metadata and task_name byte ranges.
    table=pf.read(columns=['task_name'],use_threads=False)
    assert table.column_names==['task_name'] and table.num_rows==1018733
    counts=Counter(table['task_name'].to_pylist());assert None not in counts
    kept={k:v for k,v in sorted(counts.items()) if k not in EXCLUDED}
    dropped={k:counts[k] for k in sorted(EXCLUDED) if k in counts}
    result={'revision':REV,'source_url':URL,'source_file_bytes':SIZE,'file_LFS_sha256':'e4d948bb5c64f799a9fc4f908d5421c6e0560f718df15b45387b9f037bd21e25',
        'column':'task_name','rows':table.num_rows,'row_groups':metadata.num_row_groups,'unique_task_names':len(counts),
        'all_task_counts':dict(sorted(counts.items())),'tasksource_excluded_names':sorted(EXCLUDED),'excluded_counts':dropped,
        'retained_task_counts':kept,'retained_rows':sum(kept.values()),'excluded_rows':sum(dropped.values()),
        'HTTP_range_requests':remote.requests,'downloaded_bytes':remote.bytes,'task_column_compressed_bytes':sum(n for _,n in chunks),
        'range_sha256':[{'start':a,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for a,b in sorted(remote.cache,key=lambda x:x[0])],
        'seconds':time.perf_counter()-tick,'full_dataset_downloaded':False,'other_columns_materialized':False,
        'model_downloaded':False,'GPU_used':False,'local_sealed_data_read':False,
        'limits':'Literal task names are not proof of per-example provenance or absence of contamination. Counts describe this pinned data revision and current audited Tasksource filter, not a proven training-time snapshot.'}
    (OUT/'TASK_NAME_COUNTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='range_sha256'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
