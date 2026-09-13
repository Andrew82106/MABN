"""Text-only evidence overlap control and fusion with frozen Lookback/NLL."""
from pathlib import Path
from collections import Counter
import argparse,gc,pickle,re,time
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import run_development as qa

ROOT=qa.ROOT;OUT=ROOT/'results/lexical_v1';BATCH=16384
WORD=re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*",re.UNICODE)
NAMES=['lexical','log_word_length','has_digit','initial_upper','all_upper','stopword',
 'source_unigram','log_source_frequency','question_unigram','source_left_bigram','source_right_bigram',
 'source_left_trigram','source_right_trigram','source_center_fivegram','missing_numeric','missing_content']


def protocol():
    return {'version':'qa-lexical-evidence-overlap-v1','scope':'Existing fit634/cal159; official test sealed',
      'features':NAMES,'normalization':'Unicode casefold regex words; no fitted vocabulary or external language model',
      'source':'Only actual retrieved_passages; question separately; no standard answer or span used',
      'word_to_bpe':'Character intersection weighted by alphanumeric overlap; punctuation-only BPE gets zeros',
      'window':'Mean over unchanged four raw BPE, including punctuation; no labels used for feature computation',
      'ngram':'Contiguous normalized word tuples in full retrieved text; lexical overlap only, not semantic entailment',
      'methods':{'lexical_only':16,'lookback_nll_lexical':1041},'C':[.001,.01,.1],'seed':20260928,
      'fit':'Same168123 windows, frozen group/answer/class weights, weighted fit-only scaler and liblinear L2',
      'selection':'Same cal max min(windowF1,answerF1), windowF1, precision, lowerC; separate cal thresholds',
      'new_lr_fits':6,'original_output_or_gold_changed':False,'gpu_used':False,
      'interpretation':'Calibration-selected development only. Additional text features are not model-internal signals.'}


def features(question,source,answer,offsets):
    sw=[m.group().casefold() for m in WORD.finditer(source)];count=Counter(sw)
    grams={n:{tuple(sw[i:i+n]) for i in range(len(sw)-n+1)} for n in (2,3,5)}
    qw={m.group().casefold() for m in WORD.finditer(question)}
    matches=list(WORD.finditer(answer));words=[m.group().casefold() for m in matches];wordx=[]
    for j,m in enumerate(matches):
        w=words[j];raw=m.group();digit=any(c.isdigit() for c in raw);stop=w in ENGLISH_STOP_WORDS;present=w in count
        def copied(left,n):return int(left>=0 and left+n<=len(words) and tuple(words[left:left+n]) in grams[n])
        wordx.append([1,np.log1p(len(w)),digit,raw[0].isupper(),raw.isupper(),stop,present,np.log1p(count[w]),w in qw,
          copied(j-1,2),copied(j,2),copied(j-2,3),copied(j,3),copied(j-2,5),digit and not present,not stop and not present])
    result=np.zeros((len(offsets),16),np.float32)
    for i,(a,b) in enumerate(offsets):
        rows=[];weights=[]
        for j,m in enumerate(matches):
            left=max(a,m.start());right=min(b,m.end())
            if left>=right:continue
            mass=sum(c.isalnum() for c in answer[left:right])
            if mass:rows.append(wordx[j]);weights.append(mass)
        if weights:result[i]=np.average(rows,weights=weights,axis=0)
    return result


def selfcheck():
    a='Paris has 99 people.';offsets=[[0,5],[5,9],[9,12],[12,19],[19,20]]
    x=features('Which city?', 'Paris has 10 people.',a,offsets)
    assert x[0,6]==1 and x[2,14]==1 and x[2,6]==0 and np.all(x[-1]==0)
    y=features('Which city?', 'Paris has 99 people.',a,offsets)
    assert y[2,14]==0 and y[2,6]==1
    assert x.shape==(5,16) and np.isfinite(x).all()
    z=features('', 'Paris has 99 people.',a,[[10,11],[11,12]])
    assert np.array_equal(z[0],z[1])


def run():
    assert not (OUT/'started.json').exists() and qa.read(OUT/'protocol.json')==protocol()
    selfcheck();meta=qa.metadata();start=time.perf_counter()
    complete=qa.read(qa.OUT/'complete.json')
    for n,h in complete['files_sha256'].items():assert qa.sha(qa.OUT/n)==h
    qa.save(OUT/'started.json',{'code_sha256':qa.sha(Path(__file__)),'protocol_sha256':qa.sha(OUT/'protocol.json'),
      'gold_sha256':qa.sha(ROOT/'data/gold_manifest.json'),'base_complete_sha256':qa.sha(qa.OUT/'complete.json')})
    rows={r['response_id']:r for part in ('fit','calibration') for r in qa.lines(ROOT/'data'/f'{part}.jsonl')}
    matrix=np.empty((210364,16),np.float32);tokens={};coverage=0
    for a in meta['answers']:
        rid=a['response_id'];r=rows[rid];t=meta['by_response'][rid]['tokens']
        x=features(r['question'],r['retrieved_passages'],r['original_response'],t['response_token_offsets'])
        assert len(x)==t['token_count'] and np.array_equal(x[:,0]>0,np.asarray(t['lexical_mask'],bool))
        tokens[rid]=x;coverage+=len(x)
        for j in meta['answer_windows'][rid]:matrix[j]=x[meta['windows'][j]['token_indices']].mean(0)
    assert coverage==213159
    np.save(OUT/'window_features.npy',matrix);np.savez_compressed(OUT/'token_features.npz',**tokens)
    base=np.load(qa.OUT/'matrices/base.npy',mmap_mode='r')
    with np.load(qa.OUT/'training_weights.npz',allow_pickle=False) as z:b=z['base_weights'];loss=z['loss_weights'];y=z['y']
    assert np.array_equal(y,[w['label'] for w in meta['windows'][:168123]])
    selected={};candidates={};files=[]
    for method,width in protocol()['methods'].items():
        def raw(l,r):return matrix[l:r] if method=='lexical_only' else np.column_stack((base[l:r],matrix[l:r]))
        scaler=StandardScaler()
        for l in range(0,168123,BATCH):r=min(l+BATCH,168123);scaler.partial_fit(raw(l,r),sample_weight=b[l:r])
        train=np.lib.format.open_memmap(OUT/(method+'_fit.npy'),mode='w+',dtype=np.float32,shape=(168123,width))
        for l in range(0,168123,BATCH):r=min(l+BATCH,168123);train[l:r]=scaler.transform(raw(l,r)).astype(np.float32)
        train.flush();table=[]
        for c in protocol()['C']:
            t0=time.perf_counter();model=LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=protocol()['seed'])
            model.fit(train,y,sample_weight=loss);assert model.n_iter_.max()<2000
            score=np.empty(210364,np.float64)
            for l in range(0,210364,BATCH):r=min(l+BATCH,210364);score[l:r]=model.predict_proba(scaler.transform(raw(l,r)).astype(np.float32))[:,1]
            ans=qa.answer_scores(meta,score)
            ts={'window':qa.choose_threshold([w['label'] for w in meta['windows'][168123:]],score[168123:]),
                'answer':qa.choose_threshold([a['label'] for a in meta['answers'][634:]],ans[634:])}
            name=f'{method}_C{c:g}';key=list(qa.selection_key(ts,c))
            obj={'model':model,'scaler':scaler,'C':c,'method':method,'thresholds':ts,'fit_only':True}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=score,answer_scores=ans)
            entry={'candidate':name,'C':c,'thresholds':ts,'selection_key':key,'metrics':qa.metrics(meta,score,ts),'seconds':time.perf_counter()-t0}
            qa.save(OUT/(name+'_result.json'),entry);table.append(entry);files.extend([name+'.pkl',name+'_scores.npz',name+'_result.json'])
            print('QA_LEXICAL',name,entry['metrics']['calibration']['windows']['f1'],entry['metrics']['calibration']['answers']['f1'],flush=True)
        selected[method]=max(table,key=lambda x:x['selection_key']);candidates[method]=table;del train;gc.collect()
    qa.save(OUT/'summary.json',{'selected':selected,'all_candidates':candidates,'seconds':time.perf_counter()-start,
      'feature_tokens':coverage,'test_opened':False,'calibration_results_are_selection_optimistic':True})
    files+=['protocol.json','started.json','summary.json','window_features.npy','token_features.npz']
    qa.save(OUT/'complete.json',{'status':'complete','files_sha256':{n:qa.sha(OUT/n) for n in files},'test_opened':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('initialize','run'));a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':
            assert not (OUT/'protocol.json').exists();selfcheck();qa.save(OUT/'protocol.json',protocol())
        else:run()
