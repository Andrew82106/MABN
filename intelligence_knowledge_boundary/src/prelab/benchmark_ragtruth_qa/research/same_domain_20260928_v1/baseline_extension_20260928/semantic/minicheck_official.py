"""Pinned official MiniCheck-Flan-T5-Large, native sentence checking.

Original Inferencer methods provide document splitting, 500-word chunks, prompt,
2048-token limit, decoder IDs [3,209], and max support over chunks. Only model
loading is replaced to resolve the pinned local snapshot and choose the device.
"""
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch,nltk
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import common as c
sys.path.insert(0,str(c.HERE/'official/MiniCheck'))
from minicheck.inference import Inferencer,sent_tokenize_with_newlines

OUT=c.HERE/'minicheck_run'
MODEL=c.HERE/'resources/MiniCheck-Flan-T5-Large'
NLTK=c.PROJECT/'artifacts/dependencies/benchmark_ragtruth_qa/semantic_baseline/nltk_data'
nltk.data.path.insert(0,str(NLTK))

def sentences(text):
    # Official source splits at newlines before NLTK; retain real source offsets.
    tokenizer=nltk.tokenize.PunktTokenizer('english');out=[];offset=0
    for block in text.split('\n'):
        for a,b in tokenizer.span_tokenize(block):
            if block[a:b].strip():out.append({'start':offset+a,'end':offset+b,'text':block[a:b]})
        offset+=len(block)+1
    assert [x['text'] for x in out]==[s for s in sent_tokenize_with_newlines(text) if s!='\n' and s.strip()]
    return out

def make_inferencer(device='cpu',load_weights=True):
    obj=Inferencer.__new__(Inferencer)
    obj.model_name='flan-t5-large';obj.max_model_len=2048;obj.max_output_length=256;obj.batch_size=1;obj.chunk_size=500
    obj.softmax=torch.nn.Softmax(dim=-1)
    obj.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    if load_weights:
        # Original checkpoint is FP32 and official loader has no dtype override.
        obj.model=AutoModelForSeq2SeqLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).to(device).eval()
    return obj

def prepare():
    assert not torch.cuda.is_initialized();OUT.mkdir(parents=True,exist_ok=True)
    official=c.read(c.HERE/'SOURCE_MANIFEST.json')
    for name,meta in official.items():
        if name.startswith('official/MiniCheck/') or name.startswith('official\\MiniCheck\\'):
            assert c.sha(c.HERE/name)==meta['sha256']
    summaries={};bindings={}
    checker=make_inferencer(load_weights=False)
    for split in ['validation','test']:
        aa,gg=c.inputs(split);rr=[];max_native=0;total_sentences=0;truncated=0
        for a in aa:
            rid=a['response_id'];ss=sentences(a['original_response']);total_sentences+=len(ss)
            assert ss
            # Use actual unmodified upstream document splitter, but intercept its
            # last inference call for CPU preparation. No neural evaluation here.
            captured=[]
            checker.inference=lambda docs,claims:captured.append((docs,claims))
            checker.inference_per_example(a['retrieved_passages'],ss[0]['text'])
            chunks=captured[0][0];assert chunks
            for s in ss:
                for chunk in chunks:
                    n=len(checker.tokenizer('predict: '+chunk+checker.tokenizer.eos_token+s['text'],truncation=False)['input_ids'])
                    max_native=max(max_native,n);truncated+=int(n>2048)
            rr.append({'response_id':rid,'documents':a['retrieved_passages'],'question_not_used_by_native_method':a['question'],
              'sentences':ss,'source_chunks':chunks,'original_token_count':len(gg[rid]['lexical_mask']),
              'mapping':c.mapping(a['original_response'],gg[rid]['response_token_offsets'],[(s['start'],s['end']) for s in ss]),
              'response_sha256':hashlib.sha256(a['original_response'].encode()).hexdigest()})
        c.write_rows(OUT/f'{split}_prepared.jsonl',rr)
        summaries[split]={'answers':len(rr),'sentences':total_sentences,'max_pair_tokens_before_official_truncation':max_native,'pairs_truncated_at_2048':truncated}
        bindings.update(c.source_hashes(split))
    c.save(OUT/'PREPARED.json',{'summaries':summaries,'source_input_sha256':bindings,'model_revision':c.read(c.HERE/'MODEL_REVISION.json'),
      'source_manifest_sha256':c.sha(c.HERE/'SOURCE_MANIFEST.json'),'test_labels_read':False,
      'protocol':{'training':'none; official frozen model trained on ANLI+synthetic data; validation thresholds only',
       'model':'lytang/MiniCheck-Flan-T5-Large','code_revision':'b58b9fa69acbd1015ec970fa65dd752413a053d2',
       'input':'official document+sentence; original question retained in receipt but not supplied as an assertion or evidence',
       'sentence_split':'official newline splitting then English NLTK Punkt, exact offsets retained',
       'chunks':'official 500 whitespace words grouped by sentence, max2048 native tokens, max support over chunks',
       'risk':'1-support; sentence score mapped via nonwhitespace character overlap to original Llama token',
       'precision':'FP32 as official default; model-only local snapshot loader, no structure/weight modification',
       'batch_size':1,'native_granularity':'sentence, not token','resume':'one immutable prediction file per response'}})
    print('MINICHECK_PREPARED',summaries,flush=True)

def cpu_test():
    prepare();assert not torch.cuda.is_initialized()
    for sp in ['validation','test']:
        rr=c.rows(OUT/f'{sp}_prepared.jsonl');_,gg=c.inputs(sp)
        for r in rr:
            score=c.mapped(np.ones(len(r['sentences'])),r['mapping'],r['original_token_count'])
            lex=gg[r['response_id']]['lexical_mask']
            assert all(abs(v-1)<1e-7 for v,x in zip(score,lex) if x)
    # A split sentence's risk remains distinct at the output coordinate layer.
    text='Paris is French. Rome is Italian.';ss=sentences(text)
    mp=c.mapping(text,[[0,5],[17,21]],[(s['start'],s['end']) for s in ss]);assert np.allclose(c.mapped([.2,.8],mp,2),[.2,.8])
    c.save(OUT/'CPU_CHECK.json',{'passed':True,'answer_count':550,'native_sentence_split_equivalence':True,'all_lexical_coverage':True,'GPU_used':False,'test_labels_read':False})
    print('MINICHECK_CPU_CHECK_OK',flush=True)

def one(checker,r):
    native=[];risk=[]
    for s in r['sentences']:
        result=checker.inference_per_example(r['documents'],s['text'])
        p=float(result['max_support_prob']);assert np.isfinite(p) and 0<=p<=1
        risk.append(1-p)
        native.append({**s,'support_probability':p,'risk_probability':1-p,
          'support_probability_per_chunk':np.asarray(result['support_prob_per_chunk']).tolist(),'used_chunks':result['used_chunks']})
    return {'response_id':r['response_id'],'token_scores':c.mapped(risk,r['mapping'],r['original_token_count']),
      'native_sentence_scores':native,'response_sha256':r['response_sha256']}

def smoke():
    assert c.read(OUT/'CPU_CHECK.json')['passed'];assert (c.HERE/'MODEL_MANIFEST.json').exists()
    torch.set_num_threads(4);checker=make_inferencer('cuda');r=max(c.rows(OUT/'validation_prepared.jsonl'),key=lambda r:len(r['documents'])+max(len(s['text']) for s in r['sentences']))
    start=time.time();pred=one(checker,r);torch.cuda.synchronize()
    assert len(pred['token_scores'])==r['original_token_count']
    c.save(OUT/'GPU_SMOKE.json',{'passed':True,'response_id':r['response_id'],'sentences':len(r['sentences']),
      'elapsed_seconds':time.time()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'test_labels_read':False})
    c.save(OUT/'GPU_SMOKE_PREDICTION.json',pred);print('MINICHECK_GPU_SMOKE_OK',flush=True)

def predict(split):
    assert c.read(OUT/'GPU_SMOKE.json')['passed'];torch.set_num_threads(4)
    model_binding=c.read(c.HERE/'MODEL_MANIFEST.json')
    for name,hashval in model_binding['files_sha256'].items():assert c.sha(MODEL/name)==hashval
    checkpoint=c.sha(c.HERE/'MODEL_MANIFEST.json');checker=make_inferencer('cuda');rr=c.rows(OUT/f'{split}_prepared.jsonl')
    dest=OUT/f'{split}_cache';dest.mkdir(exist_ok=True);predictions=[];start=time.time()
    for i,r in enumerate(rr):
        path=dest/(r['response_id']+'.json')
        binding=hashlib.sha256(__import__('json').dumps(r,sort_keys=True).encode()).hexdigest()
        if path.exists():
            saved=c.read(path);assert saved['model_manifest_sha256']==checkpoint and saved['input_sha256']==binding
            pred=saved['prediction']
        else:
            pred=one(checker,r);c.save(path,{'prediction':pred,'input_sha256':binding,'model_manifest_sha256':checkpoint})
        predictions.append(pred)
        if (i+1)%25==0:
            print('MINICHECK',split,i+1,len(rr),flush=True)
            c.save(OUT/'STATUS.json',{'stage':'predict','split':split,'answers':i+1,'total':len(rr),'elapsed_seconds':time.time()-start})
    path=OUT/f'{split}_predictions.jsonl';c.write_rows(path,predictions)
    if split=='validation':c.save(OUT/'VALIDATION.json',c.val_reports(predictions))
    c.save(OUT/f'{split}_PREDICTION_RECEIPT.json',{'method':'MiniCheck-Flan-T5-Large-official','split':split,'answers':len(rr),
      'model_manifest_sha256':checkpoint,'input_sha256':c.sha(OUT/f'{split}_prepared.jsonl'),'predictions_sha256':c.sha(path),
      'source_input_sha256':c.source_hashes(split),'script_sha256':c.sha(__file__),'official_source_manifest_sha256':c.sha(c.HERE/'SOURCE_MANIFEST.json'),
      'native_granularity':'sentence','elapsed_seconds':time.time()-start,'test_labels_read':False})
    print('MINICHECK_PREDICT_DONE',split,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu-test','smoke','predict']);p.add_argument('--split',choices=['validation','test'],default='validation');a=p.parse_args()
    {'prepare':prepare,'cpu-test':cpu_test,'smoke':smoke,'predict':lambda:predict(a.split)}[a.stage]()
if __name__=='__main__':main()
