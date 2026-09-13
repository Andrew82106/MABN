"""Train local evidence-check probes; freeze all fold choices before evaluation."""
from pathlib import Path
import argparse,importlib.util,pickle,time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r23_reuse24',ROOT.parent/'round24_sequence_persistence/src/run24.py')
r24=importlib.util.module_from_spec(spec);spec.loader.exec_module(r24)
r22=r24.r22;r21=r22.r21;r18,r17,r10=r22.r18,r22.r17,r22.r10
read,readl,save,savel,sha=r22.read,r22.readl,r22.save,r22.savel,r22.sha
BASES=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned','verifier_learned_broadcast','verifier_learned_slots','slots_base_smooth')
TRAINED=('local_probe','local_mean_fusion','local_slots_fusion')
NEW=('local_direct',)+TRAINED+('local_slots_fusion_smooth',)
METHODS=BASES+NEW;r18.METHODS=METHODS

def protocol():
    return {'version':'r23b-full-single-verification-probes-v2','full_forward_batch_size':1,'cache_or_padding_used':False,'scope':'Exploratory R16 actual-train fixed 5 event folds, no original heldout',
      'methods':list(METHODS),'primary_method':'local_slots_fusion','C':[.001,.01,.1],'seed':20260923,'loss_mass':3854,
      'pca':{'components':64,'n_iter':3,'whiten':False,'seed':'20260923+fold','fit_only':'all eligible fit windows, event/condition/answer/window base weighting'},
      'features':{'local_direct':'A/B restricted softmax B (untrained verdict baseline)',
         'local_probe':'fit-only PCA64 focused final hidden + (B-A) logit + full-vocabulary A+B mass',
         'local_mean_fusion':'66 focused features + original mean LB784/NLL1',
         'local_slots_fusion':'66 focused features + original ordered slots3144'},
      'lr':'Same R20 class/event weights and fit-only weighted scaler, liblinear L2 max_iter2000',
      'C_selection':'calibration maximize min(window F1,answer F1), then window F1, precision, lower C',
      'threshold':'separate calibration F1/precision/higher threshold; all-window-max answer score; gold unchanged',
      'additional_smoother':'After raw local_slots_fusion C frozen, apply same R24 stay/temperature/cal selection budget; exact identity candidate included',
      'stay':[.5,.8,.95,.99],'temperature':[.5,1.,2.],'new_lr_fits':45,'new_pca_fits':5,
      'extra_offline_focused_check':True,'original_answer_modified':False,'original_generation_confidence':False,
      'four_raw_bpe_window_definition_unchanged':True,'all_fold_choices_frozen_before_outer':True,
      'original_validation_or_test_used':False,'human_gold':False}

def prepare():
    meta=r18.train_metadata();pack=r17.cohort(r18.NEW,'train',meta,r18.Bank(meta[2]))
    assert pack['windows']==readl(r22.ROOT/'results/candidate_windows22.jsonl')
    sig=read(ROOT/'data/signature.json');fm=read(ROOT/'data/feature_manifest.json');plans=read(ROOT/'data/plans.json')
    assert fm['complete'] and fm['windows_completed']==12222 and fm['completed_count']==602
    assert sig['code_sha256']==sha(ROOT/'src/extract23.py')
    lookup={};rows={};files={}
    for rid,entry in fm['records'].items():
        p=ROOT/entry['npz'];side=ROOT/entry['json'];m=read(side)
        assert sha(p)==entry['npz_sha256']==m['npz_sha256'] and sha(side)==entry['json_sha256']
        assert m['source_generation_sha256']==meta[2][rid][1]
        assert m['signature_sha256']==fm['signature_sha256'] and m['plan_sha256']==r10.digest(plans[rid])
        assert m['labels_used'] is False and m['output_regenerated'] is False
        with np.load(p,allow_pickle=False) as z:rows[rid]={k:z[k].copy() for k in z.files}
        a=rows[rid];assert a['verifier_hidden'].shape==(m['windows'],3584)
        for j,w in enumerate(plans[rid]['windows']):
            assert w['window_key']==m['window_keys'][j]
            assert w['start']==a['window_start'][j] and w['end']==a['window_end'][j]
            assert w['window_key'] not in lookup;lookup[w['window_key']]=(rid,j)
        files[str(p.resolve())]=sha(p);files[str(side.resolve())]=sha(side)
    hh=[];ee=[];dd=[]
    for w in pack['windows']:
        rid,j=lookup[w['window_key']];a=rows[rid];assert rid==w['row_id']
        hh.append(a['verifier_hidden'][j]);ee.append([float(a['ab_logits'][j,1])-float(a['ab_logits'][j,0]),float(a['ab_full_vocab_probabilities'][j].sum())]);dd.append(a['ab_probabilities'][j,1])
    with np.load(r21.ROOT/'results/designs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ['base','slots']}
    d.update(hidden=np.asarray(hh,np.float32),extra=np.asarray(ee,np.float32),direct=np.asarray(dd,np.float64))
    assert all(np.isfinite(v).all() for v in d.values()) and d['hidden'].shape==(12222,3584)
    return pack,d,files

def projection(h,ix,windows,fold):
    rr=[windows[i] for i in ix];b=r10.base_weights(rr,'token');w=b.astype(np.float64,copy=True);w/=w.sum()
    raw=h[ix].astype(np.float64);center=w@raw
    _,sv,c=randomized_svd((raw-center)*np.sqrt(w[:,None]),n_components=64,n_iter=3,random_state=20260923+fold,flip_sign=True)
    z=((h.astype(np.float64)-center)@c.T).astype(np.float32)
    obj={'mean':center,'components':c,'singular_values':sv,'fit_ix':ix,'base_weights':b,'pca_weights':w,
         'fit_keys':[x['window_key'] for x in rr],'fit_groups':sorted({x['group_id'] for x in rr}),'seed':20260923+fold,'whiten':False}
    return obj,z

def fit():
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True);assert not (out/'fit_started.json').exists()
    cfg=read(ROOT/'protocol.json');assert cfg==protocol();pack,d,files=prepare();seqs=r24.geometry(pack['windows'])
    snap={'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json'),'feature_manifest_sha256':sha(ROOT/'data/feature_manifest.json'),
         'r22_complete_sha256':sha(r22.ROOT/'results/complete22.json'),'r24_complete_sha256':sha(r24.ROOT/'results/complete.json')}
    for folder in [r22.ROOT,r24.ROOT]:
        c=read(folder/'results'/('complete22.json' if folder==r22.ROOT else 'complete.json'))
        for n,h in c['files_sha256'].items():assert sha(folder/'results'/n)==h
    save(out/'fit_started.json',{'utc':r10.utc(),'snapshot':snap});save(out/'feature_files.json',files)
    np.savez_compressed(out/'designs.npz',**d);savel(out/'candidate_windows.jsonl',pack['windows'])
    gid=np.asarray([w['group_id'] for w in pack['windows']]);ok=np.asarray([w['main_eligible'] for w in pack['windows']]);start=time.perf_counter();names=[]
    for fold in range(5):
        old=pickle.loads((r22.ROOT/'results'/f'fold_{fold}_frozen22.pkl').read_bytes());cal,cx=r18.subset(pack,old['calibration_groups'])
        ix=np.flatnonzero(np.isin(gid,old['fit_groups'])&ok);pc,z=projection(d['hidden'],ix,pack['windows'],fold)
        local=np.column_stack((z,d['extra'])).astype(np.float32)
        designs={'local_probe':local,'local_mean_fusion':np.column_stack((d['base'],local)), 'local_slots_fusion':np.column_stack((d['slots'],local))}
        values={};ts={};models={}
        with np.load(r22.ROOT/'results'/f'fold_{fold}_scores22.npz') as prior:
            for m in BASES[:-1]:values[m]=prior[m].copy();ts[m]=old['thresholds'][m]
        sm=read(r24.ROOT/'results'/f'fold_{fold}_calibration.json')
        with np.load(r24.ROOT/'results'/f'fold_{fold}_scores.npz') as prior:values['slots_base_smooth']=prior['slots_base_smooth'].copy()
        ts['slots_base_smooth']=sm['thresholds']['slots_base_smooth']
        values['local_direct']=d['direct'];ts['local_direct']=r22.thresholds(cal,d['direct'][cx])
        for m in TRAINED:
            models[m],values[m],ts[m]=r21.fit_lr_grid(designs[m],pack,ix,cal,cx,cfg)
            print('R23_FIT',fold,m,'C',models[m]['C'],flush=True)
        table=[];smoothvalues=[]
        for stay in cfg['stay']:
            for temp in cfg['temperature']:
                v=r24.whole(values['local_slots_fusion'],seqs,stay,temp);t=r22.thresholds(cal,v[cx]);w,a=t['window'],t['answer']
                key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],int(stay==.5 and temp==1),-stay,-abs(np.log(temp))]
                table.append({'stay':stay,'temperature':temp,'thresholds':t,'key':key});smoothvalues.append(v)
        j=max(range(len(table)),key=lambda j:table[j]['key']);values['local_slots_fusion_smooth']=smoothvalues[j];ts['local_slots_fusion_smooth']=table[j]['thresholds']
        obj={'groups':{k:old[k] for k in ['fit_groups','calibration_groups','evaluation_groups']},'projection':pc,'models':models,'thresholds':ts,
             'smoothing_candidates':table,'smoothing_selected':j}
        (out/f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(out/f'fold_{fold}_scores.npz',**values,projected_hidden=z)
        names.extend([f'fold_{fold}_frozen.pkl',f'fold_{fold}_scores.npz'])
    names+=['feature_files.json','designs.npz','candidate_windows.jsonl']
    save(out/'fit_freeze.json',{'utc':r10.utc(),'seconds':time.perf_counter()-start,'snapshot':snap,'new_lr_fits':45,'files_sha256':{n:sha(out/n) for n in names}})

def test():
    out=ROOT/'results';frozen=read(out/'fit_freeze.json');assert not (out/'test_started.json').exists()
    for n,h in frozen['files_sha256'].items():assert sha(out/n)==h
    assert frozen['snapshot']['code_sha256']==sha(Path(__file__))
    pack,_,_=prepare();save(out/'test_started.json',{'utc':r10.utc(),'freeze_sha256':sha(out/'fit_freeze.json')});allw=[];alla=[];details={}
    for fold in range(5):
        f=pickle.loads((out/f'fold_{fold}_frozen.pkl').read_bytes());ev,ex=r18.subset(pack,f['groups']['evaluation_groups']);fit,ix=r18.subset(pack,f['groups']['fit_groups']);cal,cx=r18.subset(pack,f['groups']['calibration_groups'])
        with np.load(out/f'fold_{fold}_scores.npz') as z:values={m:z[m].copy() for m in METHODS}
        wr=None;ar=None;detail={}
        for m in METHODS:
            w,a=r18.scored_records(ev,values[m][ex],f['thresholds'][m],m)
            if wr is None:wr=[dict(x,scores={},predictions={},fold=fold) for x in w];ar=[dict(x,scores={},predictions={},fold=fold) for x in a]
            for x,y in zip(wr,w):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
            for x,y in zip(ar,a):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
            detail[m]={stage:r17.metrics(pp,values[m][jj],f['thresholds'][m]) for stage,pp,jj in [('fit',fit,ix),('calibration',cal,cx),('evaluation',ev,ex)]}
        allw.extend(wr);alla.extend(ar);details[str(fold)]=detail
    result=r18.pooled(allw,alla,pack);old=read(r22.ROOT/'results/summary22.json')['methods']
    for m in BASES[:-1]:assert result[m]==old[m]
    assert result['slots_base_smooth']==read(r24.ROOT/'results/summary.json')['methods']['slots_base_smooth']
    contrasts={m+'_vs_slots_smooth':[m,'slots_base_smooth'] for m in NEW}
    summary={'methods':result,'folds':details,'scope':protocol()['scope'],'primary_method':protocol()['primary_method'],
             'paired_bootstrap':{u:r18.bootstrap(rows,{'draws':2000,'seed':20260923},contrasts) for u,rows in [('windows',allw),('answers',alla)]},
             'all_old_baselines_exact':True,'extra_offline_check':True,'original_output_or_gold_changed':False}
    savel(out/'window_scores_oof.jsonl',allw);savel(out/'answer_scores_oof.jsonl',alla);save(out/'summary.json',summary)
    save(out/'complete.json',{'utc':r10.utc(),'files_sha256':{n:sha(out/n) for n in ['fit_freeze.json','summary.json','window_scores_oof.jsonl','answer_scores_oof.jsonl']}})
    for m,v in result.items():print(m,round(v['windows']['f1'],6),round(v['answers']['f1'],6),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','prepare','fit','test']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':assert not (ROOT/'protocol.json').exists();save(ROOT/'protocol.json',protocol());print('R23_TRAINING_PROTOCOL_FROZEN')
        elif a.stage=='prepare':pp,d,_=prepare();print(pp['coverage'],{k:v.shape for k,v in d.items()})
        elif a.stage=='fit':fit()
        else:test()
