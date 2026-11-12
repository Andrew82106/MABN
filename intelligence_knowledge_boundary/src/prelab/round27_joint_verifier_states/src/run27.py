"""Joint fit-only probes on generation, global-check and local-check states."""
from pathlib import Path
import argparse,importlib.util,pickle,time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r27_r26',ROOT.parent/'round26_global_local_fusion/src/run26.py')
r26=importlib.util.module_from_spec(spec);spec.loader.exec_module(r26)
r23=r26.r23;r24=r23.r24;r22=r23.r22;r21=r23.r21;r18,r17,r10=r23.r18,r23.r17,r23.r10
read,save,sha,savel=r23.read,r23.save,r23.sha,r23.savel
BASES=r26.METHODS;TRAINED=('joint_mean','joint_slots')
NEW=TRAINED+tuple(m+'_smooth' for m in TRAINED);METHODS=BASES+NEW;r18.METHODS=METHODS


def protocol():
    return {'version':'r27-joint-check-states-v1','scope':'Repeated R16 actual-train event5fold development',
      'methods':list(METHODS),'primary':'joint_mean_smooth','C':[.001,.01,.1],'seed':20260927,'loss_mass':3854,
      'features':{'joint_mean':'OriginalLB784+NLL1, local fitPCA64+ABlogitgap1, global fitPCA64+ABlogitgap1 =915',
        'joint_slots':'Original ordered3144 slots +same local65+global65 =3274'},
      'exclude':'A+B full-vocabulary mass omitted from both check streams because almost constant; no old features changed',
      'projections':'Reuse each corresponding fold R23b local/R22 global PCA; fit group subsets checked, no new PCA',
      'fit':'Same fit windows/labels/group-balanced loss and fit-only scaler; 30new LR fits',
      'C_selection':'Cal max min(windowF1,answerF1), windowF1, precision, lowerC',
      'smooth':'Same12 R24 candidates per new family after C fixed; identity included',
      'stay':[.5,.8,.95,.99],'temperature':[.5,1.,2.],
      'answer':'Max all original candidate windows including safe refusals; separate cal cutoff',
      'all_five_fold_choices_frozen_before_outer':True,'new_extraction':False,'new_labels':False,
      'human_gold':False,'original_validation_or_test_used':False,'future_windows_used_by_smoothing':True}


def prepare():
    pack=r26.prepare()
    for folder,name in [(r26.ROOT,'complete.json'),(r22.ROOT,'complete22.json')]:
        complete=read(folder/'results'/name)
        for n,h in complete['files_sha256'].items():assert sha(folder/'results'/n)==h
    assert pack['items']==r23.readl(r22.ROOT/'results/answer_index22.jsonl')
    with np.load(r23.ROOT/'results/designs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('base','slots','extra')}
    with np.load(r22.ROOT/'results/verifier_designs22.npz',allow_pickle=False) as z:ai=z['window_answer_index'].copy()
    assert all(pack['items'][j]['item_id']==w['item_ids'][0] for w,j in zip(pack['windows'],ai))
    return pack,d,ai


def fit():
    out=ROOT/'results';assert not (out/'started.json').exists() and read(ROOT/'protocol.json')==protocol()
    pack,d,ai=prepare();seqs=r24.geometry(pack['windows']);start=time.perf_counter()
    snapshot={'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json'),
      'r26_complete_sha256':sha(r26.ROOT/'results/complete.json'),'r23_fit_sha256':sha(r23.ROOT/'results/fit_freeze.json'),
      'r22_design_sha256':sha(r22.ROOT/'results/verifier_designs22.npz')}
    save(out/'started.json',snapshot);files=[]
    gid=np.asarray([w['group_id'] for w in pack['windows']]);ok=np.asarray([w['main_eligible'] for w in pack['windows']])
    for fold in range(5):
        old=pickle.loads((r23.ROOT/'results'/f'fold_{fold}_frozen.pkl').read_bytes())
        global_old=pickle.loads((r22.ROOT/'results'/f'fold_{fold}_frozen22.pkl').read_bytes())
        groups=old['groups'];assert all(groups[k]==global_old[k] for k in groups)
        assert set(old['projection']['fit_groups'])<=set(groups['fit_groups'])
        assert set(global_old['projection']['fit_groups'])<=set(groups['fit_groups'])
        cal,cx=r18.subset(pack,groups['calibration_groups']);ix=np.flatnonzero(np.isin(gid,groups['fit_groups'])&ok)
        with np.load(r23.ROOT/'results'/f'fold_{fold}_scores.npz',allow_pickle=False) as z:local=np.column_stack((z['projected_hidden'],d['extra'][:,0]))
        with np.load(r22.ROOT/'results'/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:global_state=z['probe_design'][ai,:65].copy()
        with np.load(r26.ROOT/'results'/f'fold_{fold}_scores.npz',allow_pickle=False) as z:values={m:z[m].copy() for m in BASES}
        ts=read(r26.ROOT/'results'/f'fold_{fold}_calibration.json')['thresholds'];models={};tables={};choices={}
        designs={m:np.column_stack((d[k],local,global_state)).astype(np.float32) for m,k in [('joint_mean','base'),('joint_slots','slots')]}
        assert designs['joint_mean'].shape==(12222,915) and designs['joint_slots'].shape==(12222,3274)
        for m in TRAINED:
            models[m],values[m],ts[m]=r21.fit_lr_grid(designs[m],pack,ix,cal,cx,protocol())
            table=[];scores=[]
            for stay in protocol()['stay']:
                for temp in protocol()['temperature']:
                    v=r24.whole(values[m],seqs,stay,temp);t=r22.thresholds(cal,v[cx]);w,a=t['window'],t['answer']
                    key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],int(stay==.5 and temp==1),-stay,-abs(np.log(temp))]
                    table.append({'stay':stay,'temperature':temp,'thresholds':t,'key':key});scores.append(v)
            j=max(range(len(table)),key=lambda j:table[j]['key']);name=m+'_smooth';values[name]=scores[j];ts[name]=table[j]['thresholds'];tables[name]=table;choices[name]=table[j]
            print('R27_FIT',fold,m,'C',models[m]['C'],flush=True)
        obj={'groups':groups,'models':models,'thresholds':ts,'smoothing_tables':tables,'smoothing_choices':choices,
          'local_projection_source_sha256':sha(r23.ROOT/'results'/f'fold_{fold}_frozen.pkl'),
          'global_projection_source_sha256':sha(r22.ROOT/'results'/f'fold_{fold}_frozen22.pkl')}
        (out/f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(out/f'fold_{fold}_scores.npz',**values,local65=local,global65=global_state)
        files.extend([f'fold_{fold}_frozen.pkl',f'fold_{fold}_scores.npz'])
    save(out/'fit_freeze.json',{'seconds':time.perf_counter()-start,'snapshot':snapshot,'files_sha256':{n:sha(out/n) for n in files}})


def test():
    out=ROOT/'results';frozen=read(out/'fit_freeze.json');assert not (out/'test_started.json').exists()
    for n,h in frozen['files_sha256'].items():assert sha(out/n)==h
    assert sha(Path(__file__))==frozen['snapshot']['code_sha256']
    pack,_,_=prepare();save(out/'test_started.json',{'freeze_sha256':sha(out/'fit_freeze.json')});allw=[];alla=[]
    for fold in range(5):
        f=pickle.loads((out/f'fold_{fold}_frozen.pkl').read_bytes());ev,ix=r18.subset(pack,f['groups']['evaluation_groups'])
        with np.load(out/f'fold_{fold}_scores.npz',allow_pickle=False) as z:values={m:z[m].copy() for m in METHODS}
        wr=ar=None
        for m in METHODS:
            w,a=r18.scored_records(ev,values[m][ix],f['thresholds'][m],m)
            if wr is None:wr=[dict(x,scores={},predictions={},fold=fold) for x in w];ar=[dict(x,scores={},predictions={},fold=fold) for x in a]
            for x,y in zip(wr,w):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
            for x,y in zip(ar,a):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
        allw.extend(wr);alla.extend(ar)
    result=r18.pooled(allw,alla,pack);old=read(r26.ROOT/'results/summary.json')['methods']
    for m in BASES:assert result[m]==old[m]
    contrast={m+'_vs_r26':[m,'local_slots_fusion_smooth_global'] for m in NEW}
    save(out/'summary.json',{'methods':result,'old_baselines_exact':True,'scope':protocol()['scope'],
      'paired_bootstrap':{unit:r18.bootstrap(rows,{'draws':2000,'seed':20260927},contrast) for unit,rows in [('windows',allw),('answers',alla)]}})
    savel(out/'window_scores_oof.jsonl',allw);savel(out/'answer_scores_oof.jsonl',alla)
    save(out/'complete.json',{'files_sha256':{n:sha(out/n) for n in ['fit_freeze.json','summary.json','window_scores_oof.jsonl','answer_scores_oof.jsonl']}})
    for m in NEW:print(m,result[m]['windows']['f1'],result[m]['answers']['f1'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('initialize','fit','test'));a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':assert not (ROOT/'protocol.json').exists();save(ROOT/'protocol.json',protocol())
        elif a.stage=='fit':fit()
        else:test()
