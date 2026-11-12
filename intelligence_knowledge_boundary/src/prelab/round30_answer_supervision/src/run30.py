"""Matched window-only / window+answer supervision. CPU; prepare precedes fit."""
from pathlib import Path
from collections import defaultdict
import argparse
import importlib.util
import pickle
import shutil
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'round29_full_hidden_scene_adaptation/src'))
import run29 as previous
q, cache, core = previous.q, previous.cache, previous.core
OUT, OLD = ROOT / 'results', previous.OUT
SCENE, R26 = previous.SCENE, previous.R26
SEED, EPOCHS, GROUP_BATCH = previous.SEED, 10, 8
MODES = ('window_only', 'window_plus_answer')
BASES, ALPHA = previous.BASES, previous.ALPHA
FUSIONS = {mode: (f'lookback_plus_{mode}', f'r26_plus_{mode}') for mode in MODES}
METHODS = BASES + MODES + tuple(v for pair in FUSIONS.values() for v in pair)
metadata, thresholds, answer_scores = previous.metadata, previous.thresholds, previous.answer_scores


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(769, 32), nn.SiLU(), nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def protocol():
    return {'version': 'r30-fixed-answer-supervision-v1', 'scope': previous.protocol()['scope'],
        'representation': 'Exact R29 full769 raw-window representation. Reuse each frozen R29 fit-only StandardScaler; no refit of LR/PCA/encoder and no new input feature.',
        'model': {'architecture': '769->32 SiLU->dropout0.1->1', 'parameters': 24673,
            'seed': SEED, 'initialization_seed': 'seed+fold, identical initial state and post-init CPU RNG for both modes',
            'epochs': EPOCHS, 'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .01,
            'gradient_clipping': None, 'scheduler': None, 'CPU_threads': 4, 'dtype': 'float32'},
        'modes': {'window_only': {'lambda_answer': 0.}, 'window_plus_answer': {'lambda_answer': 1.}},
        'windows': 'Only unchanged main_eligible fit windows enter window BCE; reuse exact R29 loss weights, global mass3854. Ineligible labels remain-1 and never enter BCE.',
        'answer_targets': 'Only unchanged main_eligible (resolved) fit answers enter answer BCE. Gold remains original0/1; safe refusals receive only their existing answer-level negative supervision. Unresolved answer gold remains-1 and never enters BCE.',
        'answer_weights': 'Base: equal mass across fit groups that contain resolved answers, then equal over resolved answers within group. Fit-only weighted class factors0.5/class_base_mass; after class weighting renormalize each such group equally. Final global answer mass1. No question/condition hierarchy or extra token labels in this answer term.',
        'answer_logit': 'max over ALL original candidate-window logits of the whole answer, including windows ineligible for token supervision; unchanged answer geometry. Autograd native max tie handling.',
        'batch': {'unit': '8 whole assigned fit event groups; last batch actual B. Never split a group or answer, never delete refusal/unresolved answers or ineligible candidate windows from forward.',
            'order': 'Per-fold/per-epoch fixed group permutation seed+100*fold+epoch; original answer order within group and original candidate-window order within answer; A/B exact same forward order and step count.',
            'loss': 'G/B * [sum_batch(window_weight*BCE)/3854 + lambda*sum_batch(answer_weight*BCE)/1], where G is ALL assigned fit groups and both global denominators remain fixed.',
            'zero_supervision': 'Compute BCE only on eligible selected indices. Empty component is differentiable sum(logits)*0; never evaluate unknown labels then multiply0. AdamW decay is separate from data gradients.'},
        'selection': 'Each mode/fold first selects ONE epoch1..10 for both units using calibration min(windowF1,answerF1),windowF1,windowPrecision,earlier epoch; epoch0 diagnostic only. Then same chosen mode receives both baselines and same six alpha values with original calibration-only joint-unit selection.',
        'fusion': 'Exact R28/R29 formula and per-fold R26 scores. One alpha per baseline/mode/fold shared by two units;60 candidates per mode/120total, no extra epoch selection for individual baseline.',
        'evaluation': 'Commit all5 folds, both modes and choices before outer metrics. Same eligibility, four-raw-BPE windows and answer max; original R29/R28 controls are only replayed/hashed.',
        'execution': 'Preparation and synthetic CPU selfcheck first; real fit requires a later explicit root instruction. No automatic fit after prepare.',
        'limits': 'Extra answer labels change the objective and effective gradient scale. A positive result would support this fixed supervision change, not prove a universal cure or pure token-level truth detection.',
        'historical_limit': previous.protocol()['historical_limit'],
        'GPU_used': False, 'human_gold': False, 'original_validation_test_opened': False,
        'public_QA_test_opened': False}


def answer_weights(answers, fit_groups):
    active = [i for i, a in enumerate(answers) if a['main_eligible'] and a['group_id'] in fit_groups]
    by_group = defaultdict(list)
    for i in active: by_group[answers[i]['group_id']].append(i)
    base = np.zeros(len(answers), np.float64)
    for ix in by_group.values(): base[ix] = 1 / (len(by_group) * len(ix))
    y = np.asarray([a['gold'] if a['main_eligible'] else -1 for a in answers], int)
    mass = np.asarray([base[y == c].sum() for c in (0, 1)])
    assert (mass > 0).all()
    factors = .5 / mass
    loss = np.zeros_like(base); loss[active] = base[active] * factors[y[active]]
    for ix in by_group.values(): loss[ix] /= loss[ix].sum() * len(by_group)
    assert np.isclose(loss.sum(), 1., rtol=0, atol=1e-12)
    assert all(np.isclose(loss[ix].sum(), 1/len(by_group), rtol=0, atol=1e-12) for ix in by_group.values())
    return {'base': base, 'loss': loss, 'class_factors': factors, 'eligible_indices': np.asarray(active),
            'active_groups': sorted(by_group), 'labels': y, 'target_mass': 1.}


def batches(group_order, answers, aw):
    by_group = defaultdict(list)
    for i, a in enumerate(answers): by_group[a['group_id']].append(i)
    for lo in range(0, len(group_order), GROUP_BATCH):
        groups = list(group_order[lo:lo+GROUP_BATCH])
        ai = [i for g in groups for i in by_group[g]]
        wi = []; spans = []
        for i in ai:
            begin = len(wi); wi.extend(aw[answers[i]['item_id']]); spans.append((begin, len(wi)))
        yield {'groups': groups, 'answer_indices': np.asarray(ai, int), 'window_indices': np.asarray(wi, int),
               'answer_local_bounds': np.asarray(spans, int)}


def components(z, batch, pack):
    wi, ai = batch['window_indices'], batch['answer_indices']
    window_active = np.flatnonzero(pack['window_loss'][wi] > 0)
    if len(window_active):
        target = torch.as_tensor(pack['window_y'][wi[window_active]], dtype=z.dtype)
        assert ((target == 0) | (target == 1)).all()
        weight = torch.as_tensor(pack['window_loss'][wi[window_active]], dtype=z.dtype)
        window = (F.binary_cross_entropy_with_logits(z[window_active], target, reduction='none') * weight).sum() / 3854
    else: window = z.sum() * 0
    # Every candidate participates in max, independently of window eligibility.
    answer_z = torch.stack([z[lo:hi].max() for lo, hi in batch['answer_local_bounds']])
    answer_active = np.flatnonzero(pack['answer']['loss'][ai] > 0)
    if len(answer_active):
        target = torch.as_tensor(pack['answer']['labels'][ai[answer_active]], dtype=z.dtype)
        assert ((target == 0) | (target == 1)).all()
        weight = torch.as_tensor(pack['answer']['loss'][ai[answer_active]], dtype=z.dtype)
        answer = (F.binary_cross_entropy_with_logits(answer_z[answer_active], target, reduction='none') * weight).sum()
    else: answer = z.sum() * 0
    return window, answer, answer_z


def objective(z, batch, pack, mode):
    window, answer, _ = components(z, batch, pack)
    loss = window + (answer if mode == 'window_plus_answer' else answer * 0)
    return loss * len(pack['groups']['fit_groups']) / len(batch['groups'])


def cpu_test():
    assert not torch.cuda.is_initialized(); torch.set_num_threads(4)
    answers = [dict(item_id=str(i), group_id='g'+str(i//2), main_eligible=i != 3,
                    gold=[0,0,1,None][i]) for i in range(4)]
    aw = {str(i): [2*i, 2*i+1] for i in range(4)}
    pack = {'groups': {'fit_groups': ['g0','g1']},
            'window_y': np.array([0,0,-1,-1,1,0,-1,-1]),
            'window_loss': np.array([1,1,0,0,1,1,0,0],np.float64)*3854/4,
            'answer': answer_weights(answers, ['g0','g1'])}
    batch = next(batches(['g0','g1'], answers, aw))
    values = torch.tensor([.1,.2,.3,2.,.4,.5,6.,7.], requires_grad=True)
    ga = torch.autograd.grad(objective(values,batch,pack,'window_only'), values, retain_graph=True)[0]
    gb = torch.autograd.grad(objective(values,batch,pack,'window_plus_answer'), values)[0]
    assert ga[2:4].eq(0).all() and gb[3] > 0 and gb[2] == 0
    assert ga[6:8].eq(0).all() and gb[6:8].eq(0).all()
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    # Winning negative-answer candidate3 has window target-1: max must retain it.
    _, _, az = components(values, batch, pack); assert float(az[1]) == 2.
    # A group's lack of eligible windows must not delete it from A forward.
    no_window = dict(pack, window_loss=np.zeros(8,np.float64))
    assert len(batch['window_indices']) == 8
    zz = values.detach().clone().requires_grad_(True)
    assert torch.autograd.grad(objective(zz,batch,no_window,'window_only'),zz)[0].eq(0).all()
    # Group-batch scale with an actual smaller last batch reconstructs the global
    # objective when logits are fixed. Uses9 groups -> 8+1, not 8+8.
    many = [dict(item_id=str(i),group_id='s'+str(i),main_eligible=True,gold=i%2) for i in range(9)]
    many_aw = {str(i):[i] for i in range(9)}
    many_pack = {'groups':{'fit_groups':[a['group_id'] for a in many]},
        'window_y':np.array([i%2 for i in range(9)]), 'window_loss':np.full(9,3854/9),
        'answer':answer_weights(many,[a['group_id'] for a in many])}
    mz = torch.linspace(-2,2,9); total = 0.
    for part in batches(many_pack['groups']['fit_groups'],many,many_aw):
        loss = objective(mz[part['window_indices']],part,many_pack,'window_plus_answer')
        total += float(loss)*len(part['groups'])/9
    full = {'groups':many_pack['groups']['fit_groups'],'answer_indices':np.arange(9),'window_indices':np.arange(9),
            'answer_local_bounds':np.column_stack((np.arange(9),np.arange(1,10)))}
    assert np.isclose(total,float(objective(mz,full,many_pack,'window_plus_answer')),rtol=0,atol=2e-7)
    # Actual769-wide networks start equal; same forward consumes equal RNG, while
    # only the supervised objective differs. No real examples are trained here.
    torch.manual_seed(SEED); first = Probe(); state = first.state_dict(); rng = torch.get_rng_state().clone()
    second = Probe(); second.load_state_dict(state)
    x = torch.randn(8,769)
    torch.set_rng_state(rng); za=first(x); after_a=torch.get_rng_state().clone()
    torch.set_rng_state(rng); zb=second(x); after_b=torch.get_rng_state().clone()
    assert torch.equal(za,zb) and torch.equal(after_a,after_b)
    objective(za,batch,pack,'window_only').backward()
    objective(zb,batch,pack,'window_plus_answer').backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in second.parameters())
    assert sum(p.numel() for p in first.parameters()) == protocol()['model']['parameters']
    report = {'passed': True, 'known_negative_no_window_supervision_A_gradient0_B_max_gradient_positive': True,
        'unresolved_answer_no_data_gradient_both_modes': True, 'ineligible_window_included_in_answer_max': True,
        'pure_zero_window_group_retained_in_forward': True, 'last_batch8plus1_global_normalization_checked': True,
        'MLP_equal_initial_state_forward_and_dropout_RNG': True, 'MLP_finite_gradients_both_modes': True,
        'optimizer_weight_decay_excluded_from_direct_logit_gradient_test': True,
        'real_data_training': False, 'GPU_used': False}
    assert not torch.cuda.is_initialized()
    return report


def source_hashes():
    paths = [Path(__file__), Path(previous.__file__), Path(previous.original.__file__),
        Path(core.__file__), Path(core.r13.__file__), Path(core.r10.__file__),
        ROOT/'protocol.json', OLD/'complete.json', OLD/'fit_complete.json', OLD/'AUDIT.json',
        OLD/'designs.npz', OLD/'preparation_complete.json', OLD/'PCA32_CONTROL.json',
        SCENE/'windows.jsonl', SCENE/'answers.jsonl', SCENE/'folds.json']
    paths += [OLD/f'fold_{fold}_{suffix}' for fold in range(5) for suffix in ('frozen.pkl','scores.npz','calibration.json')]
    paths += [R26/f'fold_{fold}_{suffix}' for fold in range(5) for suffix in ('scores.npz','calibration.json')]
    return {str(p.resolve()):q.sha(p) for p in paths}


def prepare():
    assert not (OUT/'preparation_complete.json').exists()
    assert not torch.cuda.is_initialized(); torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True); cache.freeze(ROOT/'protocol.json',protocol())
    previous.check_prepared()
    done=q.read(OLD/'complete.json'); assert q.read(OLD/'AUDIT.json')['passed']
    for name,h in done['files_sha256'].items(): assert q.sha(OLD/name)==h
    for name,h in q.read(OLD/'fit_complete.json')['files_sha256'].items(): assert q.sha(OLD/name)==h
    snapshot=source_hashes(); synthetic=cpu_test()
    shutil.copyfile(OLD/'designs.npz',OUT/'designs.npz')
    assert q.sha(OLD/'designs.npz')==q.sha(OUT/'designs.npz')
    windows,answers,folds,aw=metadata(); previous.group_check(windows,answers,folds)
    with np.load(OUT/'designs.npz') as d: x=np.column_stack((d['hidden'],d['logit'])).astype(np.float32)
    assert x.shape==(12222,769) and np.isfinite(x).all()
    names=['designs.npz']; records=[]
    for fold,groups in enumerate(folds):
        old=pickle.loads((OLD/f'fold_{fold}_frozen.pkl').read_bytes()); assert old['groups']==groups
        ix=np.asarray([i for i,w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups['fit_groups']])
        assert np.array_equal(ix,old['model']['fit_ix'])
        window_loss=np.zeros(len(windows),np.float64);window_loss[ix]=old['model']['loss_weights']
        window_y=np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows],int)
        assert np.array_equal(window_y[ix],old['model']['fit_y']) and np.isclose(window_loss.sum(),3854)
        ar=answer_weights(answers,groups['fit_groups'])
        scaler=old['model']['scaler']; assert scaler.n_features_in_==769
        assert np.isfinite(scaler.transform(x)).all()
        ordered=sorted(groups['fit_groups'])
        orders=[np.asarray(ordered)[np.random.default_rng(SEED+100*fold+epoch).permutation(len(ordered))].tolist()
                for epoch in range(1,EPOCHS+1)]
        torch.manual_seed(SEED+fold); model=Probe()
        pack={'groups':groups,'scaler':scaler,'scaler_source_sha256':q.sha(OLD/f'fold_{fold}_frozen.pkl'),
            'window_y':window_y,'window_loss':window_loss,'window_fit_ix':ix,'answer':ar,'orders':orders,
            'initial_state':{k:v.detach().clone() for k,v in model.state_dict().items()},
            'post_initialization_RNG':torch.get_rng_state().clone()}
        check_steps=[]
        for order in orders:
            bb=list(batches(order,answers,aw)); seen=[g for b in bb for g in b['groups']]
            assert len(seen)==len(set(seen))==len(groups['fit_groups'])
            wi=np.concatenate([b['window_indices'] for b in bb]); ai=np.concatenate([b['answer_indices'] for b in bb])
            expected=[i for i,w in enumerate(windows) if w['group_id'] in groups['fit_groups']]
            assert sorted(wi)==expected and len(ai)==len(set(ai))
            assert set(ai)=={i for i,a in enumerate(answers) if a['group_id'] in groups['fit_groups']}
            for b in bb:
                assert not (set(b['groups']) & (set(groups['calibration_groups']) | set(groups['evaluation_groups'])))
                for i,(lo,hi) in zip(b['answer_indices'],b['answer_local_bounds']):
                    assert np.array_equal(b['window_indices'][lo:hi],aw[answers[i]['item_id']])
            check_steps.append(len(bb))
        ref=[i for i,a in enumerate(answers) if a['reviewed_safe_refusal'] and a['group_id'] in groups['fit_groups']]
        assert all(ar['loss'][i]>0 and not window_loss[aw[answers[i]['item_id']]].any() for i in ref)
        unresolved=[i for i,a in enumerate(answers) if not a['main_eligible'] and a['group_id'] in groups['fit_groups']]
        assert not ar['loss'][unresolved].any()
        name=f'fold_{fold}_prepared.pkl';(OUT/name).write_bytes(pickle.dumps(pack,protocol=5));names.append(name)
        records.append({'fold':fold,'assigned_fit_groups':len(ordered),'fit_windows':len(ix),
            'fit_answers_forwarded':len(ai),'resolved_fit_answers':len(ar['eligible_indices']),
            'safe_refusals_answer_loss_only':len(ref),'unresolved_fit_answers_no_answer_loss':len(unresolved),
            'window_loss_mass':float(window_loss.sum()),'answer_loss_mass':float(ar['loss'].sum()),
            'steps_each_epoch_each_mode':check_steps,'scaler_exact_R29':True})
    q.save(OUT/'CPU_SELFCHECK.json',{**synthetic,'actual_folds':records,'all_candidate_windows_forwarded':True})
    names.append('CPU_SELFCHECK.json')
    assert snapshot==source_hashes()
    q.save(OUT/'preparation_complete.json',{'status':'prepared_CPU_checked_waiting_root_training_authorization',
        'files_sha256':{n:q.sha(OUT/n) for n in names},'source_sha256':snapshot,
        'GPU_used':False,'real_data_training':False,'human_gold':False,'original_validation_test_opened':False})
    assert not torch.cuda.is_initialized()
    print('R30_PREPARED_AND_SYNTHETIC_GRADIENTS_PASSED_NO_TRAINING',flush=True)


def check_prepared():
    p=q.read(OUT/'preparation_complete.json');assert p['source_sha256']==source_hashes()
    for n,h in p['files_sha256'].items():assert q.sha(OUT/n)==h,n
    return p


def fit():
    prepared=check_prepared();assert not (OUT/'fit_started.json').exists()
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    q.save(OUT/'fit_started.json',{'time':time.time(),'preparation_sha256':q.sha(OUT/'preparation_complete.json')})
    windows,answers,folds,aw=metadata();names=[];start=time.perf_counter()
    with np.load(OUT/'designs.npz') as d:raw=np.column_stack((d['hidden'],d['logit'])).astype(np.float32)
    for fold,groups in enumerate(folds):
        pack=pickle.loads((OUT/f'fold_{fold}_prepared.pkl').read_bytes())
        x=torch.from_numpy(pack['scaler'].transform(raw).astype(np.float32))
        values={};ts={};epochs={};tables={};candidates={};chosen={}
        with np.load(R26/f'fold_{fold}_scores.npz') as data:
            for name in BASES:values[name]=data[name].copy()
        old=q.read(R26/f'fold_{fold}_calibration.json')
        for name in BASES:
            ts[name]=thresholds(values[name],groups['calibration_groups'],windows,answers,aw)
            assert ts[name]==old['thresholds'][name]
        for mode in MODES:
            directory=OUT/f'fold_{fold}_{mode}';directory.mkdir()
            model=Probe();model.load_state_dict(pack['initial_state']);torch.set_rng_state(pack['post_initialization_RNG'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            history=[];best=None
            for epoch in range(EPOCHS+1):
                tick=time.perf_counter()
                if epoch:
                    model.train()
                    for batch in batches(pack['orders'][epoch-1],answers,aw):
                        optimizer.zero_grad(set_to_none=True)
                        z=model(x[batch['window_indices']]);loss=objective(z,batch,pack,mode)
                        assert torch.isfinite(loss);loss.backward()
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    z=model(x);prob=torch.sigmoid(z).double().numpy()
                    full={'groups':groups['fit_groups'],
                        'answer_indices':np.arange(len(answers)), 'window_indices':np.arange(len(windows)),
                        'answer_local_bounds':None}
                    # Global components use exact answer-window membership; windows
                    # are not required to be contiguous across the original table.
                    win_ix=np.flatnonzero(pack['window_loss']>0)
                    wt=torch.as_tensor(pack['window_loss'][win_ix],dtype=torch.float32)
                    wy=torch.as_tensor(pack['window_y'][win_ix],dtype=torch.float32)
                    wl=float((F.binary_cross_entropy_with_logits(z[win_ix],wy,reduction='none')*wt).sum()/3854)
                    ai=pack['answer']['eligible_indices'];az=torch.stack([z[aw[answers[i]['item_id']]].max() for i in ai])
                    ay=torch.as_tensor(pack['answer']['labels'][ai],dtype=torch.float32)
                    at=torch.as_tensor(pack['answer']['loss'][ai],dtype=torch.float32)
                    al=float((F.binary_cross_entropy_with_logits(az,ay,reduction='none')*at).sum())
                threshold=thresholds(prob,groups['calibration_groups'],windows,answers,aw)
                key=previous.selection_key(threshold,epoch)
                entry={'epoch':epoch,'thresholds':threshold,'key':key,'fit_window_BCE':wl,
                    'fit_answer_BCE':al,'fit_objective':wl+(al if mode=='window_plus_answer' else 0),
                    'seconds':time.perf_counter()-tick}
                stem=f'epoch_{epoch:02d}'; np.save(directory/(stem+'_probability.npy'),prob)
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'RNG':torch.get_rng_state(),
                    'epoch':epoch,'mode':mode,'fold':fold},directory/(stem+'.pt'))
                q.save(directory/(stem+'.json'),entry);history.append(entry)
                names += [str((directory/(stem+suffix)).relative_to(OUT)) for suffix in ('.pt','.json','_probability.npy')]
                if epoch and (best is None or key>best['key']):best=entry
            values[mode]=np.load(directory/f'epoch_{best["epoch"]:02d}_probability.npy');ts[mode]=best['thresholds'];epochs[mode]=best
            q.save(directory/'complete.json',{'history':history,'selected':best});names.append(str((directory/'complete.json').relative_to(OUT)))
            for prior,fused in zip(BASES,FUSIONS[mode]):
                table=[];scores=[]
                for alpha in ALPHA:
                    v=previous.combine(values[prior],values[mode],alpha)
                    t=thresholds(v,groups['calibration_groups'],windows,answers,aw)
                    if alpha==0:assert np.array_equal(v,values[prior]) and t==ts[prior]
                    table.append({'alpha':alpha,'thresholds':t,'key':previous.selection_key(t,alpha)});scores.append(v)
                selected=max(range(6),key=lambda j:table[j]['key'])
                tables[fused]=table;candidates[fused]=np.asarray(scores);chosen[fused]=table[selected]
                values[fused]=scores[selected];ts[fused]=table[selected]['thresholds']
            print('R30_MODE_FROZEN',fold,mode,'epoch',best['epoch'],flush=True)
        frozen={'groups':groups,'thresholds':ts,'selected_epochs':epochs,'alpha_tables':tables,'selected_alpha':chosen,'human_gold':False}
        q.save(OUT/f'fold_{fold}_calibration.json',frozen)
        np.savez_compressed(OUT/f'fold_{fold}_scores.npz',**values,**{k+'_all_alpha':v for k,v in candidates.items()})
        names += [f'fold_{fold}_calibration.json',f'fold_{fold}_scores.npz']
    assert prepared==check_prepared();assert not torch.cuda.is_initialized()
    q.save(OUT/'fit_complete.json',{'status':'all5fold_both_modes_fit_calibration_frozen_before_outer',
        'source_sha256':source_hashes(),'files_sha256':{n:q.sha(OUT/n) for n in names},
        'seconds':time.perf_counter()-start,'new_MLP_fits':10,'trained_epochs':100,
        'PCA_LR_encoder_control_refits':0,'human_gold':False,'GPU_used':False})


def evaluate():
    spec=importlib.util.spec_from_file_location('_r30_same_evaluation',previous.original.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=OUT;module.METHODS=METHODS;module.source_hashes=source_hashes;module.evaluate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','check','fit','evaluate'));args=parser.parse_args()
    with threadpool_limits(limits=4):
        try:
            {'prepare':prepare,'check':check_prepared,'fit':fit,'evaluate':evaluate}[args.stage]()
        except BaseException as exc:
            OUT.mkdir(parents=True,exist_ok=True)
            q.save(OUT/f'FAILURE_{args.stage}_{time.time_ns()}.json',{'error':repr(exc),'GPU_used':False})
            raise
