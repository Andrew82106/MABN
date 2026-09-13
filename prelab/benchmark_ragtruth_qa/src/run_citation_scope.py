"""Conservative standalone citation heading scope; CPU-only matched development."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import os
import pickle
import re
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run_development as q
import build_citation_alignment as citation
import run_completed_score_fusion as upstream

OUT=q.ROOT/'results/citation_heading_scope_v1'
OLD=q.ROOT/'results/citation_alignment_lr_v1'
LEX=q.ROOT/'results/citation_alignment_v1'
CS=(.001,.01,.1)
NAMES=['inherited_scope_token_fraction','inherited_source_coverage_mean','inherited_source_coverage_gap_mean']
BULLET=re.compile(r'^\s*(?:[*+\-•]|\d+[.)])\s+\S')


def header_id(line):
    text=line.strip()
    if text.startswith('**') and text.endswith('**'):
        text=text[2:-2].strip()
    else:
        m=re.fullmatch(r'\*\*(Passage\s*[123])\*\*\s*:',text,re.I)
        if m:text=m.group(1)+':'
    m=re.fullmatch(r'Passage\s*([123])\s*:',text,re.I)
    return int(m.group(1)) if m else None


def scope_claims(text,claims):
    """Only answer text/old parser. No question, evidence, labels or scores."""
    result={}; active=None; cursor=0; headers=[]; bullet_lines=0
    for line in text.splitlines(keepends=True):
        end=cursor+len(line.rstrip('\r\n')); hid=header_id(line)
        if hid is not None:
            active={'source_id':hid,'start':cursor,'end':end,'text':text[cursor:end]}
            headers.append(active.copy())
        elif not line.strip():
            pass
        elif active is not None and BULLET.match(line):
            bullet_lines+=1
            for c in claims:
                if cursor<=c['start'] and c['end']<=end and c['parser']['status']=='none':
                    ci=c['claim_index']; assert ci not in result
                    result[ci]=active.copy()
        else:
            active=None
        cursor+=len(line)
    return result,headers,bullet_lines


def tiny():
    text='**Passage 1:**\n\n* First fact.\n* Inline fact (passage 3).\n2. Second fact.\nSummary starts.\n* No inherited scope.\nPassage2:\n- Third fact.\nOther heading:\n- No inheritance.\n'
    claims=[];cur=0
    for line in text.splitlines(keepends=True):
        s=line.rstrip('\r\n')
        if s:claims.append({'claim_index':len(claims),'start':cur,'end':cur+len(s),'text':s,'parser':citation.parse_citations(s)})
        cur+=len(line)
    got,_,_=scope_claims(text,claims)
    bytext={c['text']:got.get(c['claim_index']) for c in claims}
    assert bytext['* First fact.']['source_id']==1 and bytext['2. Second fact.']['source_id']==1
    assert bytext['* Inline fact (passage 3).'] is None
    assert bytext['* No inherited scope.'] is None and bytext['- No inheritance.'] is None
    assert bytext['- Third fact.']['source_id']==2
    assert header_id('**Passage 3**:')==3 and header_id('**Passage 3:') is None
    assert header_id('Passage 2 states that') is None and header_id('Passage 4:') is None
    # Unknown citation and a claim crossing two bullet lines must not inherit.
    t='Passage 1:\n* First [2024].\n* Second.\n'
    c=[{'claim_index':0,'start':11,'end':25,'parser':citation.parse_citations('First [2024].')},
       {'claim_index':1,'start':11,'end':len(t),'parser':citation.parse_citations('First Second.')}]
    assert not scope_claims(t,c)[0]
    return {'passed':True,'inline_preserved':True,'plain_summary_closes':True,
            'heading_switches':True,'blank_lines_supported':True,'unknown_and_mixed_claims_excluded':True,
            'paired_bold_and_numbered_bullets_checked':True,'GPU_used':False}


def protocol():
    return {'version':'citation-heading-scope-v1','peers':list(upstream.PEERS),'C':list(CS),'new_fits':9,
        'fit':'Original634 fit /168123 windows; calibration159 /42241 windows. Original4rawBPE, all answers, no test.',
        'header':'Complete standalone Passage1/2/3: line (space optional), case-insensitive; optionally a paired Markdown ** wrapper around header or name. No narrative source mentions.',
        'scope':'Only subsequent explicit *,+,-,bullet-dot or numbered N./N) lines. Blank lines allowed. Plain prose, summary, unsupported heading or other nonempty line immediately closes. Valid new source heading switches. Wrapped plain continuation closes; no inferred multiline propagation.',
        'claim_rule':'Existing claim boundaries unchanged. Whole claim must lie within one scoped list line. Only old parser status none may inherit; inline/unknown references never overwritten. Header itself not attributed as content.',
        'new_features':NAMES,
        'calculation':'Per inherited claim: 1, word-set coverage in inherited source, max of original three source word-coverages minus that coverage; per noninherited claim all0. Mean over existing lexical-token->claim assignment in each original4raw window. Header/source IDs are metadata, never numeric predictors.',
        'common':'Exact frozen peer+tail2 two probabilities and exact original8 lexical/citation features; new model13 columns. Old10-column threeC models reused, no control refits.',
        'weights':'Exactly q.base_weights: same train-only group/class weighting and168123 loss mass. Fit-only base-weighted StandardScaler.',
        'LR':{'C':list(CS),'solver':'liblinear','penalty':'l2','max_iter':2000,'seed':20261010},
        'selection':'Each candidate gets original cal F1-optimal window/answer thresholds; choose C by q.selection_key(min twoF1,windowF1,windowprecision,smallerC). Answer score max over every original eligible window.',
        'limits':['Does not fix step IDs, temporal sequence, sentence-final detached citations, narrative scope, or entailment.',
                  'No new semantic forward. Existing upstream MiniCheck scores remain part of both models: not a pure white-box probe.',
                  'Upstream fit scores are in-sample, not cross-fitted; cal repeatedly exposed. Exploration, not independent test.'],
        'GPU_used':False,'official_test_opened':False}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'design_freeze.json').exists()
    q.save(OUT/'protocol.json',protocol());q.save(OUT/'CPU_SELFCHECK.json',tiny())
    paths=[Path(__file__),Path(q.__file__),Path(citation.__file__),Path(upstream.__file__),
           OLD/'complete.json',OLD/'summary.json',LEX/'complete.json',upstream.OUT/'complete.json',
           q.DATA/'gold_manifest.json']
    q.save(OUT/'design_freeze.json',{'protocol_sha256':q.sha(OUT/'protocol.json'),
        'source_sha256':{str(p.resolve()):q.sha(p) for p in paths},'trained':False,'official_test_opened':False})
    print('CITATION_SCOPE_FROZEN_NO_FIT',os.getpid(),flush=True)


def verify():
    f=q.read(OUT/'design_freeze.json');assert q.read(OUT/'protocol.json')==protocol()
    assert f['protocol_sha256']==q.sha(OUT/'protocol.json')
    for p,h in f['source_sha256'].items():assert q.sha(Path(p))==h,p


def build():
    verify();assert not (OUT/'features_complete.json').exists()
    meta=q.metadata();old=q.read(LEX/'complete.json')
    for n,h in old['files_sha256'].items():assert q.sha(LEX/n)==h,n
    assert old['window_order_sha256']==q.digest([w['window_id'] for w in meta['windows']])
    rows={r['response_id']:r for part in q.PARTITIONS for r in q.lines(q.DATA/(part+'.jsonl'))}
    by=defaultdict(list)
    for c in q.lines(LEX/'claims.jsonl'):by[c['response_id']].append(c)
    infos={};counts={part:Counter() for part in q.PARTITIONS};records=[];bound_checks={}
    for answer in meta['answers']:
        rid=answer['response_id'];part=answer['partition'];row=rows[rid];t=meta['by_response'][rid]['tokens']
        text=row['original_response'];claims=by[rid];assert text==t['original_response']
        scope,headers,bullets=scope_claims(text,claims)
        source_words={i:citation.word_set(s) for i,s in citation.split_sources(row['retrieved_passages']).items()}
        values=np.zeros((len(claims),3),np.float64)
        for c in claims:
            ci=c['claim_index'];assert claims[ci] is c and text[c['start']:c['end']]==c['text']
            if ci in scope:
                words=citation.word_set(c['text'])
                cover={i:len(words & ws)/len(words) if words else 0. for i,ws in source_words.items()}
                val=cover[scope[ci]['source_id']]
                values[ci]=[1,val,max(cover.values())-val]
                records.append({'response_id':rid,'partition':part,'claim_index':ci,'start':c['start'],'end':c['end'],
                    'text':c['text'],'old_parser':c['parser'],'scope':scope[ci],'features':values[ci].tolist()})
        groups=citation.token_claim_assignment(text,t['response_token_offsets'],t['lexical_mask'],claims)
        valid=groups>=0; affected=np.zeros(len(groups),bool);affected[valid]=values[groups[valid],0]>0
        infos[rid]=(values,groups)
        counts[part].update(answers=1,claims=len(claims),headers=len(headers),bullet_lines=bullets,
            newly_attributed_claims=len(scope),affected_answers=int(bool(scope)),affected_lexical_tokens=int(affected.sum()))
        if rid=='12213':
            assert scope[8]['source_id']==2 and claims[8]['parser']['status']=='none'
            assert 15 not in scope and 16 not in scope and scope[11]['source_id']==3
            bound_checks={'response_id':rid,'risk_bullet_claim8_scope':scope[8],
                'summary_claim15_no_scope':15 not in scope,'next_header_claim11_scope':scope[11]}
    assert bound_checks
    features=[]
    for w in meta['windows']:
        t=meta['by_response'][w['response_id']]['tokens'];values,groups=infos[w['response_id']]
        ii=[i for i in w['token_indices'] if t['lexical_mask'][i]];assert ii
        value=values[groups[ii]].mean(axis=0);features.append(value)
        counts[w['partition']]['affected_windows']+=int(value[0]>0)
    x=np.asarray(features,np.float32);assert x.shape==(210364,3) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
    np.save(OUT/'window_features.npy',x);q.save(OUT/'feature_names.json',NAMES)
    q.save(OUT/'inherited_claims.json',records);q.save(OUT/'coverage.json',{'partitions':counts,'label_fields_used':False,'actual12213_check':bound_checks})
    q.save(OUT/'geometry.json',{'window_order_sha256':q.digest([w['window_id'] for w in meta['windows']]),
        'bounds':meta['bounds'],'old8_features_sha256':q.sha(LEX/'window_features.npy'),'claims_unchanged':True,
        'windows_unchanged':True,'answers':793,'windows':210364})
    files=['window_features.npy','feature_names.json','inherited_claims.json','coverage.json','geometry.json']
    q.save(OUT/'features_complete.json',{'status':'complete','trained':False,'no_test':True,
        'files_sha256':{n:q.sha(OUT/n) for n in files},'design_freeze_sha256':q.sha(OUT/'design_freeze.json')})
    print('CITATION_SCOPE_FEATURES_COMPLETE',dict(counts),flush=True)


def pair_features(peer,source_entries):
    pairs=[]
    for alpha in (0.,1.):
        e=next(e for e in source_entries['all_candidates'][peer] if e['tail_weight']==alpha)
        p=upstream.OUT/(e['candidate']+'_scores.npz');assert q.sha(p)==e['scores_sha256']
        with np.load(p) as z:pairs.append(z['window_scores'].copy())
    return pairs


def check_controls(meta,peer,common,old_summary):
    family=peer+'__two_scores_and_citation';entries=old_summary['all_candidates'][family]
    assert len(entries)==3 and {e['C'] for e in entries}==set(CS)
    checks=[]
    for e in entries:
        sp=OLD/(family+'_scaler.pkl');mp=OLD/(e['candidate']+'.pkl');pp=OLD/(e['candidate']+'_scores.npz')
        for p,key in ((sp,'scaler_sha256'),(mp,'model_sha256'),(pp,'scores_sha256')):assert q.sha(p)==e[key]
        scaler=pickle.loads(sp.read_bytes());model=pickle.loads(mp.read_bytes())
        replay=model.predict_proba(scaler.transform(common))[:,1]
        with np.load(pp) as z:score=z['window_scores'].copy();answer=z['answer_scores'].copy()
        difference=float(np.max(np.abs(replay-score)));assert difference<=1e-12
        assert np.array_equal(answer,q.answer_scores(meta,score))
        assert q.metrics(meta,score,e['thresholds'])==e['metrics']
        lo,hi=meta['bounds']['calibration']
        thresholds={'window':q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],score[lo:hi]),
                    'answer':q.choose_threshold([a['label'] for a in meta['answers'][634:]],answer[634:])}
        assert thresholds==e['thresholds']
        checks.append({'candidate':e['candidate'],'score_replay_maxdiff':difference,'hashes_and_metrics_exact':True})
    assert max(entries,key=lambda e:e['selection_key'])==old_summary['selected'][family]
    return checks,old_summary['selected'][family]


def run():
    verify();assert not (OUT/'started.json').exists()
    manifest=q.read(OUT/'features_complete.json');assert manifest['status']=='complete' and manifest['no_test'] and not manifest['trained']
    assert manifest['design_freeze_sha256']==q.sha(OUT/'design_freeze.json')
    for n,h in manifest['files_sha256'].items():assert q.sha(OUT/n)==h
    meta=q.metadata();scope=np.load(OUT/'window_features.npy');lexical=np.load(LEX/'window_features.npy')
    geometry=q.read(OUT/'geometry.json');assert geometry['old8_features_sha256']==q.sha(LEX/'window_features.npy')
    assert geometry['window_order_sha256']==q.digest([w['window_id'] for w in meta['windows']])
    assert scope.shape==(210364,3) and lexical.shape==(210364,8)
    bw,lw,_,y=q.base_weights(meta);assert len(y)==168123
    source_entries=q.read(upstream.OUT/'summary.json');assert q.sha(upstream.OUT/'summary.json')==q.read(upstream.OUT/'complete.json')['summary_sha256']
    old_summary=q.read(OLD/'summary.json');assert q.sha(OLD/'summary.json')==q.read(OLD/'complete.json')['summary_sha256']
    q.save(OUT/'started.json',{'pid':os.getpid(),'time':time.time(),'design_freeze_sha256':q.sha(OUT/'design_freeze.json'),
        'features_complete_sha256':q.sha(OUT/'features_complete.json'),'GPU_used':False})
    candidates={};selected={};controls={};checks=[];start=time.perf_counter()
    for peer in upstream.PEERS:
        common=np.column_stack((*pair_features(peer,source_entries),lexical));assert common.shape==(210364,10)
        chk,control=check_controls(meta,peer,common,old_summary);checks.extend(chk);controls[peer]=control
        x=np.column_stack((common,scope));assert np.array_equal(x[:,:10],common)
        scaler=StandardScaler().fit(x[:len(y)],sample_weight=bw);scaled=scaler.transform(x)
        family=peer+'__heading_scope';sp=OUT/(family+'_scaler.pkl');sp.write_bytes(pickle.dumps(scaler,protocol=5))
        entries=[]
        for c in CS:
            tick=time.perf_counter();model=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=20261010)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(scaled[:len(y)],y,sample_weight=lw)
            score=model.predict_proba(scaled)[:,1];answer=q.answer_scores(meta,score);lo,hi=meta['bounds']['calibration']
            thresholds={'window':q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],score[lo:hi]),
                        'answer':q.choose_threshold([a['label'] for a in meta['answers'][634:]],answer[634:])}
            metrics=q.metrics(meta,score,thresholds);candidate=family+f'__C{c:g}'
            mp=OUT/(candidate+'.pkl');pp=OUT/(candidate+'_scores.npz')
            mp.write_bytes(pickle.dumps(model,protocol=5));np.savez_compressed(pp,window_scores=score,answer_scores=answer)
            e={'candidate':candidate,'peer':peer,'C':c,'thresholds':thresholds,'metrics':metrics,
               'selection_key':list(q.selection_key(thresholds,c)),'input_width':13,'iterations':model.n_iter_.tolist(),
               'seconds':time.perf_counter()-tick,'model_sha256':q.sha(mp),'scaler_sha256':q.sha(sp),'scores_sha256':q.sha(pp)}
            q.save(OUT/(candidate+'.json'),e);entries.append(e)
            print('CITATION_SCOPE_FIT',candidate,metrics['calibration']['windows']['f1'],metrics['calibration']['answers']['f1'],flush=True)
        candidates[peer]=entries;selected[peer]=max(entries,key=lambda e:e['selection_key'])
    q.save(OUT/'CONTROL_REPLAY_CHECK.json',{'status':'passed','controls':checks,'no_control_refit':True})
    summary={'selected':selected,'all_candidates':candidates,'reused_controls':controls,'new_fits':9,'control_fits':0,
        'seconds':time.perf_counter()-start,'GPU_used':False,'official_test_opened':False}
    q.save(OUT/'summary.json',summary)
    lines=['# 独立资料标题作用范围：CPU 对照','','保留原2个风险分数与8列引用特征，新增仅3列保守标题继承特征。旧三个C控制逐个回放核对，不重训。',
        '新9个LR候选全部保留；两模式均按原159校准选择阈值/C；原634训练、793全答/210364窗口保持。','','| 底层对象 | 模式 | C | 窗口F1 | 整答F1 |','|---|---|---:|---:|---:|']
    for peer in upstream.PEERS:
        for mode,e in [('原10列控制',controls[peer]),('追加3列scope',selected[peer])]:
            m=e['metrics']['calibration'];lines.append(f"| {peer} | {mode} | {e['C']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    lines+=['','独立标题确实被旧claim解析丢失作用范围；修复覆盖不等于错误检出率。它不能解决步骤编号、时间顺序、句尾括号引用或一般语义矛盾。',
        '没有新GPU、NLI推理或标签。两边都有额外MiniCheck上游分数，不能称纯原生成白盒。',
        '上游fit分数非交叉拟合，校准反复用于开发；此处不是独立测试或泛化保证。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    q.save(OUT/'complete.json',{'status':'complete','summary_sha256':q.sha(OUT/'summary.json'),
        'new_fits':9,'control_refits':0,'GPU_used':False,'official_test_opened':False,'pid':os.getpid()})
    print('CITATION_SCOPE_COMPLETE',os.getpid(),summary['seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','build','run'))
    with threadpool_limits(limits=4):globals()[parser.parse_args().stage]()
