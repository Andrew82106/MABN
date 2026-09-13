"""Post-hoc inspection of twelve already exposed calibration cases. No inference/fit."""
from pathlib import Path
import hashlib, json, re
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRE = ROOT.parent
NAME = 'semantic_claim__old_tree__large_weight0.4'
RESULT = ROOT / 'results/large_fixed_convex_v1'

SPECS = [
 ('17001',0,'missed','事件—参与者绑定',
  ['Apple was incorporated January 3, 1977 without Wayne', 'Apple Computer Inc. was established on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne'],
  '1976年成立和1977年注册的参与者被混到一起；主体名与日期大多能在资料找到，但without Wayne被改成有Wayne。'),
 ('17325',0,'missed','错误宣称资料没有答案；任务措辞边界',
  ['Stack the Chinese broccoli together and cut into 4/10cm lengths and stack on top of each other.', 'Trim the ends of the Chinese broccoli'],
  '回答否认passage1提供trim方法，同时承认其cut步骤；原标注把cut当trim。确有相关步骤，但trim与cut的任务解释不完全等价，本例不作为最干净的逻辑矛盾。'),
 ('17313',2,'missed','金额与比率的角色/单位',
  ['For this example, we will use 35%.', 'For every $100 in food sales, we would need to transfer $35 to COGS.'],
  '资料的目标是35%，$35是$100销售额对应的成本；回答将目标写成$35并拿它乘$100。同答其他错误已报警，此处仍漏。'),
 ('12687',0,'missed','因果方向反转',
  ['Taking large amounts of vitamin D along with some water pills might cause to be too much calcium in the body.', 'This could cause serious side effects including kidney problems.'],
  '资料说可能造成肾脏问题，回答写可能预防肾脏问题；疾病、钙和药物词均重合，谓词方向相反。仅比较给定资料，不提供医学事实建议。'),
 ('13575',0,'missed','来源归属跨列表；meta正确来源号错误',
  ['passage 1:Tip!', 'passage 3:Run PowerISO.', 'Choose the burning speed. Click Burn to start burning.'],
  '待检步骤本身在passage3有依据，但被放在Passage1总领的列表下。金标meta写正确来源为passage2，实际为3；错归到1仍成立。'),
 ('14229',0,'missed','引用作用域与远处事实绑定',
  ['Turn the heat to medium-low, cover, and cook for 25 minutes.', 'Cook the chicken until it reaches an internal temperature of 165F.'],
  '金标只标段首Passage2，真正错归的是同段后面的165F信息，资料该信息在passage3；不能只看引用数字附近判断。'),
 ('15519',0,'missed','实体比较方向反转',
  ['The two-toed sloth is slightly bigger than the three-toed sloth'],
  '资料bigger被写成smaller；两实体均在原文，后面的编造尺寸另有报警，比较句本身仍低风险。'),
 ('17361',0,'missed','多来源时间差异与近似表述边界',
  ['for about 15 minutes, or until cakes have risen', 'bake for approx.25 minutes until golden brown and firm to the touch.'],
  '两个来源分别约15与25分钟，回答写15–20分钟。20未直接出现；meta把不同配方拼成15–25区间，本例含源间差异/近似解释，不能当纯粹日期识别失败。'),
 ('16929',0,'detected','有报警，但EC注释未被可见资料证明',
  ["After the time has elapsed, don't open the oven but turn it off."],
  '金标meta引用Don\'t preheat the oven，但官方发布资料与实际输入均无此句；400F/200C预热步骤缺依据。仍按原EC计分，不据此宣称模型识别了“预热/不预热”矛盾。'),
 ('13305',0,'detected','合计金额被重复加倍',
  ['Long Term Care Insurance Rates for Couple Both Age 55. Average Cost: $3,381-per-year (combined).'],
  '资料3381是夫妻合计，回答当作单人费用后再合成6762。此数量/角色错误被检出；不能说全部数值冲突都学不会。仅核资料一致性。'),
 ('14187',0,'detected','期限更新条件与总年数',
  ['extended the renewal term from 28 to 47 years', 'a total term of protection of 75 years.'],
  '资料说明续期由28延至47，总75；回答仍用28+28=56。本例被检出；只比较数据内文字，不作现行法律解释。'),
 ('15585',1,'detected','谓词否定；同时存在回答内部矛盾',
  ['pi bond when dealing with saturated molecules is not used'],
  '资料not used与回答more prominent相反；回答后段又自己写not used，存在额外自相矛盾线索，不能把此检出单独归因于资料对比。资料自身科学表述质量不由本分析背书。'),
]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ht(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def lines(p):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def dump(p, obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n','utf-8')

def run():
    ids={x[0] for x in SPECS}
    report=json.loads((RESULT/(NAME+'.json')).read_text('utf-8'))
    sf=RESULT/(NAME+'_scores.npz')
    assert sha(sf)==report['scores_sha256']
    saved=np.load(sf)
    scores=saved['window_scores'][168123:]
    answerscores=saved['answer_scores'][634:]
    wt=report['thresholds']['window']['threshold']; at=report['thresholds']['answer']['threshold']
    cal=list(lines(ROOT/'data/calibration.jsonl'))
    allanswers=list(lines(ROOT/'data/answers_calibration.jsonl'))
    rows={r['response_id']:r for r in cal if r['response_id'] in ids}
    toks={r['response_id']:r for r in lines(ROOT/'data/tokens_calibration.jsonl') if r['response_id'] in ids}
    inputs={r['response_id']:r for r in lines(ROOT/'results/full_context_encoder_large_v1/inputs.jsonl') if r['partition']=='calibration' and r['response_id'] in ids}
    windows=list(lines(ROOT/'data/windows_k4_calibration.jsonl'))
    assert len(windows)==len(scores)==42241 and len(allanswers)==len(answerscores)==159
    ai={r['response_id']:i for i,r in enumerate(allanswers)}
    wi=defaultdict(list)
    for i,w in enumerate(windows):wi[w['response_id']].append(i)
    tokenizer=AutoTokenizer.from_pretrained(PRE/'models/ModernBERT-large',local_files_only=True)
    cases=[]
    for rid,si,outcome,kind,needles,note in SPECS:
        r,t,inp=rows[rid],toks[rid],inputs[rid]
        lab=r['labels'][si]
        assert lab['label_type']=='Evident Conflict' and r['original_response'][lab['start']:lab['end']]==lab['text']
        assert r['answer_sha256']==t['answer_sha256']==inp['answer_sha256']==ht(r['original_response'])
        prefix=r['retrieved_passages']+tokenizer.sep_token+r['question']+tokenizer.sep_token
        visible=prefix+r['original_response']
        encoded=tokenizer(visible,truncation=False,return_offsets_mapping=True)
        assert encoded['input_ids']==inp['input_ids'] and len(prefix)==inp['answer_start_character']
        raw_idx,enc_idx,weights=map(np.asarray,inp['mapping'])
        mass=np.bincount(raw_idx.astype(int),weights=weights,minlength=t['token_count'])
        lexical=np.asarray(t['lexical_mask'],bool)
        # Frozen preparation stores float32 fractional weights and itself uses2e-7.
        # E.g. float32(1/3)+float32(2/3)=1+2.98e-8 in float64 summation.
        # This check does not alter any mapping or numerical model gate.
        assert np.max(abs(mass[lexical]-1))<2e-7
        spanmap=next(s for s in t['span_token_mapping'] if s['span_index']==si)
        risk=set(spanmap['risk_token_indices'])
        assert risk and all(mass[i]>0 for i in risk)
        ix=[i for i in wi[rid] if risk.intersection(windows[i]['lexical_token_indices'])]
        assert ix and all(windows[i]['label']==1 for i in ix)
        local=scores[ix]; predicted=local>=wt
        assert (predicted.sum()==0) if outcome=='missed' else (predicted.sum()==len(ix))
        best=ix[int(np.argmax(local))]
        aq=[]
        for needle in needles:
            start=r['retrieved_passages'].find(needle)
            assert start>=0,(rid,needle)
            aq.append({'start':start,'end':start+len(needle),'text':needle})
        aw=scores[wi[rid]]
        assert float(aw.max())==float(answerscores[ai[rid]])
        # Only exact selected excerpts, full frozen input retained in its original file.
        answercontext=r['original_response'][max(0,lab['start']-180):min(len(r['original_response']),lab['end']+220)]
        rec={'response_id':rid,'source_id':r['source_id'],'group_id':r['group_id'],'partition':'calibration','question':r['question'],
             'span_index':si,'outcome':outcome,'interpretation':kind,'gold':lab,'source_excerpts':aq,'answer_context':answercontext,
             'interpretation_note':note,'span_windows':{'n':len(ix),'tp':int(predicted.sum()),'fn':int((~predicted).sum()),
             'score_min':float(local.min()),'score_mean':float(local.mean()),'score_max':float(local.max()),
             'all_window_ids':[windows[i]['window_id'] for i in ix],'all_scores':local.tolist(),
             'highest_window':{**windows[best],'risk_score':float(scores[best])}},
             'answer':{'score':float(answerscores[ai[rid]]),'predicted_risk':bool(answerscores[ai[rid]]>=at),'gold_risk':allanswers[ai[rid]].get('label',allanswers[ai[rid]].get('answer_risk'))},
             'mapping_check':{'encoder_tokens':len(inp['input_ids']),'raw_answer_tokens':t['token_count'],'actual_prepared_input_ids_exact':True,
             'truncation':False,'span_lexical_risk_tokens':len(risk),'all_risk_tokens_have_encoder_mapping':True,'lexical_mapping_max_mass_error':float(abs(mass[lexical]-1).max())},
             'hashes':{'retrieved_passages_sha256':ht(r['retrieved_passages']),'released_prompt_sha256':ht(r['released_prompt']),
             'answer_sha256':r['answer_sha256'],'model_visible_text_sha256':ht(visible)},
             'model_visible_prefix':{'evidence':[0,len(r['retrieved_passages'])],'question':[len(r['retrieved_passages'])+len(tokenizer.sep_token),len(prefix)-len(tokenizer.sep_token)],'answer_start':len(prefix)}}
        if rid=='14229':
            s=r['original_response'].index('internal temperature',lab['end'])
            rec['citation_scope_distance']={'later_fact_start':s,'characters_after_gold_end':s-lab['end'],'later_fact_excerpt':r['original_response'][s:s+70]}
        if rid=='13575':
            s=r['original_response'].index('Passage 1 provides')
            rec['citation_scope_distance']={'governing_source_header_start':s,'characters_before_gold_start':lab['start']-s,'governing_source_header':r['original_response'][s:r['original_response'].index('\n',s)]}
        cases.append(rec)
    # Only two expressly authorized cal source records are parsed from the raw source file.
    target={rows[rid]['source_id']:rows[rid] for rid in ('16929','13575')}
    trace=[]
    rawfile=PRE/'data/raw/source_info.jsonl'
    sid_re=re.compile(r'"source_id"\s*:\s*"([^"\\]+)"')
    with rawfile.open(encoding='utf-8') as f:
        for line in f:
            m=sid_re.search(line)
            if not m or m.group(1) not in target:continue
            source=json.loads(line);r=target[m.group(1)]
            assert source['source_info']['passages']==r['retrieved_passages']
            assert source['prompt']==r['released_prompt']
            assert source['source_info']['question']==r['question']
            assert r['retrieved_passages'] in r['released_prompt']
            trace.append({'response_id':r['response_id'],'source_id':r['source_id'],'raw_source_line_sha256':ht(line),
                'raw_file':str(rawfile.resolve()),'raw_file_sha256':sha(rawfile),
                'fields':{'source_info.passages':{'characters':len(r['retrieved_passages']),'sha256':ht(r['retrieved_passages']),'frozen_field':'retrieved_passages','exact':True,'missing_characters':0},
                'prompt':{'characters':len(r['released_prompt']),'sha256':ht(r['released_prompt']),'frozen_field':'released_prompt','exact':True,'missing_characters':0}},
                'actual_encoder_input_reconstructed_exact':True,'question_exact':True,
                'preheat_in_official_passages':bool(re.search('preheat',source['source_info']['passages'],re.I)),
                'burning_speed_official_occurrences':[m.start() for m in re.finditer('Choose the burning speed',source['source_info']['passages'])],
                'unchanged_original_gold_meta':r['labels'][0]['meta']})
    assert len(trace)==2
    result={'status':'complete_posthoc_descriptive_review','candidate':NAME,'window_threshold':wt,'answer_threshold':at,
            'scores_sha256':sha(sf),'selected_large_epoch':3,'case_selection':'Purposive post-hoc selection after reading exposed calibration examples: 8 fully missed EC spans and 4 fully detected EC spans. Not random, not representative; counts do not estimate causes or subgroup prevalence.',
            'evaluation_unit':'Original overlapping stride1 windows of4 raw Llama BPE; a span window contains at least one lexical token belonging to that selected gold span. Counts across spans can overlap. This is not span exact-match F1.',
            'scope':'Existing calibration159 only; raw source_info JSON-parsed only for source12313/15182. No model load/inference/train, no officialtest150, no changed thresholds/labels/source text.',
            'unchanged_calibration_metrics':report['metrics']['calibration'],'cases':cases,'source_trace':trace,
            'input_files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/calibration.jsonl',ROOT/'data/tokens_calibration.jsonl',ROOT/'data/windows_k4_calibration.jsonl',ROOT/'results/full_context_encoder_large_v1/inputs.jsonl',RESULT/(NAME+'.json')]}}
    dump(OUT/'CASE_REVIEW.json',result)
    print(json.dumps({'status':result['status'],'cases':len(cases),'source_trace_exact':2,'summary':[{'id':r['response_id'],'span':r['span_index'],'tp':r['span_windows']['tp'],'n':r['span_windows']['n'],'max':r['span_windows']['score_max'],'answer':r['answer']['score'],'encoder_tokens':r['mapping_check']['encoder_tokens']} for r in cases]},ensure_ascii=False))

if __name__=='__main__':run()
