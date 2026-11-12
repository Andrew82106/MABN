"""Frozen train-only three-signal replay with equal-length source-token interventions.

The intervention is token replacement, not a natural alternative retrieval.
It changes only tokens wholly inside one passage body. No labels or scores,
regeneration, original held-out parsing, head selection, or fitted compression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
R18 = ROOT.parent/'round18_input_information_diagnostics'
R17 = ROOT.parent/'round17_expanded_retraining'
R6 = ROOT.parent/'round6_evidence_grounding'
sys.path.insert(0, str(R17/'src'))
import extract17 as base
import attention7
import lumina7

SOURCE = base.SOURCE
VERSION = 'round19-fixed-length-three-signals-v1'
EXPECTED = 602
MARKER = '\n\nSearch results:\n'
SPLIT_FIELD = re.compile(r'(?<!\\)"split"\s*:\s*"([^"\\]+)"')
SCHEMA = {
    'token_lumina_mmd': 'float32[N,2], original visible passage order; top100 unnormalized cosine-kernel MMD',
    'token_redeep_ecs': 'float32[N,784], layer1..28 major / head0..27 minor; unchanged attention7 ECS',
    'token_redeep_pks': 'float32[N,28], layer1..28; unchanged attention7 standard-JSD correction',
    'token_ids': 'int64[N], frozen original response IDs',
    'response_token_offsets': 'int32[N,2], original full-response character offsets',
    'token_start': 'int32[N]', 'token_end': 'int32[N]',
}


def train_rows(path):
    """Inspect split metadata before JSON parsing; do not parse held-out content."""
    rows = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            matches = list(SPLIT_FIELD.finditer(line))
            assert len(matches) == 1, 'Unambiguous explicit split field required'
            if matches[0].group(1) != 'train':
                continue
            row = json.loads(line)
            assert row['split'] == 'train'
            rows.append(row)
    return rows


def context():
    rows = train_rows(SOURCE/'data/inputs.jsonl')
    assert len(rows) == EXPECTED
    assert len({r['question_id'] for r in rows}) == 301
    assert len({r['group_id'] for r in rows}) == 278
    assert all(len(r['passages']) == 2 and len(r['questions']) == 1 for r in rows)
    r18 = base.read(R18/'data/feature_manifest.json')
    assert r18['complete'] and r18['completed_count'] == EXPECTED
    assert set(r18['records']) == {r['row_id'] for r in rows}
    old_sig = r18['extraction_signature']['base_replay_signature']
    assert base.sha(SOURCE/'data/inputs.jsonl') == old_sig['source_inputs_sha256']
    assert base.sha(SOURCE/'data/input_freeze.json') == old_sig['source_input_freeze_sha256']
    assert base.sha(SOURCE/'data/generation_manifest.json') == old_sig['source_generation_manifest_sha256']
    source = base.read(SOURCE/'data/generation_manifest.json')
    assert source['complete'] and source['generated'] == 800
    records = {}
    for row in rows:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', row['row_id'])
        g, digest = base.generation(row, source)
        assert g['split'] == 'train'
        records[row['row_id']] = (g, digest)
    allowed = set(records)
    for pattern in ('*.json', '*.npz'):
        for p in (ROOT/'data/features').glob(pattern):
            assert p.stem in allowed, ('Not a selected training feature', p)
    return rows, records, r18


def prepare_pool():
    candidates_path = ROOT/'data/donor_candidates.json'
    review_path = ROOT/'data/donor_independence_review.json'
    inventory_path = ROOT/'data/train_visible_inventory.json'
    candidates, review = base.read(candidates_path), base.read(review_path)
    assert review['candidate_file_sha256'] == base.sha(candidates_path)
    assert review['source_inventory_sha256'] == base.sha(inventory_path)
    assert candidates['source_r6_inputs_sha256'] == base.sha(R6/'data/inputs.jsonl')
    assert candidates['source_r16_inputs_sha256'] == base.sha(SOURCE/'data/inputs.jsonl')
    entries = candidates['candidates']
    reviews = {r['donor_id']: r for r in review['records']}
    assert len(reviews) == len(review['records']) == len(entries)
    assert set(reviews) == {r['donor_id'] for r in entries}
    assert all(r['status'] in ('pass', 'exclude') for r in reviews.values())
    # Verify every retained byte comes from a real old training visible passage.
    old_rows = train_rows(R6/'data/inputs.jsonl')
    old_by_id = {r['row_id']: r for r in old_rows}
    accepted = []
    for entry in entries:
        if reviews[entry['donor_id']]['status'] != 'pass':
            continue
        assert entry['source_split'] == 'train'
        for rid in entry['source_rows']:
            assert any(p['title'] == entry['title'] and p['text'] == entry['text']
                       for p in old_by_id[rid]['passages'])
        expected = hashlib.sha256(json.dumps({'title':entry['title'], 'text':entry['text']},
                                  ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert expected == entry['passage_sha256']
        accepted.append({k: entry[k] for k in ('donor_id','title','text','source_split','source_rows','passage_sha256')})
    assert accepted
    value = {'version': VERSION, 'donors': accepted, 'count': len(accepted),
             'deduplication': 'Exact title and text; distinct related old articles may coexist in this common intervention bank',
             'source_r6_inputs_sha256': base.sha(R6/'data/inputs.jsonl'),
             'source_r16_inputs_sha256': base.sha(SOURCE/'data/inputs.jsonl'),
             'candidate_file_sha256': base.sha(candidates_path), 'review_sha256': base.sha(review_path),
             'inventory_sha256': base.sha(inventory_path),
             'review_scope': 'Old R6 actual-train visible sources versus current R16 actual-train main subjects and significant concrete events; no annotation/detector selection',
             'excluded': [r for r in review['records'] if r['status'] == 'exclude']}
    path = ROOT/'data/donor_pool.json'
    if path.exists():
        assert base.read(path) == value, 'Frozen donor pool changed'
    else:
        base.save(path, value)
    return value


def layout(tokenizer, row, g):
    visible = base.visible(row)
    assert visible['prompt'].count(MARKER) == 1
    head, text = visible['prompt'].split(MARKER)
    assert text == lumina7._render_passages(visible['passages'])
    rendered = tokenizer.apply_chat_template(
        [{'role':'system','content':visible['system']}, {'role':'user','content':visible['prompt']}],
        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert list(encoded['input_ids']) == g['input_token_ids']
    start = rendered.find(visible['prompt'])
    assert start >= 0 and rendered.find(visible['prompt'], start+1) < 0
    cursor = start+len(head)+len(MARKER)
    result = []
    for i, passage in enumerate(visible['passages']):
        header = f'[{i+1}] {passage["title"]}\n'
        assert rendered[cursor:cursor+len(header)] == header
        a = cursor+len(header); b = a+len(passage['text'])
        assert rendered[a:b] == passage['text']
        mask = [j for j,(left,right) in enumerate(encoded['offset_mapping'])
                if left >= a and right <= b and right > left]
        assert mask and mask == list(range(mask[0],mask[-1]+1))
        result.append({'passage_index':i, 'title':passage['title'],
                       'rendered_body_character_interval':[a,b],
                       'eligible_body_token_positions':mask})
        cursor = b+2
    assert not set(result[0]['eligible_body_token_positions']) & set(result[1]['eligible_body_token_positions'])
    return result


def donor_stream(tokenizer, question, slot, pool):
    question_sha = base.model7.digest(question)
    def rank(entry):
        key = f'{question_sha}\0{slot}\0{entry["donor_id"]}'.encode('utf-8')
        return hashlib.sha256(key).hexdigest(), entry['donor_id']
    ordered = sorted(pool['donors'], key=rank)
    text, ranges = '', []
    for entry in ordered:
        fragment = entry['title']+'\n'+entry['text']+'\n\n'
        ranges.append({'donor_id':entry['donor_id'], 'title':entry['title'],
                       'character_interval':[len(text),len(text)+len(fragment)]})
        text += fragment
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded['input_ids']), encoded['offset_mapping']
    assert ids and not set(ids) & set(tokenizer.all_special_ids), 'Donor stream contains a special token'
    assert len(ordered) == len({r['donor_id'] for r in ordered}), 'No cycling or repeated donor entry'
    for entry in ranges:
        a,b = entry['character_interval']
        ii = [i for i,(x,y) in enumerate(offsets) if y>x and y>a and x<b]
        assert ii
        entry['overlapping_stream_token_interval'] = [ii[0],ii[-1]+1]
    return {'question_text_sha256':question_sha, 'passage_index':slot, 'token_ids':ids,
            'token_ids_sha256':base.model7.digest(ids), 'donor_ranges':ranges,
            'text_sha256':base.model7.digest(text)}


def validate_plan(row, g, plan):
    assert plan['row_id'] == row['row_id'] and plan['question_text_sha256'] == base.model7.digest(row['questions'][0])
    assert plan['original_prefix_token_ids'] == g['input_token_ids']
    assert plan['response_token_ids'] == g['response_token_ids']
    assert len(plan['interventions']) == 2
    original = np.asarray(g['input_token_ids'])
    for slot, entry in enumerate(plan['interventions']):
        assert entry['passage_index'] == slot
        mask = entry['eligible_body_token_positions']
        replacement = np.asarray(entry['replacement_prefix_token_ids'])
        assert len(original) == len(replacement)
        changes = np.flatnonzero(original != replacement).tolist()
        assert changes == entry['changed_token_positions']
        assert changes and set(changes) <= set(mask)
        assert replacement[mask].tolist() == entry['replacement_body_token_ids']
        assert len(mask) == entry['donor_stream_tokens_used']
        assert entry['donor_stream_total_tokens'] >= len(mask)


def prepare_plans(tokenizer, rows, records, pool):
    path, manifest_path = ROOT/'data/perturbation_plans.jsonl', ROOT/'data/perturbation_manifest.json'
    if path.exists() or manifest_path.exists():
        assert path.exists() and manifest_path.exists(), 'Incomplete intervention-plan commit'
        pm = base.read(manifest_path)
        assert pm['plans_sha256'] == base.sha(path)
        assert pm['donor_pool_sha256'] == base.sha(ROOT/'data/donor_pool.json')
        assert pm['source_inputs_sha256'] == base.sha(SOURCE/'data/inputs.jsonl')
        assert pm['extract19_sha256'] == base.sha(Path(__file__))
        plans = [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
        assert [p['row_id'] for p in plans] == [r['row_id'] for r in rows]
        for row, plan in zip(rows, plans):
            g, digest = records[row['row_id']]
            assert plan['source_generation_sha256'] == digest
            validate_plan(row,g,plan)
        return {p['row_id']:p for p in plans}
    streams, plans = {}, []
    for row in rows:
        g,digest = records[row['row_id']]
        q = row['questions'][0]; qsha = base.model7.digest(q)
        interventions = []
        for spec in layout(tokenizer,row,g):
            slot = spec['passage_index']; key = (qsha,slot)
            if key not in streams:
                streams[key] = donor_stream(tokenizer,q,slot,pool)
            stream = streams[key]; mask = spec['eligible_body_token_positions']; n = len(mask)
            if n > len(stream['token_ids']):
                raise ValueError(f'Donor stream too short: {row["row_id"]} slot{slot}; no cycling')
            replacement = list(g['input_token_ids']); used = stream['token_ids'][:n]
            for position,token in zip(mask,used):
                replacement[position] = token
            changes = [i for i,(a,b) in enumerate(zip(g['input_token_ids'],replacement)) if a!=b]
            donors = []
            for entry in stream['donor_ranges']:
                a,b = entry['overlapping_stream_token_interval']
                if a >= n:
                    break
                donors.append({**entry, 'used_stream_token_interval':[a,min(b,n)]})
            interventions.append({**spec, 'replacement_prefix_token_ids':replacement,
                'replacement_body_token_ids':used, 'changed_token_positions':changes,
                'replacement_prefix_sha256':base.model7.digest(replacement),
                'donor_stream_total_tokens':len(stream['token_ids']), 'donor_stream_tokens_used':n,
                'donor_stream_sha256':stream['token_ids_sha256'], 'donor_stream_text_sha256':stream['text_sha256'],
                'used_donors':donors})
        plan = {'row_id':row['row_id'], 'actual_split':'train',
                'source_generation_sha256':digest, 'question_text_sha256':qsha,
                'original_prefix_token_ids':g['input_token_ids'], 'response_token_ids':g['response_token_ids'],
                'interventions':interventions}
        validate_plan(row,g,plan); plans.append(plan)
    # Same question and slot must consume the same stream prefix in both conditions.
    prefixes = {}
    for plan in plans:
        for entry in plan['interventions']:
            key = (plan['question_text_sha256'],entry['passage_index'])
            tokens = entry['replacement_body_token_ids']
            if key in prefixes:
                other = prefixes[key]; n = min(len(tokens),len(other)); assert tokens[:n] == other[:n]
            else:
                prefixes[key] = tokens
    pending = path.with_suffix('.jsonl.pending')
    pending.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in plans),'utf-8'); pending.replace(path)
    base.save(manifest_path, {'version':VERSION, 'complete':True, 'rows':len(plans), 'interventions':2*len(plans),
        'plans_sha256':base.sha(path), 'donor_pool_sha256':base.sha(ROOT/'data/donor_pool.json'),
        'source_inputs_sha256':base.sha(SOURCE/'data/inputs.jsonl'), 'extract19_sha256':base.sha(Path(__file__)),
        'tokenizer_config_sha256':base.sha(base.model7.MODEL/'tokenizer_config.json'),
        'mask':'Only nonempty original prompt tokens with a>=body_start and b<=body_end; crossing-boundary tokens unchanged',
        'stream':'Order unique reviewed donors by SHA256(question_text_SHA256 + NUL + zero-based passage index + NUL + donor_id); concatenate title+newline+body+two newlines, tokenize once without specials, take first mask-count IDs; no recycling',
        'constant':'Question, title and all outside-mask IDs; original prompt token count; exact answer IDs, positions and offsets',
        'two_conditions':'Same question text and passage slot use same ordered stream; consume different prefix lengths if needed',
        'natural_text_claim':False, 'labels_or_detector_scores_read':False})
    print('PERTURBATION_PLANS_FROZEN',len(plans), 'DONORS',pool['count'],flush=True)
    return {p['row_id']:p for p in plans}


def signature():
    code = (Path(__file__), Path(base.__file__), Path(attention7.__file__), Path(lumina7.__file__), Path(base.model7.__file__))
    return {'version':VERSION, 'code_sha256':{str(p.resolve()):base.sha(p) for p in code},
        'base_model_and_source_signature':base.signature(),
        'r18_feature_manifest_sha256':base.sha(R18/'data/feature_manifest.json'),
        'donor_pool_sha256':base.sha(ROOT/'data/donor_pool.json'),
        'donor_review_sha256':base.sha(ROOT/'data/donor_independence_review.json'),
        'perturbation_manifest_sha256':base.sha(ROOT/'data/perturbation_manifest.json'),
        'perturbation_plans_sha256':base.sha(ROOT/'data/perturbation_plans.jsonl'),
        'schema':SCHEMA, 'actual_split':'train', 'expected_rows':EXPECTED,
        'lumina_top_k':lumina7.TOP_K, 'lumina_chunk_size':lumina7.DEFAULT_CHUNK_SIZE,
        'lumina_formula_version':lumina7.FORMULA_VERSION,
        'attention_version':attention7.ATTENTION7_VERSION,
        'no_head_or_layer_selection':True, 'no_fitted_compression':True}


def compare_lb(row,g,digest,lb,r18):
    rec = r18['records'][row['row_id']]; path = R18/rec['npz']; side = R18/rec['json']
    assert base.sha(path) == rec['npz_sha256'] and base.sha(side) == rec['json_sha256']
    meta = base.read(side)
    assert meta['source_generation_sha256'] == digest and rec['source_generation_sha256'] == digest
    with np.load(path,allow_pickle=False) as old:
        assert old['token_ids'].tolist() == g['response_token_ids']
        assert old['response_token_offsets'].tolist() == g['response_token_offsets']
        delta = float(np.abs(lb-old['lb']).max())
        assert np.array_equal(lb,old['lb']), ('Original LB differs from R18; do not relax silently',row['row_id'],delta)
    return {'passed':True,'max_absolute_difference':delta,'all_response_tokens':len(g['response_token_ids']),
            'r18_npz_sha256':rec['npz_sha256'],'r18_json_sha256':rec['json_sha256']}


def stats(model,prefix,answer):
    hidden,_ = lumina7.collect_response_hidden(model,prefix,answer,capture_layers=False)
    result = lumina7.final_statistics(model,hidden,answer)
    del hidden
    return result


def mmd(model,p,q):
    return lumina7.cosine_mmd_from_topk(p['top_probs'],p['top_ids'],q['top_probs'],q['top_ids'],
                                       model.get_input_embeddings()).numpy()


def validate(arrays,g):
    n = len(g['response_token_ids'])
    assert set(arrays) == set(SCHEMA)
    for key,width in [('token_lumina_mmd',2),('token_redeep_ecs',784),('token_redeep_pks',28)]:
        assert arrays[key].shape == (n,width) and arrays[key].dtype == np.float32
        assert np.isfinite(arrays[key]).all()
    assert np.all(arrays['token_lumina_mmd'] >= 0)
    assert np.all(np.abs(arrays['token_redeep_ecs']) <= 1+1e-5)
    assert np.all(arrays['token_redeep_pks'] >= -1e-6)
    assert np.all(arrays['token_redeep_pks'] <= np.log(2)+1e-5)
    assert arrays['token_ids'].dtype == np.int64 and arrays['token_ids'].tolist() == g['response_token_ids']
    assert arrays['response_token_offsets'].dtype == np.int32
    assert arrays['response_token_offsets'].tolist() == g['response_token_offsets']
    for key,column in [('token_start',0),('token_end',1)]:
        assert arrays[key].dtype == np.int32 and np.array_equal(arrays[key],arrays['response_token_offsets'][:,column])


@torch.inference_mode()
def extract(tokenizer,model,row,g,digest,plan,r18,identity_check=False):
    assert row['split'] == 'train'
    validate_plan(row,g,plan)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    visible = dict(base.visible(row),row_id=row['row_id'])
    # Empty item list prevents parser-dependent auxiliary aggregation. Raw token
    # signals are identical and all tokens survive, including malformed outputs.
    generation = {key:g[key] for key in ('input_token_ids','response_token_ids','response','response_token_offsets')}
    generation.update(row_id=row['row_id'],items=[])
    attention, attention_meta = attention7.extract_attention(tokenizer,model,visible,generation)
    lb_check = compare_lb(row,g,digest,attention['token_lookback'],r18)
    original = stats(model,g['input_token_ids'],g['response_token_ids'])
    values = []
    for intervention in plan['interventions']:
        altered = stats(model,intervention['replacement_prefix_token_ids'],g['response_token_ids'])
        values.append(mmd(model,original,altered)); del altered
    identity = None
    if identity_check:
        clone = list(g['input_token_ids'])
        for position in plan['interventions'][0]['eligible_body_token_positions']:
            clone[position] = g['input_token_ids'][position]
        assert clone == g['input_token_ids']
        unchanged = stats(model,clone,g['response_token_ids'])
        identity_values = mmd(model,original,unchanged)
        identity = {'max_abs_mmd':float(np.abs(identity_values).max()),
                    'top100_probabilities_exactly_equal':torch.equal(original['top_probs'],unchanged['top_probs']),
                    'top100_token_ids_exactly_equal':torch.equal(original['top_ids'],unchanged['top_ids'])}
        assert identity['max_abs_mmd'] <= 1e-12
        assert identity['top100_probabilities_exactly_equal'] and identity['top100_token_ids_exactly_equal']
    offsets = np.asarray(g['response_token_offsets'],dtype=np.int32)
    arrays = {'token_lumina_mmd':np.stack(values,axis=1),
        'token_redeep_ecs':attention['token_redeep_ecs'], 'token_redeep_pks':attention['token_redeep_pks'],
        'token_ids':np.asarray(g['response_token_ids'],dtype=np.int64), 'response_token_offsets':offsets,
        'token_start':offsets[:,0], 'token_end':offsets[:,1]}
    validate(arrays,g); torch.cuda.synchronize()
    meta = {'version':VERSION, 'row_id':row['row_id'], 'actual_split':'train',
        'response_tokens':len(g['response_token_ids']), 'input_tokens':len(g['input_token_ids']),
        'attention7_metadata':attention_meta, 'original_lb_r18_check':lb_check,
        'intervention_plan':plan, 'intervention_plan_sha256':base.model7.digest(plan),
        'lumina_definition':'Only unchanged top100 unnormalized cosine-kernel MMD; no IPR or mixed LUMINA score',
        'mmd_timing':'Pre-read P+j-1 distribution for frozen token j; original and each equal-length single-body intervention',
        'ecs_pks_timing':'Unchanged attention7 post-read P+j; final-RMSNorm ECS and standard FFN JSD',
        'baseline_forward_passes':2+1+2, 'identity_extra_forward_passes':int(identity_check),
        'identity_check':identity, 'seconds':time.perf_counter()-started,
        'peak_allocated_gib':float(torch.cuda.max_memory_allocated()/2**30),
        'labels_or_detector_scores_read':False, 'response_regenerated':False,
        'original_heldout_content_parsed':False, 'schema':SCHEMA}
    return arrays,meta


def expected(row,digest,plan,sig):
    return {'row_id':row['row_id'],'source_generation_sha256':digest,
            'input_row_sha256':base.model7.digest(row), 'intervention_plan_sha256':base.model7.digest(plan),
            'extraction_signature_sha256':base.model7.digest(sig)}


def cached(row,g,digest,plan,sig):
    path = ROOT/'data/features'/(row['row_id']+'.json')
    if not path.exists():
        return None
    meta = base.read(path)
    for key,value in expected(row,digest,plan,sig).items():
        assert meta[key] == value,(row['row_id'],key)
    assert base.sha(path.with_suffix('.npz')) == meta['arrays_sha256']
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:
        arrays = {key:data[key] for key in data.files}
    validate(arrays,g)
    return arrays,meta


def store_row(row,g,digest,plan,arrays,meta,sig):
    assert signature() == sig, 'Frozen source/code/donor/intervention signature changed'
    assert base.sha(SOURCE/'data/generation_records'/(row['row_id']+'.json')) == digest
    path = ROOT/'data/features'/(row['row_id']+'.npz')
    base.save_arrays(path,arrays)
    meta.update(expected(row,digest,plan,sig),arrays_sha256=base.sha(path),extraction_signature=sig)
    base.save(path.with_suffix('.json'),meta)


def manifest(rows,records,plans,sig):
    entries,missing = {},[]
    for row in rows:
        g,digest = records[row['row_id']]; value = cached(row,g,digest,plans[row['row_id']],sig)
        if value is None:
            missing.append(row['row_id']); continue
        _,meta = value; relative = 'data/features/'+row['row_id']
        entries[row['row_id']] = {'json':relative+'.json','json_sha256':base.sha(ROOT/(relative+'.json')),
            'npz':relative+'.npz','npz_sha256':meta['arrays_sha256'], 'source_generation_sha256':digest,
            'response_tokens':meta['response_tokens'],'split':'train',
            'original_lb_r18_max_absolute_difference':meta['original_lb_r18_check']['max_absolute_difference']}
    value = {'version':VERSION,'complete':not missing,'expected_rows':EXPECTED,
        'completed_count':len(entries),'completed_rows':len(entries),'generated':len(entries),
        'records':entries,'missing_rows':missing,'schema':SCHEMA,
        'total_response_tokens_completed':sum(r['response_tokens'] for r in entries.values()),
        'selected_questions':301,'selected_group_ids':278,'actual_split':'train',
        'validation_or_test_rows_extracted':0,'all_training_outputs_including_refusals_and_parse_failures':True,
        'source_root':str(SOURCE.resolve()),'extraction_signature':sig,
        'extraction_signature_sha256':base.model7.digest(sig),
        'labels_or_detector_scores_read':False,'response_regenerated':False,
        'intervention_is_natural_retrieval_text':False,
        'limitations':['MMD measures sensitivity to a fixed artificial equal-token-count intervention, not truth or entailment.',
                      'ECS and PKS preserve previously adapted definitions; no claim that high/low raw values are calibrated hallucination risks.',
                      'ECS/PKS are post-read while MMD distribution is pre-read; the supervisor must preserve this timing statement.']}
    base.save(ROOT/'data/feature_manifest.json',value)
    return value


def selfcheck(tokenizer,model,rows,records,plans,r18,sig):
    first = rows[0]['question_id']; pair = [r for r in rows if r['question_id']==first]
    assert len(pair)==2
    checks=[]
    for row in pair:
        g,digest = records[row['row_id']];plan = plans[row['row_id']]
        arrays,meta = extract(tokenizer,model,row,g,digest,plan,r18,identity_check=True)
        repeat,_ = extract(tokenizer,model,row,g,digest,plan,r18)
        delta = {key:float(np.abs(arrays[key].astype(np.float64)-repeat[key].astype(np.float64)).max()) for key in arrays}
        assert all(np.array_equal(arrays[key],repeat[key]) for key in arrays), ('Repeated full replay differs',row['row_id'],delta)
        store_row(row,g,digest,plan,arrays,meta,sig)
        checks.append({'row_id':row['row_id'],'source_generation_sha256':digest,
            'original_lb_r18_check':meta['original_lb_r18_check'],'identity_check':meta['identity_check'],
            'repeat_max_absolute_differences':delta,'shapes':{k:list(v.shape) for k,v in arrays.items()},
            'only_body_mask_changed':True,'prefix_length_fixed':True,'answer_ids_and_offsets_fixed':True})
    base.save(ROOT/'data/extraction_selfcheck.json',{'passed':True,'checks':checks,
        'extraction_signature_sha256':base.model7.digest(sig),'same_question_slot_stream_prefix_across_conditions':True,
        'labels_or_detector_scores_read':False,'response_regenerated':False})
    print('SELFCHECK_PASSED',json.dumps(checks,ensure_ascii=False),flush=True)


def run(prepare_only=False,audit_only=False,selfcheck_only=False):
    rows,records,r18 = context();pool = prepare_pool()
    tokenizer = AutoTokenizer.from_pretrained(base.model7.MODEL,local_files_only=True)
    plans = prepare_plans(tokenizer,rows,records,pool);sig = signature()
    if prepare_only or audit_only:
        result = manifest(rows,records,plans,sig)
        print('PREPARED' if prepare_only else 'AUDIT',result['completed_count'],EXPECTED,result['complete'],flush=True)
        return
    model = None; check_path = ROOT/'data/extraction_selfcheck.json'
    if check_path.exists():
        check = base.read(check_path)
        assert check['passed'] and check['extraction_signature_sha256']==base.model7.digest(sig)
    else:
        tokenizer,model = base.model7.load_model();assert getattr(model,'is_loaded_in_4bit',False)
        selfcheck(tokenizer,model,rows,records,plans,r18,sig)
    if selfcheck_only:
        manifest(rows,records,plans,sig);return
    started = time.perf_counter()
    for i,row in enumerate(rows,1):
        g,digest = records[row['row_id']];plan=plans[row['row_id']]
        if cached(row,g,digest,plan,sig) is not None:
            continue
        if model is None:
            tokenizer,model=base.model7.load_model();assert getattr(model,'is_loaded_in_4bit',False)
        arrays,meta=extract(tokenizer,model,row,g,digest,plan,r18)
        store_row(row,g,digest,plan,arrays,meta,sig)
        print('DONE',i,EXPECTED,row['row_id'],meta['response_tokens'],round(meta['seconds'],3),flush=True)
        if i%100==0:
            manifest(rows,records,plans,sig)
    assert signature()==sig
    result=manifest(rows,records,plans,sig)
    print('COMPLETE',result['completed_count'],EXPECTED,result['complete'],
          'elapsed_seconds',round(time.perf_counter()-started,3),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--selfcheck-only',action='store_true')
    args=parser.parse_args()
    run(prepare_only=args.prepare_only,audit_only=args.audit,selfcheck_only=args.selfcheck_only)
