"""Serialize root's item-by-item evidence review, never infer labels from scores.

All 100 generated answers and the 50 paired complete/partial source bundles
were read before these explicit decisions were written. This is assistant
annotation, not independent human gold. Reference correctness concerns the
stored reference, not independent verification of present-day world facts.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# S=supported assertion, U=unsupported assertion, C=contradicted assertion,
# A=explicit abstention, X=ambiguous assertion/abstention, retained unresolved.
# These are reviewed decisions about actual text, not a missing-context rule.
PARTIAL = {
    41: ('U', 'incorrect', 'Transfers X Money from the unrelated X platform to Dashen Bank. Full reference names Dashen Bank SuperApp.'),
    42: ('A', 'not_applicable', 'Explicitly says the fleet size is absent from the search results.'),
    78: ('U', 'incorrect', 'Attributes Kpop4planet carbon-emissions campaigning against Dior/YSL to Kwek and his lawsuit. The source contains no such Kwek lawsuit motive.'),
    114: ('A', 'not_applicable', 'Explicitly reports no source information about the South Carolina candidates.'),
    120: ('A', 'not_applicable', 'Says the results do not provide Rabouin vote share.'),
    122: ('A', 'not_applicable', 'Says the Sorocaba project objectives are not provided.'),
    127: ('A', 'not_applicable', 'Says the Gikomba workforce number is unspecified in the results; makes no numerical assertion.'),
    164: ('A', 'not_applicable', 'Says the results lack the Indian national description.'),
    166: ('U', 'incorrect', 'Transfers the Maldives amendment-procedure dispute to Bhuiyan. Full reference attributes his view to the government being unelected.'),
    175: ('U', 'unresolved', 'Invents visual-representation campaigning by Mrs Frank Crowell/NSDU; sources only give founding/history. Also fails to answer the EDU/Bambie question. Full reference does not independently disprove every new NSDU claim.'),
    188: ('U', 'unresolved', 'Invents LG collaboration with Manus. Manus architecture description is sourced, but the relationship to LG is absent. Full reference names Google Cloud without establishing an exhaustive list of all LG partners.'),
    207: ('U', 'unresolved', 'Transfers Human Rights Watch from Yemeni detainees to these Iranian lawyers. Full reference identifies HRANA; it does not independently establish whether HRW ever also reported this event.'),
    249: ('U', 'correct', 'Mismanagement/corruption suspicion matches the complete reference, but current Queensland-flood material does not support Aliasgar allegations; question supplies an inquiry, not the specific wrongdoing.'),
    251: ('U', 'incorrect', 'Transfers the railway passenger-service late-February2025 schedule to the ship. Full reference expected earlyJune2025.'),
    266: ('A', 'not_applicable', 'Explicitly says Azuka resting place is not mentioned.'),
    267: ('A', 'not_applicable', 'No train number asserted; states source information is unavailable.'),
    272: ('A', 'not_applicable', 'Explicitly withholds Johnson general-election margin.'),
    330: ('A', 'not_applicable', 'States lack of ban information and inability to determine duration.'),
    441: ('A', 'not_applicable', 'Explicitly reports no relevant MBH shareholding data.'),
    459: ('A', 'not_applicable', 'Reports no Yukon/mining-period data; does not substitute Australian emissions.'),
    490: ('U', 'incorrect', 'Invents 48-year Guam majority interval using unrelated election information; complete reference says16years.'),
    547: ('U', 'unresolved', 'Transfers a Mongolian-film USD320000 total budget/project-market funding to Storytailor. Full reference supports NMotion100000, but does not establish total funding from every source.'),
    582: ('U', 'incorrect', 'Transfers October2023 from Dhaka university events to Bulawayo boarding intake; full reference planned January2025.'),
    617: ('A', 'not_applicable', 'Explicitly says Azubuike demise is not mentioned; does not assign Abraham death circumstances.'),
    625: ('A', 'not_applicable', 'Explicitly says Trudeau departure date is unavailable in the supplied results.'),
    676: ('U', 'unresolved', 'Transfers AMI Paris700points-of-sale2024 to Unioil. Full reference provides Unioil165stations inFebruary2025, not an exhaustive2024 distribution history.'),
    774: ('U', 'incorrect', 'Treats two named contenders in another election as the total Kawempe candidate count; full reference says10.'),
    853: ('U', 'incorrect', 'Invents the Department of the Premier and Cabinet portfolio relationship and reverses the asked administration relation. Complete source identifies the Women/Aboriginal/TorresStrait/Multiculturalism department.'),
    965: ('U', 'incorrect', 'Attributes another company Innoscience unprofitability/international expansion as Manycore IPO motive; full reference states the technology-sector crackdown.'),
    1029: ('U', 'unresolved', 'Invents a Twenty Ideas link to SNARF. The supplied snippet explains SNARF aims without linking it to Twenty Ideas; complete reference supports MVP/user-feedback/market-validation work.'),
    1066: ('U', 'incorrect', 'Transfers the unrelated Collective party to Holli Willibey; complete candidate list gives Republican.'),
    1099: ('A', 'not_applicable', 'States no parade slogan was found in the supplied sources.'),
    1253: ('U', 'incorrect', 'Reattributes a different Syrian organization ultimatum to the presidential declaration. Complete reference concerns coordinated national-security/political response.'),
    1258: ('U', 'incorrect', 'Transfers Sumit Gupta from an Indian crypto council to SASC; complete source names Sheikh Hamdan.'),
    1295: ('U', 'incorrect', 'Transfers A16 iPad March4,2025 announcement to M5 iPadPro release; full source expectation is early2026.'),
    1346: ('U', 'incorrect', 'Transfers TalkMoney financial-literacy goal to Ajebo Jeni/Keni; full source states child education/entertainment.'),
    1371: ('U', 'unresolved', 'Transfers Hero of the Russian Federation from Time of Heroes participants to Maslovsky. No visible source supports the award for him; complete reference lists Ukrainian awards without an exhaustive all-award history.'),
    1551: ('A', 'not_applicable', 'Explicitly says Qudah appointment information is absent and declines to answer.'),
    1557: ('X', 'unresolved', 'Does not provide specific functionality ... based on the given information can mean a negative claim about the plan or an awkward abstention. Retain unresolved pending independent semantic review; no keyword rule decides it.'),
    1558: ('A', 'not_applicable', 'Explicitly reports no information about the Southeast European dialogue groups.'),
    1565: ('U', 'incorrect', 'Transfers Borgwardt alleged obstruction to Huttle driving prohibition; full reference says habitual traffic violator.'),
    1611: ('A', 'not_applicable', 'Explicitly reports no Jamini Sen portrait information in the supplied sources.'),
    1633: ('U', 'incorrect', 'Reattributes Treasury aims concerning Syrian drug trafficking to Ansar Allah own reaction. Full reference attributes pathetic/powerless dismissal to the group.'),
    1703: ('U', 'unresolved', 'Names the unrelated North American blizzard as cause of the Chinese vessel grounding. Full source supports a strong storm near Sakhalin but does not name the storm; do not turn source absence into independent meteorological disproof.'),
    1710: ('U', 'incorrect', 'Transfers an environmental researcher water-quality methods to WAVES creative-economy goal; full source supports media/entertainment partnerships and prominence.'),
    1858: ('U', 'incorrect', 'Invents Sexton spouse Montserrat and1940death by combining Schilling spouse/daughter details; full source says Sexton wife died2024.'),
    1994: ('U', 'incorrect', 'Asserts Microsoft produced no Majorana1information in the press statements; this is a negative real-world claim, not a statement that the supplied snippets lack it. Full source explicitly reports those statements.'),
    2027: ('A', 'not_applicable', 'Explicitly says Jacomar distinction information is absent.'),
    2309: ('A', 'not_applicable', 'Explicitly declines to give the Bakassi superhighway distance.'),
    2337: ('U', 'incorrect', 'Transfers LeapmotorB10April2025 deliveries to JetourFreedom sales; complete reference says February2025.'),
}
COMPLETE_EXCEPTIONS = {
    127: ('C', 'incorrect', 'Uses the earlier65000estimate as official2024workforce, contradicting the source explicit2024government100000figure.'),
    175: ('U', 'unresolved', 'Bambie Thug identity/campaign imagery is supported, but the extra performance title Evilution is absent from every visible and full-reference passage. No independent world-fact verification of that added title.'),
}


def readl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    area = ROOT/'data/external_ragognize'
    refs = {r['question_id']: r for r in readl(area/'external_references.jsonl')}
    inputs = readl(area/'external_inputs.jsonl')
    assert len(PARTIAL) == len(refs) == 50 and len(inputs) == 100
    records, generation_hashes = [], {}
    mapping = {'S': ('asserted', 'supported', 0), 'U': ('asserted', 'unsupported', 1),
               'C': ('asserted', 'contradicted', 1), 'A': ('abstained', 'not_applicable', None),
               'X': ('asserted', 'unresolved', None)}
    for row in inputs:
        path = ROOT/'data/generation_records'/(row['row_id']+'.json')
        generated = json.loads(path.read_text(encoding='utf-8'))
        generation_hashes[path.name] = sha(path)
        assert len(generated['items']) == 1
        item = generated['items'][0]
        assert item['parse_ok'] and generated['response'][item['start']:item['end']] == item['text']
        number = int(row['source_question_id'])
        if row['condition'] == 'partial':
            code, correctness, reason = PARTIAL[number]
        else:
            code, correctness, reason = COMPLETE_EXCEPTIONS.get(number, ('S', 'correct',
                'Reviewed actual generated statement against all visible complete passages: the asserted target and any stated attributes are supported; no unsupported added fact identified.'))
        stance, relation, risk = mapping[code]
        ref = refs[row['question_id']]
        rec = {**item, 'row_id': row['row_id'], 'question_id': row['question_id'],
               'group_id': row['group_id'], 'split': 'external_test', 'dataset': 'RAGognize',
               'condition': row['condition'], 'stance': stance, 'evidence_relation': relation,
               'risk': risk, 'reference_correctness': correctness, 'rationale': reason,
               'reference_evidence': ref['items'][0]['evidence'],
               'visible_source_titles': [p['title'] for p in row['passages']],
               'multi_claim': (number == 175 and row['condition'] == 'complete') or
                              (number in (78, 188, 1858) and risk is not None),
               'annotation_method': 'Root assistant item-by-item evidence review, not independent human gold',
               'source_generation_sha256': generation_hashes[path.name],
               'detector_scores_used': False}
        if number == 175 and row['condition'] == 'complete':
            word = 'Evilution'
            local = item['text'].index(word)
            rec['risk_spans'] = [{'start': item['start']+local, 'end': item['start']+local+len(word), 'text': word,
                                  'note': 'Unsupported extra named performance within an otherwise supported answer.'}]
        elif number == 127 and row['condition'] == 'complete':
            word = '65,000'
            local = item['text'].index(word)
            rec['risk_spans'] = [{'start': item['start']+local, 'end': item['start']+local+len(word), 'text': word,
                                  'note': 'Wrong count for the stated2024time and government source.'}]
        records.append(rec)
    out = ROOT/'data/annotations_external_test.jsonl'
    out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    manifest = {'status': 'assistant_initial_annotation_pending_second_review', 'items': len(records),
                'risk_items': sum(r['risk'] == 1 for r in records), 'supported_items': sum(r['risk'] == 0 for r in records),
                'abstentions': sum(r['stance'] == 'abstained' for r in records),
                'unresolved': [r['item_id'] for r in records if r['evidence_relation'] == 'unresolved'],
                'annotations_sha256': sha(out), 'source_generation_hashes': generation_hashes,
                'input_hash': sha(area/'external_inputs.jsonl'), 'reference_hash': sha(area/'external_references.jsonl'),
                'annotation_code_sha256': sha(Path(__file__)), 'scores_consulted': False, 'human_gold': False}
    (ROOT/'data/external_annotation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'source_generation_hashes'}))


if __name__ == '__main__':
    main()
