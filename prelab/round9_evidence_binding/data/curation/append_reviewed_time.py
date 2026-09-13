"""Serialize manually chosen source spans after full-article reading; no model use."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
pool = {r['candidate_id']: r for r in map(json.loads, (P/'source_pool.jsonl').read_text(encoding='utf-8').splitlines())}
out = json.loads((P/'time_quantity_candidates.json').read_text(encoding='utf-8'))
existing = {r['candidate_id'] for r in out}
review = json.loads((P/'time_quantity_review.json').read_text(encoding='utf-8'))

def add(n, question, subject, answer, common, evidence, partial, rationale):
    cid = f'ragognize_train_{n:04d}'
    if cid in existing:
        return
    src = pool[cid]
    text = src['source_content']
    def extract(value):
        if isinstance(value, str):
            assert text.count(value) == 1, (cid, value)
            return value
        start, end = value
        assert text.count(start) == 1, (cid, start)
        i = text.index(start)
        j = text.index(end, i) + len(end)
        quote = text[i:j]
        assert text.count(quote) == 1
        return quote
    row = dict(candidate_id=cid, category='time', question=question, subjects=[subject], reference_answer=answer, answer_aliases=[], common_quote=extract(common), evidence_quote=extract(evidence), partial_quote=extract(partial), rationale=rationale, question_rewrite_reason='Neutral single-fact question; removed confirmation wording, false suggestions, or answer hints where present. Preserve the source tense and qualification.', status='source_reviewed_candidate')
    out.append(row)
    review['reviewed_full_article_ids'].append(cid)

add(1187, 'In what month and year did Pravesh Ratn join the Aam Aadmi Party?', 'Pravesh Ratn', 'December 2024',
    ['Pravesh Ratn (born 1984)', 'in New Delhi district.'],
    'Later in December 2024, he shifted to the Aam Aadmi Party after being with BJP for over a decade.',
    ['He studied Class 12', 'his wife runs a fitness centre.'],
    'Constituency, education and occupation remain. Only the evidence dates the party switch; the retained facts do not establish its month.')
add(2101, "For which season was the inaugural Saudi Women's Super Cup scheduled?", "Saudi Women's Super Cup", '2025–26 season',
    ['The Saudi Women\'s Super Cup (Arabic:', 'the Premier League and the SAFF Cup.'],
    'The inaugural edition is scheduled for the 2025–26 season.',
    'The competition will be held in a centralized format, where four teams will compete in three matches (two semi-finals and a final).',
    'The identity, qualification criteria and match format remain; none dates the inaugural edition. The answer remains a schedule, not an assertion that the edition occurred.')
add(867, 'On what date was Bill Elam sworn in to the Alaska House of Representatives?', 'Bill Elam', 'January 21, 2025',
    ['Bill Elam is an American', '8th district since 2025.'],
    'He was sworn in on January 21, 2025.',
    'A Republican, he previously served two terms in the Kenai Peninsula Borough Assembly.',
    'The introduction gives only the service year, while the replacement describes a prior office; only evidence supplies the swearing-in day.')
add(1875, 'In what month and year was construction of The Castings completed?', 'The Castings', 'June 2024',
    'The Castings is a 78 m (256 ft) tall, 25-storey residential tower containing 352 apartments in Manchester, England.',
    'Construction of the building was completed in June 2024.',
    'It is located in Piccadilly as part of the Portugal Street East regeneration area.',
    'Building height, capacity and location cannot establish its construction completion month.')
add(519, 'On what date was The Safe House scheduled for release in French cinemas?', 'The Safe House', '19 March 2025',
    ['The Safe House (French:', 'Michel Blanc and Dominique Reymond.'],
    'The film will be released in the French cinemas on 19 March 2025 by The Diamond Films.',
    'In January 2025, France’s mk2 Films acquired the international sales rights of the film.',
    'The replacement dates an international sales-rights transaction, not the French cinema release. The shared film identity does not supply a release date.')
add(623, "On what date was the 2025 Colombian Women's Football League scheduled to end?", "2025 Colombian Women's Football League", '7 September 2025',
    ['The 2025 Colombian Women\'s Football League (officially', "Colombia's women's football league."],
    'The season began on 21 February and is scheduled to end on 7 September 2025.',
    'Both league finalists will qualify for the 2025 Copa Libertadores Femenina.',
    'Qualification for another tournament and the league identity do not specify the scheduled final day.')
add(1153, "On what date did Vaitiare Pardo make her senior debut for Chile?", 'Vaitiare Pardo', '22 February 2025',
    ['Vaitiare Anahí Pardo Pinilla (born', "the Chile women's national team."],
    ['At senior level, she received her first call-up', 'She made her debut in the first match on 22 February.'],
    'She is the daughter of Sebastián Pardo and the niece of Mauricio Pinilla, both former Chile international footballers.',
    'The two evidence sentences jointly resolve senior level, February 2025 and the exact debut day. Family connections and club identity in partial cannot date the debut.')
add(1122, "On what date was Erla Harewood-Christopher removed as Trinidad and Tobago's police commissioner?", 'Erla Harewood-Christopher', 'January 31, 2025',
    'Erla Harewood-Christopher (b. May 14, 1963) is a Trinidadian and Tobagonian police officer.',
    ["Harewood-Christopher was appointed Trinidad and Tobago's first female police commissioner", 'She was removed on January 31, 2025.'],
    ['Erla Harewood-Christopher graduated from the University of the West Indies', 'the University of Cambridge.'],
    'The evidence explicitly identifies the commissioner office before the removal statement. Biography and degrees retained in partial contain no removal date.')
add(2036, "During which season did Mikey Ghossaini begin training regularly with Melbourne City's first team?", 'Mikey Ghossaini', 'the 2024 season',
    ['Michael Ghossaini (born', 'Australian club Melbourne City.'],
    'He began training regularly with the first team during the 2024 season.',
    'Born in Australia, Ghossaini is of Lebanese descent and is eligible to play for both Australia and Lebanon at the international level.',
    'Source title Mikey and introduction Michael resolve the same player. National eligibility does not date the start of first-team training.')
add(1149, 'On what date did Evy Pereira announce her departure from Racing Power?', 'Evy Pereira', '8 July 2024',
    ['In July 2022, Evy moved to Campeonato Nacional II Divisão side Racing Power', "second-division title-winning campaign."],
    'On 8 July 2024, She announced her departure from the club.',
    "On 17 July 2024, Turkcell Women's Football Super League club Beşiktaş announced the signing of Evy.",
    'Common context identifies Racing Power and the source title identifies Evy Pereira. A later signing and the approximate contract period do not establish the exact departure-announcement day.')
add(1347, 'When was production at the Madsen Mine expected to restart following West Red Lake Gold’s announcement?', 'Madsen Mine', 'later in 2025',
    'The Madsen Mine is a past-producing Canadian underground gold mine, currently owned by West Red Lake Gold Mines.',
    'On January 2, 2025, West Red Lake Gold announced that they would restart the Madsen Mine, with production beginning later in 2025.',
    'In 1934, Marius Madsen staked claims of the property of the future Madsen Mine for the Falcon Gold Syndicate.',
    'The target is the planned new production period, not historic claims. The other full-source reopening sentence is omitted from both retained snippets. Preserve the imprecise later-in-2025 wording.')
add(2077, 'What dates were planned for the Freedom Shield joint exercise associated with the 2025 Pocheon bombing?', '2025 Pocheon bombing', '10 to 20 March 2025',
    ['On 6 March 2025, two South Korean Air Force KF-16 jets', 'close to a bombing range.'],
    ['The live-fire exercise at the Seungjin Science and Technology Training Center', 'on 10 to 20 March.'],
    ['After the bombing, residents of Pocheon were evacuated', 'in the aftermath of the bombing.'],
    'The evidence dates the future Freedom Shield exercise; the common bombing date and evacuation/suspension aftermath do not establish that separate schedule.')
add(1307, 'On what date did police hand over Prakriti Lamsal’s body to her family following the autopsy?', 'Prakriti Lamsal', 'February 18, 2025',
    'Prakriti Lamsal was a student from Siddharthanagar, Bhairahawa, in the Rupandehi District of Nepal.',
    'On February 18, 2025, police handed over the body of Prakriti Lamsal to her family following an autopsy.',
    'She was pursuing her higher studies at the Kalinga Institute of Industrial Technology (KIIT) in Bhubaneswar, Odisha, India.',
    'Identity, home region and place of study remain, while only evidence dates the handover. Unproven cause-of-death allegations are not included.')
add(2271, 'By what season and year was construction of the Colchester rapid transit system expected to finish?', 'Colchester rapid transit system', 'spring 2025',
    'The Colchester rapid transit system is a bus rapid transit system to serve the central city and its suburbs.',
    'The project is set to finish construction by spring 2025.',
    'It is set to have one line and eight stations, terminating in the proposed new garden community.',
    'The common sentence resolves the project. Route size and terminus information supply no construction completion season.')
add(313, "In what month and year did Andy Comfort begin his show on Hull's 107FM?", "Hull's 107FM", 'February 2025',
    "Hull's 107FM is a community radio station broadcasting to Kingston upon Hull, England.",
    'Andy Comfort, also formerly of BBC Humberside, began his show in February 2025.',
    'David Burns hosts a show on the station, having joined in 2023 after his stint at BBC Radio Humberside.',
    'The replacement concerns a different presenter, David Burns. His 2023 arrival does not date Andy Comfort’s show.')
add(1041, 'On what date did the USL Championship release the regular-season schedule that included Oakland Roots SC for 2025?', 'Oakland Roots SC', 'December 19, 2024',
    "The 2025 Oakland Roots SC season is the club's seventh season of existence and fifth in the USL Championship.",
    'On December 19, 2024, the USL Championship released the regular season schedule for all 24 teams.',
    'The Roots, after spending their first seasons playing at various college campuses, moved into the Oakland Coliseum during the offseason.',
    'Offseason venue relocation does not date the separate league-schedule announcement. Other articles describing this same announcement must not become separate formal targets or donors.')
add(302, 'In what month and year did Bangladesh’s Ministry of Foreign Affairs decide to send Toufique Hasan as ambassador to Austria?', 'Toufique Hasan', 'November 2024',
    'Toufique Hasan is a Bangladeshi diplomat and Ambassador of Bangladesh to Austria.',
    ['In 2025, Hasan was appointed ambassador of Bangladesh to Austria', 'Bangladesh Embassies around the world.'],
    ['Hasan has served at the Bangladesh Embassy in Paris', 'from 2014 to 2017.'],
    'Evidence distinguishes the November 2024 decision from the 2025 appointment. Earlier postings do not supply the decision month.')
add(2030, 'In what year was Mentor Marsh listed as a National Natural Landmark?', 'Mentor Marsh', '1964',
    ['Owned by the private sector, but protected by the state of Ohio', 'Cleveland, Ohio.'],
    'It was listed as a Landmark in 1964.',
    'It is a lakeshore wetland with marsh vegetation, aquatic plants, swamp and bottomland forest, and upland forest.',
    'Protected status is retained, but neither ownership nor habitat characteristics specifies the designation year.')
add(1409, 'What revised time and date were set for Freeski Big Air at the 2025 Asian Winter Games?', '2025 Asian Winter Games', '10:10 on 12 February 2025, local time (UTC+8)',
    'Freestyle skiing competitions at the 2025 Asian Winter Games in Harbin, China, was held at the Yabuli Ski Resort between 8–13 February, 2025.',
    ['All times are in local time (UTC+8)', 'to 10:10 on 12 February.'],
    'Thailand won its first Asian Winter Games medal.',
    'Evidence supplies the revised time, date and time zone; the overall competition range does not isolate the Big Air revision. This candidate has a substantial evidence/replacement length difference and needs final matching review.')
add(1128, 'On what date did Andrew Furey announce his pending resignation as Liberal leader and Premier of Newfoundland and Labrador?', 'Andrew Furey', 'February 25, 2025',
    'The 2025 Liberal Party of Newfoundland and Labrador leadership election will be held at an unannounced date to select a successor to Andrew Furey.',
    'February 25, 2025 – Liberal leader and Premier Andrew Furey announces his pending resignation in a press conference.',
    'The details of the leadership convention have not yet been publicly announced.',
    'The shared and replacement sentences describe an undated succession process. The other introductory sentence giving the trigger date is omitted from partial; only evidence dates the resignation announcement.')
add(2180, 'In what year did Andrew Duarte receive a hero award from the Denver Police Department?', 'Andrew Duarte', '2021',
    'West York officer Andrew Duarte was killed in the shooting.',
    'Duarte was highly regarded according to the Denver Police Department and won a hero award in 2021.',
    'His death was confirmed in a Facebook post by the West York Borough.',
    'Death and its official confirmation identify the officer but do not date his earlier Denver award.')
add(903, 'How many years of investment had been secured for the World Sevens Football format?', 'World Sevens Football', 'five years',
    "World Sevens Football (W7F) is an upcoming women's association football competition series format, which will feature existing clubs playing professional seven-a-side football.",
    'Investment for five years of the format has been made.',
    'Tournaments will be streamed live on DAZN.',
    'This is a duration, hence time. The event format and broadcaster do not determine the funded duration.')
add(2212, 'In what month and year did Okongo Community Library reopen to the public after reconstruction?', 'Okongo Community Library', 'January 2025',
    'The Okongo Community Library is a government library located in Okongo in the Ohangwena region.',
    'After reconstruction, the library opened its doors to the general public in January 2025.',
    'The library was constructed at a cost of approximately N$4 million.',
    'Location and construction cost do not date reopening after reconstruction.')
add(692, 'In what month and year did Brigitte Laganière announce her retirement?', 'Brigitte Laganière', 'July 2024',
    ['Brigitte Laganière (born August 1, 1996)', "Professional Women's Hockey League (PWHL)."],
    'In July 2024, she announced her retirement.',
    'She previously played for the Montreal Force of the Premier Hockey Federation (PHF).',
    'The description as a former player and her former teams do not determine when the retirement announcement occurred.')
add(656, 'In what month and year did Te Herenga o Te Rā begin transmitting electricity to the national grid?', 'Te Herenga o Te Rā', 'January 2025',
    'The Te Herenga o Te Rā solar farm is a photovoltaic power station near Waiotahe in the Ōpōtiki District of New Zealand.',
    'It began transmitting electricity to the grid in January 2025.',
    'It was the first solar farm to be connected directly to New Zealand\'s national grid.',
    'The ordinal first-grid-connected status is retained but does not establish its month of operation.')
add(860, 'In what year did the Tianjin Federation of Trade Unions receive the national People’s Satisfactory Civil Service Collective award?', 'Tianjin Federation of Trade Unions', '2022',
    ['The Tianjin Federation of Trade Unions (TFTU;', 'the Chinese Communist Party.'],
    'It received the national "People\'s Satisfactory Civil Service Collective" award in 2022.',
    'The TFTU oversees 16 district-level unions and multiple industrial unions, focusing on labor rights mediation, vocational training, and welfare programs.',
    'Organizational identity, scope and activities do not date the specific award.')
add(1298, 'In what year was OmiSoore Dryden appointed to the Dalhousie University School of Nursing?', 'OmiSoore Dryden', '2025',
    'OmiSoore H. Dryden is an academic whose work aims to identify barriers queer Black men encounter when donating blood in Canada.',
    'In 2025, Dryden was appointed to the Dalhousie University School of Nursing.',
    'Her dissertation focused on how blood donation regulations discriminate against queer people with a specific focus on Black queer people.',
    'The research topic and dissertation contain no date of the nursing-school appointment.')
add(379, 'During which years did Youssef Rajji serve as Lebanon’s ambassador to Jordan?', 'Youssef Rajji', '2022 to 2025',
    ['Youssef "Joe" Rajji (born', 'since February 2025.'],
    "He previously served as Lebanon's ambassador to Jordan from 2022 to 2025.",
    'Rajji has emphasized the need for Lebanon to maintain neutrality in regional conflicts and strengthen ties with Arab and European partners.',
    'The later ministerial position provides at most an upper bound, not the full ambassadorial interval. The duplicate timeline entry is not displayed in partial.')
add(1180, 'What was the filing deadline for candidates in the 2026 Kentucky Senate election?', '2026 Kentucky Senate election', 'January 9, 2026',
    'The 2026 Kentucky Senate election will be held on November 3, 2026.',
    'The deadline for candidates to file is January 9, 2026.',
    'The Republican and Democratic primary elections will be held on May 19.',
    'The general and primary election dates do not establish the filing deadline.')
add(939, 'On what date was the trailer for Kanneda released?', 'Kanneda', '26 February 2025',
    'Kanneda is an 2025 Indian Hindi-language crime drama television series directed by Chandan Arora.',
    'The trailer of the series was released on 26 February 2025.',
    'The series premiered on 21 March 2025 on JioHotstar.',
    'The series premiere date is a distinct event and cannot establish the exact trailer-release day.')

for cid, reason in [
    ('ragognize_train_0277', 'Full article places the official announcement in December 2025 after a January 2025 release, an apparent chronology error; excluded rather than guessing a correction.'),
    ('ragognize_train_0636', 'Full article gives conflicting weekly Oricon chart debut positions in the same sentence; prefer an internally clean source.'),
    ('ragognize_train_0907', 'Full article associates February 12 with Valentine Day and Eid al-Adha with March 2024, evident date-quality concerns; excluded.'),
    ('ragognize_train_1206', 'Source repeats a January 22 theatrical release but the box-office section states February 5 release. Excluded for internally inconsistent release dates.')]:
    if cid not in review['reviewed_full_article_ids']:
        review['reviewed_full_article_ids'].append(cid)
    if not any(x['candidate_id'] == cid for x in review['excluded']):
        review['excluded'].append({'candidate_id':cid,'reason':reason})

for row in out:
    source = pool[row['candidate_id']]
    text = source['source_content']
    intervals = []
    for key in ['common_quote', 'evidence_quote', 'partial_quote']:
        quote = row[key]
        assert text.count(quote) == 1, (row['candidate_id'], key)
        start = text.index(quote)
        row[key.replace('_quote', '_span')] = {'start':start,'end':start+len(quote)}
        intervals.append((start,start+len(quote)))
    intervals.sort()
    assert all(a[1] <= b[0] for a,b in zip(intervals,intervals[1:])), row['candidate_id']
    for key in ['source_url','source_title','revision_id','retrieval_date_utc','source_content_sha256','original_question','original_answer_quote','official_information_type']:
        row[key] = source[key]
(P/'time_quantity_candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'time_quantity_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
print('reviewed candidates:',len(out))
