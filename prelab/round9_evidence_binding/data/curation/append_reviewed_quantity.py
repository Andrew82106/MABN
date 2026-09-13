"""Serialize manually reviewed quantitative sources; no output-based selection."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
pool = {r['candidate_id']:r for r in map(json.loads,(P/'source_pool.jsonl').read_text(encoding='utf-8').splitlines())}
out = json.loads((P/'time_quantity_candidates.json').read_text(encoding='utf-8'))
existing = {r['candidate_id'] for r in out}
review = json.loads((P/'time_quantity_review.json').read_text(encoding='utf-8'))

def add(n, question, subject, answer, common, evidence, partial, rationale, aliases=None):
    cid = f'ragognize_train_{n:04d}'
    if cid in existing:
        return
    source=pool[cid];text=source['source_content']
    def extract(value):
        if isinstance(value,str):
            assert text.count(value)==1,(cid,value)
            return value
        first,last=value
        assert text.count(first)==1,(cid,first)
        i=text.index(first);j=text.index(last,i)+len(last)
        return text[i:j]
    out.append(dict(candidate_id=cid,category='quantity',question=question,subjects=[subject],reference_answer=answer,answer_aliases=aliases or [],common_quote=extract(common),evidence_quote=extract(evidence),partial_quote=extract(partial),rationale=rationale,question_rewrite_reason='Normal single-attribute question without confirmation wording or suggested answer; retained source attribution and numerical qualification.',status='source_reviewed_candidate'))
    review['reviewed_full_article_ids'].append(cid)

add(2295, 'How many votes did Reyna Walters-Morgan receive on the third ballot in her 2025 DNC vice-chair election?', 'Reyna Walters-Morgan', '210 votes',
    ['Reyna Walters-Morgan is an American attorney', 'for the Democratic National Committee.'],
    ["On Saturday, February 1, 2025, Walters-Morgan was elected", "National Harbor, Maryland."],
    ['She has previously served as the Director of Civic Engagement', 'for the Democratic National Committee.'],
    'Only evidence supplies the 210 votes. Current and prior DNC roles do not determine the vote count.', ['210'])
add(2179, 'What career-high pitch speed did Sha Tzu-chen record during the World Baseball Classic 2025 Qualifiers?', 'Sha Tzu-chen', '152 kilometers per hour',
    'Sha Tzu-chen (born 15 October 2003) is a Taiwanese professional baseball pitcher.',
    'Sha set a career high in pitch speed (152 kilometers per hour) during the  World Baseball Classic 2025 Qualifiers.',
    'Sha was a member of the Chinese Taipei national baseball team during the 2024 U-23 Baseball World Cup, facing South Korea and Nicaragua.',
    'Nationality, position and a previous tournament appearance do not imply pitch velocity.', ['152 km/h','152 kph'])
add(1586, 'How many signatures were on the lawyers’ petition opposing the proposed fifth amendment to the Judicature Act 2010?', 'Judicature Act 2010', '62 signatures',
    ['The Judicature Act (Law no: 22/2010)', '21 October 2010.'],
    'Many lawyers had submitted a 62 signature petition against the bill and asked the government to reconsider the bill.',
    'The bill was passed in parliament following 50 in favour, 4 against, and 6 abstaining.',
    'The replacement counts parliamentary votes on enactment, not signatures opposing a later amendment; its numbers cannot determine the petition size.', ['62'])
add(940, 'What is Aaliyah Chavez’s school record for assists in a season?', 'Aaliyah Chavez', '240 assists',
    'Aaliyah Chavez (born November 20, 2006) is an American basketball player who attends Monterey High School.',
    ['She holds school records for most points in a game (57)', 'career three-pointers (639).'],
    ['During her senior year she averaged 34.9 points', 'for the first time since 1981.'],
    'The evidence explicitly binds 240 to single-season assists. The retained senior-year averages and championship result supply no season game count or maximum total, so the record cannot be reconstructed.', ['240'])
add(1158, 'By December 2024, how many copies of the Yūsha Party o Oidasareta Kiyōbinbō series were in circulation?', 'Yūsha Party o Oidasareta Kiyōbinbō', 'over 3.7 million copies',
    ['Yūsha Party o Oidasareta Kiyōbinbō: Party Jijō de Fuyojutsushi o Yatteita Kenshi, Bannō e to Itaru is', 'illustrated by Yuri Kisaragi.'],
    'By December 2024, the series had over 3.7 million copies in circulation.',
    'It originally began as a web novel series on the Shōsetsuka ni Narō website in February 2021.',
    'The original question calls the work a light novel; the rewrite asks the source’s broader series circulation without claiming a novels-only breakdown. Creators and web-publication date do not give circulation.', ['more than 3.7 million copies'])
add(473, 'What was Astrotalk’s post-money valuation following the April 2024 funding round led by Elev8 Venture Partners?', 'Astrotalk', '$300 million',
    'Astrotalk is an Indian online astrology platform that connects users with astrologers through chat and voice calls.',
    ['This was followed by another $9.5 million funding round in April 2024', "post-money valuation to $300 million."],
    'Astrotalk offers services such as horoscope readings, birth chart analysis, numerology, tarot reading, and Vedic astrology consultations.',
    'Service descriptions retain the company without funding or prior valuation figures; only evidence states the relevant post-money valuation.', ['300 million dollars','US$300 million'])
add(1379, 'How many Hwasong-12B missiles were inspected during Kim Jong Un’s visit to a missile base in late October 2024?', 'Hwasong-12B', 'at least two missiles',
    ['The Hwasong-12B (Korean:', 'intermediate-range ballistic missile.'],
    "At least two Hwasong-12B missiles were inspected during a Kim Jong Un's visit to a missile base in late October 2024.",
    ['The Hwasong-12B is a single-stage missile', '6-axle transporter erector launcher.'],
    'The replacement gives missile stages and transporter axles, not inspected missiles. The lower bound at least two must be preserved.', ['at least 2 missiles'])
add(1520, 'How much crude oil production was reported to be at risk from the March 2025 Trans-Niger Pipeline explosion?', 'Trans-Niger Pipeline', '450,000 barrels of crude oil production',
    ['The Trans-Niger Pipeline (TNP) is a major oil pipeline', 'to export terminals.'],
    'In March 2025, an explosion was reported on the Trans-Niger Pipeline the putting 450,000 barrels of crude oil production at risk.',
    "Renaissance Africa Energy Holdings took over operations of the pipeline after acquiring Shell's onshore subsidiary.",
    'Only evidence gives threatened production. The text does not specify a per-day denominator, so neither the question nor reference adds one. Route and operator information do not imply the amount.', ['450,000 barrels'])
add(1532, 'What was the estimated total cost of the renovation project at the Church of the Immaculate Conception in Tallow?', 'Church of the Immaculate Conception', '€1.4 million over five years',
    'The Church of the Immaculate Conception, Tallow is a Catholic church in the parish of Tallow, County Waterford.',
    'It is estimated that the total cost of the project will be €1.4 million over five years.',
    'It is listed in the Record of Protected Structures maintained by Waterford City and County Council.',
    'Only evidence gives an estimated project cost. The location and protected-building status supply no budget; no phase costs are retained.', ['€1.4 million','1.4 million euros'])
add(1842, 'What share of the vote did Gina Jacobs receive in the November 2024 San Diego County Board of Supervisors general election?', 'Gina Jacobs', '40.2%',
    ['Incumbent Joel Anderson, a Republican, and Gina Jacobs, a Democrat', 'since no other candidates qualified to run.'],
    'Anderson went on to defeat Jacobs 59.8% to 40.2% in the November general election.',
    'District 2 comprises the cities of El Cajon, Poway, Santee, as well as over 40 unincorporated communities and tribes in eastern San Diego County.',
    'The common sentence identifies candidates but retains no other candidate’s percentage. District geography cannot yield Jacobs’s vote share.', ['40.2 percent'])
add(1766, 'How much seed investment did Youscan secure from The OpenFund in 2010?', 'Youscan', '€50,000',
    'Youscan is a social media and online media monitoring, analytics and social media listening platform.',
    'In 2010, the company secured a seed investment of €50,000 from the European venture fund The OpenFund.',
    'In July 2022, Youscan raised $2 million in investments from existing investors.',
    'The replacement concerns another funding round twelve years later and in a different currency. It cannot establish the 2010 seed amount.', ['50,000 euros'])
add(420, 'Approximately how much money was raised to create the Go-Go Museum?', 'Go-Go Museum', 'about $2.5 million',
    ['The Go-Go Museum is a museum located in Washington, D.C.', 'a variety of funk music developed in the city.'],
    'About $2.5 million was raised to fund the creation of the museum.',
    'The museum launched a soft opening on November 18, 2024, with a grand opening scheduled for February 2025.',
    'Location, theme and opening schedule do not give the funds raised; the approximate qualification is preserved.', ['approximately $2.5 million','about 2.5 million dollars'])
add(406, 'How many startups had Udit Goenka provided with mentorship and angel investments?', 'Udit Goenka', 'over 35 startups',
    ['Udit Goenka is an Indian entrepreneur', 'artificial intelligence (AI) sectors.'],
    'He has also provided mentorship and angel investments to over 35 startups, emphasizing scalable business models and automation.',
    'He co-founded Power Up Hosting, a web hosting service, which he operated for seven years before exiting.',
    'The replacement describes a single venture and its operating duration. These do not determine the number of startups supported; retain over rather than exact 35.', ['more than 35 startups'])
add(2141, 'How many V163-4.5 MW wind turbines did the contract for Khangela Emoyeni Wind Farm specify?', 'Khangela Emoyeni Wind Farm', 'thirty-two turbines',
    'The SPV company is called Khangela Emoyeni Wind Farm (Pty) Limited.',
    'The contract calls for the supply, installation and 10-year maintenance of the thirty-two V163-4.5 MW wind turbines.',
    'Rand Merchant Bank is arranging all the necessary funding for this renewable energy project.',
    'No total MW capacity is displayed: otherwise dividing 144 MW by the question’s 4.5 MW per turbine would leak 32. The legal company identity and funding arranger do not supply the count.', ['32 turbines','32 wind turbines'])
add(414, 'What percentage of the vote did Morgan Foreman receive against Jason Rogers in the general election?', 'Morgan Foreman', '75%',
    ['Morgan Foreman (born 1989 or 1990', 'Michigan House of Representatives since 2025.'],
    'She faced Republican Jason Rogers in the general election. She defeated Rogers with 75% of the vote.',
    "Foreman planned to build on Brabec's work in the legislature, highlighting her experience in Brabec's office.",
    'Both evidence sentences resolve the opponent and general election. Legislative plans and current office in partial do not supply a vote share.', ['75 percent'])
add(1861, 'In TROPION-Breast01, what proportion of the chemotherapy comparison group received eribulin rather than datopotamab deruxtecan?', 'datopotamab deruxtecan', '60%',
    'Datopotamab deruxtecan, sold under the brand name Datroway, is an anti-cancer medication used for the treatment of breast cancer.',
    ["A total of 732 patients were randomized (1:1)", 'or gemcitabine (9%).'],
    'Participants must have experienced disease progression, been deemed unsuitable for further endocrine therapy, and have received one or two lines of prior chemotherapy for unresectable or metastatic disease.',
    'The evidence directly lists the chemotherapy allocation. All component percentages and arm counts are removed together, so eligibility criteria cannot reconstruct 60%. The question identifies the trial without claiming clinical advice.', ['60 percent'])
add(1953, 'According to the Fiscal Policy Office study associated with the Integrated Entertainment Business Bill, to what amount was visitor spending projected to increase?', 'Integrated Entertainment Business Bill', '60,000 Baht',
    ['The Integrated Entertainment Business Bill, also known as', 'within integrated resorts.'],
    ['According to a study conducted by the Fiscal Policy Office', 'from 40,000 Baht to 60,000 Baht.'],
    'In October 2024, Deputy Finance Minister Julapun Amornvivat announced the draft Integrated Entertainment Business Bill would be submitted to the cabinet for consideration later in the year.',
    'The attributed projection and both spending figures are removed together. The legislative timetable does not imply projected expenditure.', ['60,000 baht','THB 60,000'])
add(683, 'What commission rate did Hubtel say it receives from payments processed through its platform under the ECG agreement?', 'Hubtel', '0.95%',
    'Hubtel is a Ghanaian financial technology and e-commerce company headquartered in Accra, Ghana.',
    ['Hubtel also addressed concerns about its revenue-sharing agreement with ECG', 'reports suggesting a 3% fee.'],
    "Hubtel was contracted to develop and manage the ECG PowerApp, a mobile application aimed at improving customer payments and revenue collection.",
    'The question attributes the disputed rate to Hubtel rather than independently certifying it. The application-development role does not determine its commission.', ['0.95 percent'])
add(1782, 'What revenue did Gentoo Media report for the fourth quarter of 2024?', 'Gentoo Media', '€35.9 million',
    'Gentoo Media Inc. (often referred to as Gentoo) is a publicly traded company operating in the iGaming affiliate marketing industry.',
    'In the fourth quarter of 2024, Gentoo Media reported €35.9 million in revenue, reflecting a 38% increase compared to the same quarter in 2023.',
    'In 2023 and 2024, Gentoo Media acquired several iGaming-related websites and businesses, including AskGamblers, KaFeRocks, Time2Play, Casinomeister and Titan Inc.',
    'The common identity and acquisition list provide no revenue or prior-year amount. The question retains reported revenue attribution.', ['35.9 million euros'])
add(382, 'What total assets did the All United States Kendo Federation report for the fiscal year ending December 2023?', 'All United States Kendo Federation', '$1,014,357',
    'The All United States Kendo Federation (AUSKF) is the national governing body for kendo in the United States.',
    ['Financially, the AUSKF reported revenues of $683,540', 'total assets amounting to $1,014,357.'],
    'The AUSKF comprises 14 regional member federations, each coordinating training programs, grading examinations, and tournaments within their respective areas.',
    'All financial figures are removed together; regional structure cannot imply total assets.', ['1,014,357 dollars','US$1,014,357'])
add(1747, 'How many drivers participated in the inaugural Busch Clash in 1979?', 'Busch Clash', 'nine drivers',
    ["The inaugural Busch Clash, held on February 11, 1979", "NASCAR's season-opening exhibition races."],
    'This 20-lap, 50-mile event featured nine drivers who had secured pole positions during the previous season.',
    'The race was a sprint, with no points awarded, but a substantial purse of $150,000, including $50,000 for the winner.',
    'The complete source has a nine-name roster, but no roster or results table is displayed in partial. The race date, venue and prize purse cannot yield the driver count.', ['9 drivers'])
add(858, 'How many same-sex couples did Thailand’s Department of Provincial Administration report as married on 23 January 2025 under the Marriage Equality Act?', 'Marriage Equality Act', '1,832 couples',
    'The Marriage Equality Act came into effect on 23 January 2025.',
    ['On 23 January 2025, 1,839 same-sex couples married in Thailand', 'The Department of Provincial Administration reported 1,832 same-sex couples married on the same day.'],
    ['Same sex couples can register their marriage', '94 embassies and consulates worldwide.'],
    'Two attributed totals differ, so the question explicitly asks the Department’s 1,832. All marriage counts are removed in partial, leaving registration locations; registration-office counts cannot reconstruct marriages.', ['1832 couples'])
add(1783, "How many teams in the 2025 Women's National Invitation Tournament bracket had at least 20 victories?", "2025 Women's National Invitation Tournament", '22 teams',
    ["The 2025 Women's National Invitation Tournament is", "or the 2025 WBIT."],
    'There are 22 teams with 20 or more victories in the bracket.',
    'The 2025 field features 11 automatic qualifiers and 37 at-large selections, chosen after consideration of a mix of criteria by WNIT officials.',
    'Qualification route is not determined by 20-win status, so 11 automatic plus 37 at-large selections do not reveal how many teams satisfy the win threshold.', ['22'])
add(177, 'How many entries were listed for the 2025 4 Hours of Abu Dhabi?', '2025 4 Hours of Abu Dhabi', '47 entries',
    ['The 2025 4 Hours of Abu Dhabi is an endurance sportscar racing event', 'Abu Dhabi, United Arab Emirates.'],
    ['The entry list was published on 11 February 2025', '10 in LMP2, 7 in LMP3, and 30 in GT.'],
    'It will be the penultimate and last of six rounds of the 2024–25 Asian Le Mans Series season.',
    'The total and every per-class count are removed together. Event duration in its title and number of series rounds cannot imply entry count.', ['47'])
add(2125, 'By March 2025, how much had John Day School District spent on planning its energy usage project?', 'John Day School District', 'hundreds of thousands of dollars',
    'John Day School District, also known as Grant School District No. 3, is a school district based in Canyon City, Oregon.',
    'The district, by March 2025, had spent money in the hundreds of thousands of dollars for the planning for the energy usage project.',
    'Louis Dix was superintendent in the 2022-2023 school year, and retired at the end of the year. Mark Witty became superintendent in 2023.',
    'Administrative succession supplies no project expenditure. The source gives only an order-of-magnitude amount, which the reference preserves rather than inventing an exact cost.')
add(1396, 'How many people attended the maiden edition of Adedoyin Oseni’s Evening of Worship in Sheffield?', 'Adedoyin Oseni', '250 people',
    'Adedoyin Oseni (born August 9, 1994) is a Nigerian-born Afro-contemporary gospel saxophonist and worship leader.',
    ['He organised the “Evening of Worship” held in Sheffield, United Kingdom.', 'traditional and western gospel tunes.'],
    ['He bagged a degree in political science from Olabisi Onabanjo University', 'the University of Huddersfield in 2024.'],
    'The evidence names the concert before its maiden-edition attendance. Education history does not indicate audience size.', ['250 persons','250 attendees'])

exclusions = [
    (1927,'Internal chronology says elected March 3 but the same election votes counted March 5. The target count is embedded in a very long sentence; prefer clean, concise sources.'),
    (1407,'Not selected: source connects hashtag history to 2006 and 2016 without clear scope and offers a broad social-meme definition. Prefer directly attributable institutional or event counts.'),
    (1504,'Full source assigns two unexplained different emissions quantities to the same region and year. Avoid ambiguous source accounting.'),
    (1930,'Source was read but not selected: Yuval Raphael already appears as a reserved geography target, creating a cross-group subject overlap.'),
    (989,"'profit skyrocketed into 84%' does not distinguish profit margin from profit growth. Do not reinterpret the ambiguous source as a precise financial metric."),
    (2234,'Full article states seven alpine skiers and four per gender, an internal count contradiction.'),
    (2254,'Source is quantitatively interpretable but not selected: the same 2025 Asian Winter Games is already represented among time candidates, so avoid adding another event-linked target.'),
    (1991,"The target review sentence is malformed ('rated the 9/10'). Cleaner quantitative passages available; not selected.")]
for n,reason in exclusions:
    cid=f'ragognize_train_{n:04d}'
    if cid not in review['reviewed_full_article_ids']:review['reviewed_full_article_ids'].append(cid)
    if not any(r['candidate_id']==cid for r in review['excluded']):review['excluded'].append({'candidate_id':cid,'reason':reason})

for row in out:
    source=pool[row['candidate_id']];text=source['source_content'];intervals=[]
    for key in ['common_quote','evidence_quote','partial_quote']:
        quote=row[key];assert text.count(quote)==1,(row['candidate_id'],key)
        i=text.index(quote);row[key.replace('_quote','_span')]={'start':i,'end':i+len(quote)};intervals.append((i,i+len(quote)))
    intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),row['candidate_id']
    for key in ['source_url','source_title','revision_id','retrieval_date_utc','source_content_sha256','original_question','original_answer_quote','official_information_type']:row[key]=source[key]
(P/'time_quantity_candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'time_quantity_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
from collections import Counter
print(dict(Counter(r['category'] for r in out)))
