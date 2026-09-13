"""Explicit source decisions after reading each full packet; not an automatic labeler."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from curation_io9 import pool, segments, add_manual


def decision(cid, question, subject, answer, evidence, common, partial, reason):
    r = pool()[cid]
    ss = segments(r["source_content"])
    def quote(index):
        if isinstance(index, tuple):
            return r["source_content"][ss[index[0]]["start"]:ss[index[1]]["end"]]
        return ss[index]["text"]
    return {"candidate_id": cid, "category": "relation", "question": question, "subjects": [subject],
            "reference_answer": answer, "answer_aliases": [], "evidence_quote": quote(evidence),
            "common_quote": quote(common), "partial_quote": quote(partial), "rationale": reason,
            "question_rewrite_reason": "Neutral self-contained relation question; remove upstream confirmation and answer suggestion.",
            "review_scope": "Independently read every source packet segment; explicitly chose three nonoverlapping exact quotes."}


BATCH1 = [
    decision("ragognize_train_0061", "Who became CEO of Hero Esports following its November 2024 corporate restructure?", "Hero Esports", "Danny Tang", 3, 2, 7,
             "证据明确区分Dino Ying转任主席与Danny Tang任CEO；共同句仅更名重组，替换句仅ESIC成员关系，不提供CEO。"),
    decision("ragognize_train_0151", "What roles does Stacy Sims hold at Osmo Nutrition?", "Stacy Sims", "co-founder and CEO", 13, 0, 12,
             "证据She回指Sims并给Osmo两个职务；共同职业描述、替换句的大学研究职位都不能推出她在Osmo的角色。"),
    decision("ragognize_train_1333", "Which AI coding assistant did Cursor developer Anysphere announce it had acquired in November 2024?", "Anysphere", "Supermaven", 19, 2, 18,
             "共同句建立Cursor与开发公司Anysphere关系；证据给收购对象Supermaven，替换句仅融资方与估值，不给被收购对象。"),
    decision("ragognize_train_1661", "Which ministry issued the notification announcing Monir Haidar's appointment as Special Assistant to the Chief Adviser?", "Monir Haidar", "Ministry of Public Administration", 12, 1, 8,
             "共同句说明Haidar现任特助；证据给发通知部门；替换仅BanglaVision顾问岗位。避开原文有误的2024任命日期，问题只问发布部门。"),
    decision("ragognize_train_1068", "What was the name of the agent model H Company announced in November 2024?", "H Company", "Runner", 5, 0, 6,
             "证据含模型专名Runner；共同句公司身份和替换句agent目标不提供模型名称。"),
    decision("ragognize_train_2028", "Which Almawave platform integrates the Velvet AI models?", "Velvet AI", "AIWave", 9, 0, 2,
             "证据明确集成到AIWave；共同句仅开发者Almawave，替换为训练用Leonardo与CINECA，不能把训练设备当集成平台。"),
]

BATCH2 = [
    decision("ragognize_train_2331", "Which family carried out the major restoration of Rivarossa Castle in 1825?", "Rivarossa Castle", "the Daziani family", 3, 0, 4,
             "证据给1825修复者；共同地点和替换建筑防御墙不提供家族，亦不展示曾经的Valperga所有者干扰为答案。"),
    decision("ragognize_train_0403", "Who recorded the original Telugu-language song Kissik for Pushpa 2: The Rule?", "Kissik", "Sublahshini", 0, 2, 14,
             "证据区分作曲、填词、演唱角色且明确Sublahshini录音；共同视频发布时间和替换舞蹈编导更替只给其他信息，不能推出歌手。"),
    decision("ragognize_train_0230", "Which speedskating club is Peter Groseclose affiliated with?", "Peter Groseclose", "Potomac Speedskating Club", 6, 0, 7,
             "证据明确俱乐部归属；共同国籍项目和替换冬青奥参赛不提供俱乐部，不能把国家队当俱乐部。"),
    decision("ragognize_train_1147", "What academic rank does Muhammad Mashud hold in KUET's Department of Mechanical Engineering?", "Muhammad Mashud", "professor", 1, 0, 3,
             "证据给机械系教授学术职级；共同校长行政岗位不逻辑等于教授，替换研究生学历不能推出教授职级，避免原题Dean诱导。"),
    decision("ragognize_train_1219", "Which company acquired Sara Khaki's documentary Cutting Through Rocks?", "Sara Khaki", "Autlook Filmsales", 4, 1, 2,
             "共同片名及获奖信息保留目标身份，证据给合作导演与收购公司；替换导演教育没有收购方。原文只说acquired the film，问题不擅加发行权种类。"),
    decision("ragognize_train_0466", "Who were Ashwini Shinde's first kho kho coaches?", "Ashwini Shinde", "Kishore Gund and Hari Shinde", 8, 0, 4,
             "证据两个最初教练姓名；共同运动员身份和替换学校信息不提供教练。"),
    decision("ragognize_train_0653", "Which other defense-related industry is named alongside SkyWin Aeronautics Industry in Ethiopia's industrialization initiative?", "SkyWin Aeronautics Industry", "Homicho Ammunition Engineering Industry", 12, 0, 10,
             "证据直接列出同一政府工业化计划的另一企业；共同SkyWin业务和替换传感器研发不提供Homicho，问示例不宣称唯一企业。"),
    decision("ragognize_train_1505", "What position does Nima Shahbazi hold at Mindle.ai?", "Nima Shahbazi", "President", 4, 0, 3,
             "证据给President，区别原题错误CEO假设；共同Zillow竞赛和替换LLM专业能力不提供现公司职务。"),
    decision("ragognize_train_1657", "What was the name of the law firm John Tracy Cagas established in Digos?", "John Tracy Cagas", "Cagas Law Office", 6, 0, 5,
             "证据明确律师事务所专名；共同政治律师身份、替换开始执业年份不包含或推出事务所名称。"),
    decision("ragognize_train_1773", "Which organization has Jonathan Hill on its Board of Directors?", "Jonathan Hill", "Literary Arts", 16, 0, 6,
             "证据给Literary Arts董事会；共同职业和替换任教院校不提供该董事组织，院校职位不能移接成董事会归属。"),
    decision("ragognize_train_0513", "Which reform commission is Shahnaz Huda a member of in Bangladesh's interim government?", "Shahnaz Huda", "Police Reform Commission", 2, 0, 10,
             "证据给警察改革委员会，共同法学院教授和替换CALS主任是不同组织职务，不提供改革委员会名称。"),
]

BATCH3 = [
    decision("ragognize_train_1542", "Which organization is listed as a contributing partner of the International Year of Quantum Science and Technology?", "International Year of Quantum Science and Technology", "Google Quantum AI", 32, 0, 31,
             "共同句保留IYQ目标活动；证据区别distinguished partners与Google的contributing partner，替换Microsoft leading partner是另一角色，不能移接。改问角色对应机构，避免partial仅因问题所问Google名字消失。"),
    decision("ragognize_train_1401", "At which firm does Yousif Yahya serve as a Venture Partner?", "Yousif Yahya", "African Renaissance Partners", 3, 0, 6,
             "证据明确投资合伙人机构；共同创业身份和替换风投培训不提供任职公司。注意来源标题为Wikipedia Draft，留作可替换候选，须在来源质量限制中披露。"),
    decision("ragognize_train_1478", "Which faction of the Liberal Party does Leah Blyth belong to?", "Leah Blyth", "Conservative faction", 9, (0, 1), 7,
             "证据明确Conservative派别；共同参议员所属党和替换教育高管及妇女委员会经历不推出派系；不选带right-wing支持者的第4句防间接泄漏。"),
    decision("ragognize_train_2062", "What position did Tim Gokey hold on Princeton University's sailing team?", "Tim Gokey", "co-captain", 9, 8, 10,
             "共同毕业于Princeton使证据university回指明确；证据co-captain，替换Oxford奖学金经历不提供Princeton队内岗位；不选Oxford队长句避免错误移接。"),
    decision("ragognize_train_1155", "Which United Nations interagency group did Khalilur Rahman chair?", "Khalilur Rahman", "United Nations Interagency Group on Non-Tariff Barriers to Trade", 10, 0, 6,
             "证据给贸易非关税壁垒跨机构组；共同前UN官员和替换LDC发言人职务不提供所主持的小组名称。"),
    decision("ragognize_train_0646", "Which two organizations are named as Kirrilee Warr's previous workplaces?", "Kirrilee Warr", "ATLAS and Disability Services Commission", 3, 0, 2,
             "证据列两家过去工作单位；共同政党选区和替换地方议员及议会主席岗位不提供这两家单位。不沿原题声称仅有一个过去职位。"),
    decision("ragognize_test_1021", "Which music band is Mariam Shengelia a member of?", "Mariam Shengelia", "Mix2ura", 2, 0, 6,
             "证据给乐队Mix2ura；共同歌手身份和替换音乐学院教育不提供乐队，不能从专业训练推出归属。"),
    decision("ragognize_test_2227", "Which publisher was announced for Kyle Starks's horror comic Those Not Afraid in September 2024?", "Kyle Starks", "Dark Horse", 20, 0, 19,
             "证据明确Those Not Afraid的Dark Horse出版关系；共同漫画家身份、替换另一漫画合作公告没有该书出版社。问题只问真实出版合作，不把漫画剧情当现实犯罪。"),
    decision("ragognize_test_1668", "Who is the president of Kemp FC?", "Kemp FC", "Kathleen De Saren", 4, 0, 9,
             "证据给俱乐部主席姓名；共同女子俱乐部身份、替换学院与持证教练不提供主席。"),
    decision("ragognize_train_0615", "What consultor title did Vitalis Sekhonyana Marole hold in the Missionary Oblates of Mary Immaculate in South Africa from 2018 to 2025?", "Vitalis Sekhonyana Marole", "provincial consultor", 23, (0, 1), 18,
             "证据区别大教区consultor与修会provincial consultor；共同主教及修会身份，替换学校chaplain职务不推出该职级。原全文另有任命日期前后矛盾及20125错字，不采用那些句子，所问2018–2025任职区间只有证据明确给出。"),
    decision("ragognize_train_0001", "What office did Alejandro Acha hold on Athletic Club's board in 1903?", "Alejandro Acha", "secretary", 9, 0, 2,
             "证据区分Arana董事成员与Acha秘书；共同球员守门员身份和替换骑行专长不推出行政岗位。"),
    decision("ragognize_test_1722", "Which two writers does Tegwen Bruce-Deans cite as inspirations for her work?", "Tegwen Bruce-Deans", "Mererid Hopwood and Ocean Vuong", 12, 0, 14,
             "完整原文读后改问明确的文学影响关系，避免原源题出版社名字疑似拼写错误；共同作家身份与替换BBC研究员岗位不提供两位影响者。"),
    decision("ragognize_train_0642", "Who played bass in Hirax's live lineup for the Faster than Death tour?", "Hirax", "Jose Gonzalez", 2, 0, 3,
             "证据精确分配现场阵容三人乐器，问贝斯手Jose Gonzalez；共同Hirax专辑身份和替换单曲发行不提供贝斯手。不混淆录音人员表Neil Metcalf的贝斯职责与本次巡演。"),
    decision("ragognize_test_0690", "Who is the head coach of the Alpha Insurance Protectors?", "Alpha Insurance Protectors", "Mike Santos", (3, 4), 0, 1,
             "证据连续句与明确Head coach标题给Mike Santos；共同排球队身份、替换保险公司关联不提供教练。"),
    decision("ragognize_test_1012", "What executive role does Sheikh Aliur Rahman hold at the London Tea Exchange?", "Sheikh Aliur Rahman", "Chief Executive", 1, 0, 2,
             "证据给London Tea Exchange的Chief Executive；共同商人身份与替换创办另一公平薪酬组织的事实不推出茶交易所职务。"),
]

if __name__ == "__main__":
    dest = ROOT/"data/curation/relation_additional.json"
    import json
    if not dest.exists():
        dest.write_text(json.dumps({"curator": "evaluation", "decisions": []}), encoding="utf-8")
    batch = sys.argv[1] if len(sys.argv) > 1 else "1"
    add_manual(dest, {"1": BATCH1, "2": BATCH2, "3": BATCH3}[batch])
