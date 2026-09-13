"""Render all six frozen results, including null effects and the prior reference."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

r=read(OUT/'summary.json')
old=read(ROOT.parent/'round18_input_information_diagnostics/results/summary.json')
config=read(ROOT/'protocol.json')
names={'base':'原均值探针（LB＋NLL）','base_mmd':'原探针＋资料扰动MMD',
       'base_ecs':'原探针＋资料表示对应ECS','base_pks':'原探针＋FFN预测变化PKS',
       'base_all':'原探针＋三类信号','signals_only':'只用三类新信号'}
width={'base':785,'base_mmd':787,'base_ecs':789,'base_pks':789,'base_all':795,'signals_only':10}
base=r['methods']['base'];main=r['methods']['base_all']
delta=main['windows']['f1']-base['windows']['f1']
lo,hi=r['paired_bootstrap']['windows']['contrasts']['all_vs_base']['ci95']
finding=('窗口F1提高，固定拟合条件下的探索性差值区间也高于零。' if lo>0 else
         '窗口F1下降，固定拟合条件下的探索性差值区间低于零。' if hi<0 else
         '窗口F1差值区间跨零，尚无稳定改善的证据。')
lines=[
'**R19：三类信号分别与联合测试**','',
'结论：整条回答的风险区分有改善，局部定位基本没有改善；三类一起加入也没有优于所有单独加入的方案。',
'',
f"主比较是预先指定的三类全加：窗口F1从{base['windows']['f1']:.3f}到{main['windows']['f1']:.3f}，差值{delta:+.3f}，95%区间[{lo:.3f}, {hi:.3f}]。{finding}",
'',
'本轮已经实际提取三种逐token信号，训练并评测5折×6组共30个逻辑回归。原问题、资料、生成回答、标签、4BPE窗口和五折事件划分保持不变；只改变提供给探针的特征。R18均值基线的标准化、权重、系数、全部候选窗口分数和阈值逐折精确重现，概率最大差为0。',
'',
'| 配置 | 输入维数 | 拟合集窗口F1均值 | 未见事件窗口F1 | 整答F1 | 合并高亮词元F1 |',
'|---|---:|---:|---:|---:|---:|']
for key,label in names.items():
    v=r['methods'][key]
    lines.append(f"| {label} | {width[key]} | {r['fold_mean'][key]['fit']['f1']:.3f} | {v['windows']['f1']:.3f} | {v['answers']['f1']:.3f} | {v['highlight_tokens']['f1']:.3f} |")
lines += ['',
f"另作既有方案参考：R18保留4个词元顺序的方案在完全相同划分上的窗口F1为{old['methods']['slots_lr']['windows']['f1']:.3f}。本轮没有把新信号与该结构再组合，不根据当前结果另加网络或调参。该数值只是既有固定方案参考，不是本轮重训的第七组。",
'',
'拟合集列是五折均值；其他三列由所有事件各出现一次的折外判断合并计算。数据为R16原训练集301题、278事件组、602份回答，其中598份可以整答评测，151份为风险回答；117份明确安全拒答作为整答负类，4份未解决回答排除。局部定位只评9,526个有标签窗口（1,063个风险），高亮评9,716个词元（768个风险）。窗口只要与风险词元重叠就算风险，窗口F1不等于逐词元F1。',
'',
'**每类信号是否提供了增量**','',
'| 预定对照（前者减后者） | 窗口F1差值 | 窗口差值95%区间 | 整答F1差值 | 整答差值95%区间 |',
'|---|---:|---:|---:|---:|']
for tag,entry in r['paired_bootstrap']['windows']['contrasts'].items():
    a,b=entry['methods']; wl,wh=entry['ci95']
    al,ah=r['paired_bootstrap']['answers']['contrasts'][tag]['ci95']
    dw=r['methods'][a]['windows']['f1']-r['methods'][b]['windows']['f1']
    da=r['methods'][a]['answers']['f1']-r['methods'][b]['answers']['f1']
    lines.append(f'| {names[a]} − {names[b]} | {dw:+.3f} | [{wl:.3f}, {wh:.3f}] | {da:+.3f} | [{al:.3f}, {ah:.3f}] |')
lines += ['',
'主比较始终是三类全加减原均值探针，其余对照用于分辨单独信号和组合效果。按事件组做2,000次配对重抽样，固定模型和阈值；区间不包含重新训练的不确定性，也没有多重比较校正。各折训练/校准/评测AUROC与AP及题型、资料条件、回答长度分层均保存于summary.json。不同折原始分数不混在一起计算整体AUROC。',
'',
'**实际输入了什么**','',
'- MMD：每次只替换一个来源正文中严格位于正文范围内的token，长度完全相同。保持问题、标题、另一来源、聊天模板、固定回答及位置不变。原分布与两个单来源干预分布使用LUMINA原top100未重归一化概率/余弦核计算差异，每token先对两来源取均值和最大值，形成2维，再平均4个BPE。它测当前回答前缀条件下的资料敏感度。',
'- ECS：复用本地ReDeEP-token适配，比较生成处内部表示与各注意力头所关注资料的内部表示。原784维按固定四个7层段、全部头平均，得到4维；它是资料利用的近似指标，不是事实关系核查器。',
'- PKS：复用FFN前后残差经LogitLens后的词表分布标准JSD，原28层同样按四段平均，得到4维。它测模块处理造成的预测变化，不直接等于参数知识、错误或谎言。',
'',
'MMD使用预测当前token前的P+j−1位置；ECS/PKS使用读入当前token后的P+j位置。三者都按原4BPE窗口平均，不按金标风险范围提取或挑选信号。没有按标签另选头、层、扰动来源或特征方向。三个新增块共10维，分类器只学线性组合；C=.01、损失总权重3854、随机种子、优化器和阈值规则均沿用R18。',
'',
'干扰资料来自原R6训练来源90篇候选，独立性审核排除1篇与本轮训练主体有具体公司重叠的资料，固定保留89篇。按可见问题文本及来源位置决定干扰流，不循环复制；完整/部分资料条件下，同题同位置使用同一流的相应长度前缀。全体602份回答共1,204次单来源正文干预，每次替换20–252个token。它是保持长度的token干预，不宣称是自然检索文本；截断/拼接、主题变化、保留的标题与冗余资料都可能影响MMD。',
'',
'**定位、误报与训练差距**','',
'| 配置 | 窗口精准率 | 窗口召回率 | 安全拒答误报/117 | 风险区域有命中 | 风险区域完整覆盖 | 高亮词元占比 |',
'|---|---:|---:|---:|---:|---:|---:|']
for key,label in names.items():
    v=r['methods'][key];w=v['windows'];h=v['highlight_tokens']
    lines.append(f"| {label} | {w['precision']:.3f} | {w['recall']:.3f} | {v['safe_refusal_false_positives']}/117 | {h['risk_regions_any_hit']}/{h['risk_regions']} | {h['risk_regions_fully_hit']}/{h['risk_regions']} | {h['marked_token_fraction']:.1%} |")
lines += ['',
'整答风险取全部候选窗口的最高分，窗口和整答阈值分别只由该轮校准组选择。安全拒答没有参与定位探针的窗口训练，因此整答结果还包含拒答处理问题。分数未经真实概率校准，不能把0.8解释成80%幻觉率；最高窗口碰到风险范围也不等于完整正确定位。',
'',
'整答提升主要表现为误报减少：三类全加后，正确识别的风险回答100→102，误报39→26，漏报51→49。误报净少13条，其中安全拒答净少12条（26→14）；这比新增发现错误的数量大。逐条对比则是新增识别4条风险回答、丢失2条；安全拒答修正14条误报、又新增2条误报。这里描述的是冻结模型及各自校准阈值的共同结果，不能据此单独归因于某个信号。',
'',
'窗口方面，误报491→456，但正确命中的风险窗口也从732降到720，因此F1只略变。按预定规则选出的两个最大改善案例主要减少了无问题文字上的误报；两个最大退步案例主要改变重叠窗口判断或高亮边界，原有核心风险片段仍被覆盖。这四例没有显示新增识别另一条关系事实，亦不代表全部样本。具体原文、金标与两法高亮见案例说明。',
'',
'过拟合也没有因此解决：三类全加的拟合集窗口F1均值仍为0.860，折外为0.643。只用三类新信号的拟合/折外F1为0.426/0.415，在本轮固定压缩加线性分类器的条件下区分能力有限。这不能证明原始信号无用，也不能从本轮结果断言增加网络复杂度会奏效。',
'',
'**计算与检查**','',
'全部计算使用同一个本地Qwen2.5-7B-Instruct、NF4量化模型，重放原固定回答，不重新生成答案，不另调用生成或裁判模型。小样本检查包括identity资料干预MMD为0、重复提取一致、原LB与R18逐token相同、正文以外token完全不变和回答坐标一致。全量特征与模型都由文件哈希绑定。',
'',
'独立审计已通过：30个分类器、5折基线精确重现、全部冻结概率与离散预测、三种粒度的计数、分层、配对区间、602份特征与1,204次干预均核对一致。特征提取主循环约17分钟，正式30组分类器训练及汇总约25秒；这些是本机本轮记录，不含全部准备与审计耗时。',
'',
'本轮延续已经分析过的R18数据/五折，属于探索性证据，不能作为全新独立测试或SOTA证明。ECS/PKS固定压缩可能舍弃少数有效头层；干扰敏感度低也可能因为模型已有知识、其他资料或已生成前缀足以支持答案。负面结果不能证明这些原始信号无用；正面结果也需要新事件及人工审核标签再验证。现有标签是此前助手标注与裁决，human_gold=false，场景仍为受控的英文短问答。',
'',
'方法来源：[LUMINA官方实现](https://github.com/deeplearning-wisc/LUMINA)、[ReDeEP原论文](https://arxiv.org/abs/2410.11414)。本轮只提取对应信号并使用统一小探针，不冒充原论文全流程复现。',
'',
'[完整方案](../PLAN.md) · [冻结配置](../protocol.json) · [全部指标](summary.json) · [整答变化记录](answer_change_diagnosis.json) · [案例说明](CASE_REVIEW19.md) · [基线精确复现](baseline_reproduction.json) · [独立审计](INDEPENDENT_AUDIT19.json) · [特征清单](../data/feature_manifest.json) · [扰动方案](../data/perturbation_manifest.json)',
]
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(OUT/'REPORT.md')
