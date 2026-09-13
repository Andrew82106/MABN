import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
r=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
names={'base':'原均值LB＋NLL','slots_base':'原顺序窗口强基线','r19_all':'R19三信号联合',
       'lookback_tuned':'纯Lookback，校准选C','redeep_tuned':'ReDeEP局部适配，校准选组合',
       'harp256':'HARP启发的256维投影','hidden256':'原hidden拟合内PCA256',
       'delta256':'证据差分拟合内PCA256','base_harp':'原均值＋HARP投影',
       'base_delta':'原均值＋证据差分','base_harp_delta':'原均值＋两类向量（主方案）'}
lines=['**R21：内部向量增加了，但两项F1目标仍未达到**','',
'主方案4词元窗口F1为0.665、整答F1为0.682；原顺序窗口基线为0.655/0.691。定位差值的事件配对95%区间为[-0.025,0.042]，没有明确改善证据，整答还下降了。本轮不作为最终模型，也不把开发成绩称为新独立测试或SOTA。','',
'| 方法 | 拟合窗口F1均值 | 折外窗口F1 | 整答F1 | 高亮词元F1 |','|---|---:|---:|---:|---:|']
for k,n in names.items():
 v=r['methods'][k]
 lines.append(f"| {n} | {r['fold_mean'][k]['fit']['f1']:.3f} | {v['windows']['f1']:.3f} | {v['answers']['f1']:.3f} | {v['highlight_tokens']['f1']:.3f} |")
lines+=['',
'本轮沿用R16实际train的301题、278事件组、602份原生Qwen回答；598份整答、9,526个原4BPE窗口可评。未修改标签、分母或事件分区。每折3折拟合、1折校准、1折评测；两个阈值分别只用校准标签，候选结构不根据外层分数另选。原均值、顺序及R19全加三组基线全部分数/阈值/指标精确重现。','',
'新增的两类向量均实际提取并核对：HARP启发投影用输出头bottom256方向，证据差分用同一答案在有/无可见资料时最终层状态之差。无资料回放只改变检查输入，原生成回答没有重写。全部602份的投影、差分、token ID及坐标代数核验通过；首尾两份原提示回放与R18状态、无资料重复回放均为0差。模型仍是同一Qwen2.5-7B-Instruct NF4。','',
'新增六类LR各用3个C，每折另拟合3个纯Lookback候选，合计105次LR拟合；ReDeEP另有每折27个公式组合，选头/层排序和归一化仅从拟合集计算。每个候选先在校准组分别选整答和窗口阈值，再最大化两项F1中较小的一项，平分依次看窗口F1、窗口precision、较小复杂度/C。所有候选模型与校准记录都保存，没有拿外层分数调参。','',
'**结果指向什么**','',
'- HARP投影或差分向量单独使用都偏弱，和原注意力/NLL联合后才较好，但主方案拟合集0.938、折外0.665，泛化差距仍大。不能据此说所有原始隐藏信息都无效。','- 差分与hidden的PCA维数相同，提供了投影方向对照；HARP局部LR与CORTEX仅差分属于本项目适配，尚未复制它们原本的弱监督MLP、注意力残差或平滑全流程。','- ReDeEP已给予未压缩的全部头层和27组合预算，但此数据上的局部适配仍弱。标准JSD实现与原官方代码的KL写法存在差别，不能把这个数值说成原论文普遍失效。LUMINA原组合还需要IPR，本轮没有冒充完整LUMINA。','- R19联合的整答0.731仍高于本轮主方案0.682；当前不存在一个已经满足两项约0.75且超过全部强基线的最终模型。','',
'下一轮将检验同一Qwen在明确核查“回答是否被资料支持”时的额外内部状态，并区分整答证据核查与局部相对排名。新人工QA数据作为额外基准继续准备；它保留原公开提示（包含缺证据可拒答说明），不能替代原中性提示的情报控制场景。','',
'静态输出头分解约15.1秒、全部HARP投影约15.1秒，无资料提取主循环约195.6秒，正式拟合与汇总约99.8秒；这些为本机阶段计时，不是线上端到端性能承诺。','',
'[完整结果](summary.json) · [方案](../PLAN.md) · [基线适配说明](../BASELINES.md) · [提取核验与成本](EXTRACTION_REVIEW21.md) · [实现](../src/run21.py) · [人工QA准备](../../benchmark_ragtruth_qa/README.md)']
p=OUT/'INDEPENDENT_AUDIT21.json'
if p.exists():lines+=['',f"独立评测审计：{json.loads(p.read_text(encoding='utf-8')).get('status','见文件')}。[核验记录](INDEPENDENT_AUDIT21.json)"]
else:lines+=['','独立评测审计进行中；提取自检与正式运行内的基线精确核对已通过。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(OUT/'REPORT.md')
