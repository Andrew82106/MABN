"""Generate the Chinese six-baseline report from completed audited results only.

CPU/file I/O only. Does not fit, select, score, modify predictions, or edit papers.
The original HGB candidate stays fixed and all comparisons are data-derived.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
RUN = EXT.parent
PROJECT = next(p for p in RUN.parents if p.name == 'intelligence_knowledge_boundary')
COPY_DIR = PROJECT / 'reports/paper/tmp/six_baselines_20260928'
PRIMARY = 'Fusion-window-HGB'
BASELINES = [
    'Lookback-tuned', 'RAGognizer-retrained-NF4', 'ReDeEP-paper-FP16',
    'LUMINA-NF4', 'LettuceDetect-v1-retrained-BF16-forward', 'MiniCheck-FlanT5-FP32-sentence',
]
LABELS = {
    'Lookback-tuned': 'Lookback Lens（验证调参）',
    'RAGognizer-retrained-NF4': 'RAGognizer（同域重训，NF4）',
    'ReDeEP-paper-FP16': 'ReDeEP（论文公式，FP16）',
    'LUMINA-NF4': 'LUMINA（NF4）',
    'LettuceDetect-v1-retrained-BF16-forward': 'LettuceDetect v1（同域重训）',
    'MiniCheck-FlanT5-FP32-sentence': 'MiniCheck-FlanT5（原生句级）',
    PRIMARY: '本文方法（固定 HGB）',
    'Lookback-default': 'Lookback Lens（默认参数，仅补充）',
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def link(path, label=None):
    path = Path(path).resolve()
    return f'[{label or path.name}](<{path.as_posix()}>)'


def num(value, digits=4):
    if value is None:
        return '—'
    assert math.isfinite(value)
    return f'{value:.{digits}f}'


def interval(values):
    a, b = values
    assert math.isfinite(a) and math.isfinite(b) and a <= b
    return f'[{a:+.4f}, {b:+.4f}]'


def interval_description(values):
    a, b = values
    if a > 0:
        return '区间全高于 0'
    if b < 0:
        return '区间全低于 0'
    return '区间含 0'


def strongest_conclusion(results, level, cohort='no_hint'):
    score = lambda name: results['methods'][name]['cohorts'][cohort][level]['f1']
    best = max(score(name) for name in BASELINES)
    winners = [name for name in BASELINES if abs(score(name) - best) <= 1e-12]
    ours = score(PRIMARY)
    level_label = '回答级' if level == 'answer' else '4 词元窗口级'
    names = '、'.join(LABELS[name] for name in winners)
    delta = ours - best
    direction = '高于' if delta > 1e-12 else ('低于' if delta < -1e-12 else '等于')
    statement = (f'{level_label}最强基线为 {names}（F1={best:.4f}）；'
                 f'本文方法 F1={ours:.4f}，{direction}该基线，差值 {delta:+.4f}。')
    if cohort == 'no_hint':
        descriptions = []
        for name in winners:
            d = results['paired_f1_differences'][PRIMARY + ' minus ' + name][level]
            assert abs(d['difference'] - delta) <= 1e-12
            descriptions.append(f'{interval(d["paired_95ci"])}（{interval_description(d["paired_95ci"])}）')
        statement += ' 配对 95% 区间：' + '；'.join(descriptions) + '。'
    return statement, winners


def metric_table(results, cohort):
    lines = ['| 方法 | 回答 P | 回答 R | 回答 F1 | 回答 AUROC | 窗口 P | 窗口 R | 窗口 F1 | 窗口 AUROC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in [*BASELINES, PRIMARY]:
        row = results['methods'][name]['cohorts'][cohort]
        values = [row[level][key] for level in ['answer', 'window']
                  for key in ['precision', 'recall', 'f1', 'auroc']]
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(num(v) for v in values) + ' |')
    return lines


def check_inputs():
    required = [HERE / n for n in ['SIX_BASELINES_RESULTS.json', 'SIX_BASELINES_AUDIT.json',
        'EXTENSION_FROZEN.json', 'WHITEBOX_INPUT_AUDIT.json', 'LETTUCE_INPUT_AUDIT.json', 'MINICHECK_INPUT_AUDIT.json']]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError('Final results/audits are not complete: ' + '; '.join(missing))
    result, audit, frozen, white, lettuce, mini = map(read, required)
    assert audit['status'] == 'PASS' and audit['result_sha256'] == sha(required[0])
    assert result['manifest_sha256'] == sha(required[2])
    assert result['primary_candidate'] == PRIMARY == frozen['primary_candidate']
    assert result['test_answers_primary'] == 155 and result['test_answers_all'] == 279
    assert result['test_groups_primary'] == 75 and result['bootstrap_samples'] == 2000
    assert result['bootstrap_seed'] == 20260928 and result['test_previously_released']
    assert set(result['methods']) == set(BASELINES + [PRIMARY, 'Lookback-default'])
    entries = {r['name']: r for r in frozen['methods']}
    assert entries.keys() == result['methods'].keys()
    for name, entry in entries.items():
        assert audit['methods'][name]['status'] == 'PASS'
        assert entry['thresholds'] == result['methods'][name]['thresholds']
        assert entry['pooling'] == result['methods'][name]['pooling']
        for part in ['validation', 'test']:
            assert sha(entry[f'{part}_predictions']) == entry[f'{part}_predictions_sha256']
        for cohort, count in [('no_hint', 155), ('all', 279)]:
            assert result['methods'][name]['cohorts'][cohort]['answer']['n'] == count
    assert white['status'] == lettuce['status'] == mini['status'] == 'PASS'
    assert set(white['completed_score_checks']) == {'lumina', 'redeep'}
    assert set(lettuce['completed_prediction_checks']) == set(mini['completed_prediction_checks']) == {'validation', 'test'}
    # The independent native-score audits must refer to the final frozen files.
    for name, key in [('ReDeEP-paper-FP16', 'redeep'), ('LUMINA-NF4', 'lumina')]:
        for part in ['validation', 'test']:
            assert white['completed_score_checks'][key]['scores'][part]['predictions_sha256'] == entries[name][f'{part}_predictions_sha256']
    for part in ['validation', 'test']:
        assert lettuce['completed_prediction_checks'][part]['sha256'] == entries[BASELINES[4]][f'{part}_predictions_sha256']
        assert mini['completed_prediction_checks'][part]['prediction_sha256'] == entries[BASELINES[5]][f'{part}_predictions_sha256']
    return result, audit, frozen, entries, required


def actual_receipts(entries):
    result = {}
    for method, folder, stem, field, count_field in [
        (BASELINES[2], EXT / 'whitebox/redeep', '{part}_predictions.receipt.json', 'predictions_sha256', 'records'),
        (BASELINES[3], EXT / 'whitebox/lumina', '{part}_predictions.receipt.json', 'predictions_sha256', 'records'),
        (BASELINES[4], EXT / 'semantic/lettuce_v1_run', '{part}_PREDICTION_RECEIPT.json', 'predictions_sha256', 'rows'),
        (BASELINES[5], EXT / 'semantic/minicheck_run', '{part}_PREDICTION_RECEIPT.json', 'predictions_sha256', 'answers')]:
        result[method] = {}
        for part, count in [('validation', 271), ('test', 279)]:
            path = folder / stem.format(part=part)
            receipt = read(path)
            assert receipt[field] == entries[method][f'{part}_predictions_sha256']
            assert receipt[count_field] == count and not receipt['test_labels_read']
            result[method][part] = {'path': path, 'value': receipt}
    return result


def timings(receipts):
    """Report recorded phases, never infer absent wall times from file mtimes."""
    rows, evidence = [], []

    def add(name, phase, seconds, source, note=''):
        assert seconds is None or seconds >= 0
        rows.append([name, phase, '未单列' if seconds is None else f'{seconds:.2f} 秒（{seconds / 60:.2f} 分）', note, link(source)])
        evidence.append(Path(source))

    path = RUN / 'baseline_audit/lookback_run_v2/tuned/CONVERGENCE.json'
    add('Lookback Lens', '既有选中 C 的拟合', sum(r['seconds'] for r in read(path)), path, '不含共享 Llama 特征提取及其他 C 候选')
    path = RUN / 'baseline_audit/predictions/lookback_tuned_test/PREDICTION_RECEIPT.json'
    add('Lookback Lens', '既有 279 条测试打分', read(path)['elapsed_seconds'], path, '使用冻结特征缓存')
    path = RUN / 'baseline_audit/ragognizer_run_v1/COMPLETE.json'
    add('RAGognizer', '既有 5 轮训练流程', read(path)['elapsed_seconds'], path, '使用其训练完成记录')
    path = RUN / 'baseline_audit/ragognizer_test_predictions_v1/PREDICTION_RECEIPT.json'
    add('RAGognizer', '既有 279 条测试预测', read(path)['elapsed_seconds'], path)
    for key, display, expected in [('redeep', 'ReDeEP', 1842), ('lumina', 'LUMINA', 550)]:
        base = EXT / 'whitebox' / key
        path = base / 'EXTRACTION_COMPLETE.json'
        complete = read(path)
        assert complete['records'] == expected and complete['status'] == 'complete'
        add(display, f'{expected} 条特征提取进程记录', complete.get('elapsed_seconds'), path, '该进程记录范围；含载入开销，断点续跑时不等同全程')
        feature_receipts = sorted((base / 'features').glob('*/*.json'))
        assert len(feature_receipts) == expected
        total = sum(read(p)['seconds'] for p in feature_receipts)
        add(display, f'{expected} 条逐样本计算合计', total, base / 'features', '逐条 receipt 累加；不含模型载入/样本间调度')
        evidence.extend(feature_receipts)
    path = EXT / 'semantic/lettuce_v1_run/TRAIN_COMPLETE.json'
    complete = read(path)
    assert complete['epochs'] == 6
    add('LettuceDetect', '6 轮训练流程', complete['elapsed_seconds'], path, '含各轮验证与检查点保存')
    for name, display in [(BASELINES[4], 'LettuceDetect'), (BASELINES[5], 'MiniCheck')]:
        for part, label in [('validation', '271 条验证预测'), ('test', '279 条测试预测')]:
            item = receipts[name][part]
            add(display, label, item['value'].get('elapsed_seconds'), item['path'], '按实际 receipt；未记录的时段不估算')
    path = RUN / 'fusion/nonlinear_window_v1/COMPLETE.json'
    add('本文 HGB', '既有 4 候选融合训练/选择', read(path)['seconds'], path, '仅融合器，非语义/白盒分支总训练时长')
    return rows, evidence


def build_report(result, audit, frozen, entries, receipts, source_paths):
    main_a, answer_winners = strongest_conclusion(result, 'answer')
    main_w, window_winners = strongest_conclusion(result, 'window')
    total_a, _ = strongest_conclusion(result, 'answer', 'all')
    total_w, _ = strongest_conclusion(result, 'window', 'all')
    cohorts = result['methods'][PRIMARY]['cohorts']
    lines = ['# 六基线共同评测报告', '',
        '本次比较 6 种独立基线与此前固定的 HGB 融合方法。Lookback 的默认版单独列为补充，不增加基线数量。', '',
        main_a, '', main_w, '',
        '差值均为“本文方法减基线”。区间含 0 时，只报告本次分数差，不认定提升已经得到明确统计支持。', '',
        '## 数据与评测口径', '',
        '数据来自 F4biian/RAGognize 官方 train 的 Llama-2-7B-chat 输出，按来源组划分为训练 1292 条、验证 271 条、测试 279 条；使用发布的有效自动错误范围。', '',
        f'主测试为原提示没有资料不足暗示的 155 条回答、75 个来源组，共 {cohorts["no_hint"]["window"]["n"]} 个有效窗口；全量 279 条、{cohorts["all"]["groups"]} 组、{cohorts["all"]["window"]["n"]} 个窗口作补充。', '',
        '共同窗口为原始 Llama 分词后的连续 4 个词元，步长 1；排除全无字母数字的窗口，窗口内任一有效错误词元即为正例。回答风险取窗口最高分。P 为精确率，R 为召回率，F1 为两者的综合指标，AUROC 衡量风险分数的排序能力。', '',
        'Lookback、RAGognizer 复用本轮既有同域训练的冻结预测并重评分；ReDeEP、LUMINA、LettuceDetect、MiniCheck 在本次扩充中实际适配运行。HGB 的检查点、预测和阈值保持固定。测试此前已经解封，本轮属于同一固定划分上的补充比较。', '',
        '## 主测试：155 条', '', *metric_table(result, 'no_hint'), '',
        '## 全量补充：279 条', '', *metric_table(result, 'all'), '', total_a, '', total_w, '',
        '## 验证选择与阈值', '',
        '新增方法的 mean4／lexical max4 适配仅在无暗示验证集 135 条上选择，按“两级 F1 的较小值、窗口 F1、窗口精确率”依次比较，并列优先 mean。回答与窗口阈值分别最大化验证 F1，并列依次取精确率、高阈值。表内分数均为风险方向，越高越可疑。', '',
        '| 方法 | 窗口分数 | 回答阈值 | 窗口阈值 | 验证回答 F1 | 验证窗口 F1 |',
        '|---|---|---:|---:|---:|---:|']
    for name in [*BASELINES, PRIMARY]:
        entry = entries[name]
        val = entry['validation']['cohorts']['no_hint']
        pooling = '直接预测共同窗口' if entry['scores_are_windows'] else ('原始4词元均值' if entry['pooling'] == 'mean' else '有效词元最大值')
        lines.append('| ' + ' | '.join([LABELS[name], pooling, num(entry['thresholds']['answer'], 10),
            num(entry['thresholds']['window'], 10), num(val['answer']['f1']), num(val['window']['f1'])]) + ' |')
    lines += ['', '## 与各基线的配对比较', '',
        '按主测试的 75 个来源组进行 2000 次有放回重采样；同一次抽样对所有方法一致，随机种子为 20260928。下列为 F1 差值及其 95% 百分位区间。', '',
        '| 基线 | 回答 F1 差 | 回答 95% 区间 | 窗口 F1 差 | 窗口 95% 区间 |',
        '|---|---:|---|---:|---|']
    for name in BASELINES:
        differences = result['paired_f1_differences'][PRIMARY + ' minus ' + name]
        a, w = differences['answer'], differences['window']
        lines.append(f'| {LABELS[name]} | {a["difference"]:+.4f} | {interval(a["paired_95ci"])}（{interval_description(a["paired_95ci"])}） | '
                     f'{w["difference"]:+.4f} | {interval(w["paired_95ci"])}（{interval_description(w["paired_95ci"])}） |')
    lines += ['', '## 方法身份与运行方式', '',
        '| 方法 | 保留的方法与训练来源 | 精度及原生粒度 |', '|---|---|---|',
        '| Lookback Lens | 注意力比率、原生8词元平均、逻辑回归；本轮1292条重新拟合，C只由验证选择 | NF4 Llama回放；原生8词元窗口 |',
        '| RAGognizer | 原LoRA、中间层MLP、语言模型损失与风险损失联合训练；通用Llama初始化，本轮1292条训练5轮 | NF4基座、BF16计算；原生词元 |',
        '| ReDeEP | 使用论文ECS/JSD公式；原始训练词元Pearson选择头/层，MinMax仅fit拟合；固定1头、10层、α=1、β=0.2 | FP16及CPU卸载；原生词元 |',
        '| LUMINA | 保留官方IPR与top100余弦MMD组合；不训练检测器，替换资料仅来自训练集并固定抽取 | NF4、BF16计算、FP32归约；原生词元 |',
        '| LettuceDetect v1 | 保留官方v0.1.6成对输入、两类词元交叉熵及AdamW；通用ModernBERT-large初始化，本轮1292条训练6轮，验证选轮次 | 参数/梯度/优化器FP32，前向BF16；原生ModernBERT词元 |',
        '| MiniCheck | 官方冻结FlanT5-large检查点；保留doc+claim模板、500词资料分块与最大支持概率；仅验证校准阈值 | FP32；原生句级 |',
        '| 本文HGB | 固定既有完整资料分支、白盒分支与17项窗口特征；融合器使用训练集来源组折外预测拟合 | 原结果冻结；共同4词元窗口 |', '',
        'MiniCheck 的 doc 仅为检索资料，claim 仅为原回答中的一句话；问题不作为证据或待核查断言输入。句级风险为 1−最大支持概率，再按字符对应到共同词元坐标，因此该项仍是句级方法的窗口适配。', '',
        'ReDeEP 的论文公式与其发布代码有差异，本轮固定采用论文公式；LUMINA 保留已锁定官方代码公式。二者使用共同原始输入，不额外改写问题或截断回答。LettuceDetect/MiniCheck 的新坐标转换保留空白词元的原生重叠或固定最近邻分数，未将其硬设为零。', '',
        '## 实际耗时', '',
        '下表逐项注明计时范围，不把共享特征、分支训练和单次打分当作同一种成本。时间来自实际完成记录和逐样本 receipt。', '',
        '| 方法 | 计时范围 | 耗时 | 说明 | 来源 |', '|---|---|---:|---|---|']
    timing_rows, timing_sources = timings(receipts)
    lines += ['| ' + ' | '.join(row) + ' |' for row in timing_rows]
    lines += ['', '## 可复核文件', '',
        f'审计状态：{audit["status"]}。共同指标、原预测哈希、逐方法输入/坐标审计及报告来源索引均保留。', '',
        '| 方法 | 验证预测 | 测试预测 |', '|---|---|---|']
    for name in [*BASELINES, PRIMARY]:
        lines.append(f'| {LABELS[name]} | {link(entries[name]["validation_predictions"], "完整验证JSONL")} | {link(entries[name]["test_predictions"], "完整测试JSONL")} |')
    lines += ['', f'- 完整共同结果：{link(HERE / "SIX_BASELINES_RESULTS.json")}。',
        f'- 统一审计：{link(HERE / "SIX_BASELINES_AUDIT.json")}。',
        f'- 冻结模型/预测/阈值及完整哈希：{link(HERE / "EXTENSION_FROZEN.json")}。',
        f'- 原生分数审计：{link(HERE / "WHITEBOX_INPUT_AUDIT.json")}、{link(HERE / "LETTUCE_INPUT_AUDIT.json")}、{link(HERE / "MINICHECK_INPUT_AUDIT.json")}。',
        f'- 完整来源路径、SHA-256与计时依据：{link(EXT / "REPORT_SOURCES.json")}。', '',
        '| 关键脚本/配置 | SHA-256 |', '|---|---|']
    key_files = [HERE / 'generate_report.py', HERE / 'evaluate_extensions.py', HERE / 'new_methods.json',
                 EXT / 'whitebox/run_whitebox.py', EXT / 'whitebox/CONFIG.json',
                 EXT / 'semantic/lettuce_v1.py', EXT / 'semantic/minicheck_official.py', EXT / 'semantic/common.py']
    for path in key_files:
        lines.append(f'| {link(path)} | `{sha(path)}` |')
    default = result['methods']['Lookback-default']['cohorts']['no_hint']
    lines += ['', f'补充：Lookback默认参数版本的主测试回答F1为 {default["answer"]["f1"]:.4f}，窗口F1为 {default["window"]["f1"]:.4f}；该版本不计入6种独立基线。', '']
    source_paths += key_files + timing_sources
    for method in receipts.values():
        source_paths += [item['path'] for item in method.values()]
    sources = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'primary_candidate': PRIMARY, 'six_baseline_names': BASELINES,
        'strongest_baseline_by_main_answer_f1': answer_winners,
        'strongest_baseline_by_main_window_f1': window_winners,
        'test_previously_released': True,
        'source_files': {str(p.resolve()): sha(p) for p in dict.fromkeys(source_paths) if p.is_file()},
        'artifact_hashes_from_frozen_manifest': {name: entries[name]['artifact_hashes'] for name in [*BASELINES, PRIMARY]}}
    return '\n'.join(lines), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result, audit, frozen, entries, sources = check_inputs()
    receipts = actual_receipts(entries)
    report, source_index = build_report(result, audit, frozen, entries, receipts, sources)
    destination = EXT / 'RESULTS_ZH.md'
    copy = COPY_DIR / 'RESULTS_ZH.md'
    write_json(EXT / 'REPORT_SOURCES.json', source_index)
    COPY_DIR.mkdir(parents=True, exist_ok=True)
    for path in [destination, copy]:
        path.write_text(report, encoding='utf-8')
    assert sha(destination) == sha(copy)
    write_json(EXT / 'REPORT_OUTPUTS.json', {'status': 'complete', 'report_sha256': sha(destination),
        'reports': [str(destination), str(copy)], 'source_index_sha256': sha(EXT / 'REPORT_SOURCES.json'),
        'generator_sha256': sha(__file__), 'no_paper_document_modified': True})
    print(json.dumps({'reports': [str(destination), str(copy)], 'sha256': sha(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
