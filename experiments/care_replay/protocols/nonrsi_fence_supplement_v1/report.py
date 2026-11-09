"""Readable export of already computed supplemental results, with provenance."""
from collections import Counter
from .runner import OUT, EVAL, original
from protocols.nonrsi_cross_model_v1.common import load, save, digest
from protocols.nonrsi_cross_model_v1 import freeze


def export(config):
    result = load(EVAL / 'analysis.json'); old = load(original.OUTPUT / 'analysis.json')
    derived = load(OUT / 'DERIVED_GENERATION.json')
    before = {(r['model'], r['contrast']): r for r in old['primary']}
    models = [m['id'] for m in config['models']]
    short = dict(zip(models, ['Luna', 'Haiku', 'Qwen']))
    unchanged = []
    for model in models:
        if derived['counts'][model]['newly_valid'] == 0:
            for row in [r for r in result['primary'] if r['model'] == model]:
                if row != before[(model, row['contrast'])]:
                    raise RuntimeError('Primary result changed for a model with unchanged decisions')
            unchanged.append(model)
    if result['draws_sha256'] != old['draws_sha256']:
        raise RuntimeError('Bootstrap draws changed')
    lines = ['实验一补充分析：已有响应去除完整外层代码块后的敏感性分析', '',
        '这是事后补充分析，原严格 JSON 结果仍保留且不被替换。没有新增模型请求，也未运行实验二、三、四。',
        '处理规则统一用于全部 660 个已有响应：仅移除一个完整外层 json/无语言代码块，不修补 JSON、字段、取值、规则数量或内容。',
        '其余响应信封、五策略、字段目录及数值范围校验不变，仍未通过的槽继续回退。',
        '沿用原候选池、随机种子、50 个校准/100 个留出种子、选择与筛选规则及描述性 bootstrap。',
        '原有模式和轨迹直接复用；只对新增可用模式做本地回放，并核对重新计算的基线与原记录完全一致。',
        '所有目标的校准选择锁定后才进入留出评估。补充分析使用过往已见的留出种子，不是新的独立验证。', '', '响应有效性：']
    for model in models:
        c = derived['counts'][model]
        errors = Counter(reason for p in derived['provenance'] if p['model'] == model for reason in p['supplemental_reasons'])
        lines.append(f"{short[model]}：外层代码块移除 {c['outer_fences_removed']}；新增有效 {c['newly_valid']}；有效 {c.get('VALID', 0)}/220；仍回退 {c.get('INVALID_ANCHOR_FALLBACK', 0)}。")
        if errors:
            lines.append('剩余校验原因：' + str(dict(errors)))
    lines += ['', '四个固定任务（相对共同优化基线的 ΔScore）：', '任务 | Luna | Haiku | Qwen']
    by = {(r['model'], r['contrast']): r for r in result['primary']}
    for cell in [c for c in config['generation_cells'] if c['model'] == models[0] and c['core_fixed_task']]:
        label = cell['cell'] + ' minus common BO'
        lines.append(cell['cell'] + ' | ' + ' | '.join(f"{by[m,label]['delta_score']['mean']:+.5f}" for m in models))
    lines += ['', '八条迁移路线（完整源信息 F 相对目标信息 T 的 ΔScore）：', '路线 | Luna | Haiku | Qwen']
    for route in config['routes']:
        label = route['route_id'] + ' F minus T'
        lines.append(route['route_id'] + ' | ' + ' | '.join(f"{by[m,label]['delta_score']['mean']:+.5f}" for m in models))
    lines += ['', '全部主要比较及名义 95% 敏感性区间（原严格结果 → 补充结果）：']
    for row in result['primary']:
        previous = before[row['model'], row['contrast']]['delta_score']['mean']
        value = row['delta_score']; interval = value['nominal_95']
        lines.append(f"{short[row['model']]} | {row['contrast']} | {previous:+.7f} → {value['mean']:+.7f} | [{interval['low']:+.7f}, {interval['high']:+.7f}]")
    lines += ['', '预定随机规则对照（仅有效父规则库）：']
    for row in result['random_controls']:
        value = row['summary']['delta_score'] if row['summary'] else None
        lines.append(f"{short[row['model']]} | {row['target']} | {row['family']} | 有效父库 {row['valid_parents']} | " +
                     ('NA' if value is None else f"ΔScore {value['mean']:+.7f}; 名义95% [{value['nominal_95']['low']:+.7f}, {value['nominal_95']['high']:+.7f}]"))
    costs = load(OUT / 'SOURCE_AUDIT.json')
    lines += ['', '费用与边界：',
        f"新增实验模型请求 0，新增模型费用 0。历史已知费用仍为 {costs['known_model_cost_usd']} 美元；历史异常请求保守预留 {costs['unknown_charge_holds_usd']} 美元，不能视为确认支出。",
        'ΔScore 是按固定目标参考尺度归一化后的平均差，不是提升百分比；正数平均更好，负数平均更差。',
        '不把任务合并成总体胜率。名义 bootstrap 区间仅作敏感性展示，不作显著性、等效性或整体错误率控制声明。',
        '解析策略是在看到原格式失败后增加的，结果只支持本任务中的格式敏感性诊断，不能冒充预注册主结果。',
        '新旧结果相同的模型：' + ', '.join(short[m] for m in unchanged),
        '原始主结果：' + str(original.OUTPUT / 'FINAL_REPORT.txt'),
        '本组完整数值：' + str(EVAL / 'analysis.json')]
    path = EVAL / 'SUPPLEMENT_REPORT.txt'
    if path.exists() and path.read_text(encoding='utf-8') != '\n'.join(lines) + '\n':
        raise RuntimeError('Human supplement report changed')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(OUT / 'SUPPLEMENT_ACCOUNTING.json', {'post_hoc': True, 'new_model_requests': 0,
        'new_model_cost_usd': '0', 'historical_accounting': costs,
        'unchanged_model_primary_results_verified': unchanged,
        'original_bootstrap_draws_reused': True, 'analysis_sha256': freeze.file_hash(EVAL / 'analysis.json'),
        'report_sha256': freeze.file_hash(path)})
