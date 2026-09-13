# FAVA 局部修复对：候选导出完成

**10040对全部保留，均只修改原错误答案的一处；两个版本分成独立模型输入。** 没有把修后整答标成正确，没有分词、模型前向、训练或读取QA金标/校准/测试。

| 项目 | 实际数量 |
|---|---:|
| 局部候选对 | 10040 |
| entity / relation | 5335 / 4705 |
| 独立输入行：原版＋单处修复版 | 20080 |
| 涉及原回答 / 原材料组 | 5275 / 5238 |
| 两侧字符范围、范围外不变、逆向恢复检查 | 10040全部通过 |
| 结构可用 / 结构标记不适用 | 10036 / 4，**未删除任何对** |

4个标记项是修复目标只有句号或逗号，其中1项原目标也只有标点，来自原行6655、17103、27188。它们没有被强塞进词级监督。详见[STRUCTURAL_FLAGS.json](STRUCTURAL_FLAGS.json)。该固定10040子集中没有空Reference段；预先定义了空资料/无字母数字目标的标记规则，也做了CPU小例检查。

## 文件与输入约束

- [candidate_pairs.jsonl](candidate_pairs.jsonl)：每对的原response/source/group、作者类型、两侧独立字符target、输入ID/哈希、结构eligibility、建议基础权重。范围为各自原文Python字符半开区间；两侧长度可不同。
- [model_inputs.jsonl](model_inputs.jsonl)：每行只有`input_id`和`model_inputs`。后者严格只有`retrieved_passages`、空`question`、**一个版本**的`response`。五段资料及原编号逐字复用；没有另一版本、纠错markup、目标标签或作者检查指令。
- [provenance.jsonl](provenance.jsonl)：原官方train行号/版本、prompt与completion哈希、原类型节点先序号、原completion内节点字符范围和哈希。定位不用首次字符串查找。
- [original_answer_index.jsonl](original_answer_index.jsonl)、[material_group_index.jsonl](material_group_index.jsonl)：原答多对和原资料组映射，供后续分组与权重使用。

作者建议的`preferred`只是**有噪声的局部修复**，不是独立核验的人标真答案。`outside_targets`明确为未知；导出没有`answer_label`、整答正确标签或修后全词元标签。其他原错误原样保留，也没有使用整篇编辑投影。

## 来源隔离与权重

逐条核对所属旧`auxiliary_fava_v2`的7482已隔离候选、原source/group、资料正文/哈希和未隔离状态，绑定旧complete、manifest、源材料隔离报告。**继承原非空SHA/20连续词及共享Reference连通组隔离结论，没有重新扫描QA来源或答案。** 原隔离无法排除一切改述重叠的限制照旧；5238是保守材料组，不能称独立实体/事件。材料中干扰段也可能连接原答，保留这些连接。

建议基础权重为`1 / (5238 × 该组原答数 × 该原答局部对数)`；同一对的两版本作为一个单元。已核每组质量相同，总和1；单原答最多17对。这里只导出候选权重建议，未拟合类别因子或损失。如果后续仅用结构可用对，应明确重算参与组/原答/对的分母，不能直接删4行后冒称仍完全等权。

后续公平对照是：**同一批局部对上的局部银标BCE，与局部银标BCE＋相对排序损失**。两边保持训练预算、输入、分组和QA阶段一致；只在对应局部赋予银标监督，不通过修复版整答标签扩大负例。本轮没有启动这个训练。

## 实际检查

导出 actual exit0，约3.4秒。另用独立流式markup字符计数器重建全部7482候选的可选节点，逐条回放10040对的原文、两侧target、模型输入白名单、补丁逆向、类型、原分组和权重，actual exit0，约3.2秒。[INDEPENDENT_QUALITY_CHECK.json](INDEPENDENT_QUALITY_CHECK.json)

初次审计遇到Windows相对路径键与正斜杠键不一致，留下[AUDIT_FAILURE_01.json](AUDIT_FAILURE_01.json)；只修正审计器路径键规范化，未改导出数据或源文件，随后全量通过。独立重建指实现独立于导出树遍历，不冒称人工逐条语义认证。

数据导出快照见[manifest.json](manifest.json)，最终状态及报告/审计哈希见[complete.json](complete.json)。旧FAVA/QA文件未改。
