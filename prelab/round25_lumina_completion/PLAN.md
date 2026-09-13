# R25：补齐 LUMINA 官方代码公式

已固定主方案，当前仅完成 CPU 准备，未加载大模型或占用 GPU。范围是 R16 actual train 的 301 题、278 组、602 份原回答、14,968 个原 response token；原 validation/test 内容和任何风险标签均未用于提取。

原输入回放直接复用 R7 `collect_response_hidden`、`final_statistics`、`ipr_from_hidden`。IPR 包含固定官方代码全部 28 个输出层及末层再次 norm 的约定。MMD 复用官方 top-100 未归一化余弦核公式。

主干预同时应用 R19 两个已冻结的正文 token mask，两个来源分别沿用原 donor 流，不重新选择来源。问题、标题、正文边界 token、模板、输入长度、答案 IDs 和位置保持；这叫联合正文干预，不是自然全篇检索替换。全 602 条实际 mask 和联合 prefix 已重新核对。

固定主分数：`0.5*IPR - 0.5*MMD_joint_body`。附加版本以 R19 两列单来源 MMD 的同权均值代入同一公式。可在后续评测中另外报告 calibration 选择 lambda∈{.25,.5,.75} 的次要版本；不得代替固定 .5 主结果。当前没有新 LR/PCA，也没有编写拟合评测脚本。

## 已验证

- 602 条新联合计划保留原 prefix/response ID 和 offset；两个 mask 不交叉，联合改变严格等于旧单源改变并集。
- R19 全602条MMD缓存的行、原生成、计划、签名、NPZ哈希、token IDs、offset均匹配。
- 独立7项CPU检查全通过。IPR对固定官方代码最大差8.20e-8，MMD对原核公式最大差2.68e-7；包含非单位norm权重、P−1首token、未来token不影响此前分布、mask非法输入拒绝、mix负值保留等。
- `cpu_selfcheck.json`绑定当前extract25和测试源码。没有实际7B GPU数值或速度结论。

## GPU运行接口（由根代理排队）

先QA，再R23，最后R25。当前不要与其他任务并发使用GPU。

```powershell
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/extract25.py selfcheck
prelab\.venv\Scripts\python.exe -X utf8 prelab/round25_lumina_completion/src/extract25.py run
```

`run`自身也会检查CPU记录、完成两条固定长度工程样例的GPU自检，再提取全部602。工程样例按最短/最长原输入+回答token长度选，完全不看标签；通过输出直接复用于正式缓存。自检必须重复原统计/IPR精确一致、identity MMD≤1e-7、重算单源MMD与R19最大绝对差≤1e-6。失败停止，不放宽阈值、不改冻结协议。

常规每答两次backbone前向：原输入取得IPR、双正文联合输入取得MMD。两个工程样例各增加一次原回放与两次单源回放，总工程额外6次。IPR另有28层分块词表投影，实际耗时待测。

## 缓存接口

每行`data/features/<row_id>.npz`及同名JSON。逐词元数值均float32，原ID为int64，offset为int32：

- token_ipr [N]
- token_mmd_single [N,2]（原R19数值）
- token_mmd_single_mean [N]
- token_mmd_joint_body [N]
- token_lumina_single_mean [N]
- token_lumina_joint_body [N]（主公式）
- token_ids [N]、response_token_offsets [N,2]、token_start/token_end [N]

sidecar含source_generation_sha256、arrays_sha256、extraction_signature_sha256、joint_plan_sha256和旧R19缓存哈希。manifest用complete、completed_count、records下json/json_sha256/npz/npz_sha256。续跑逐行核身份、坐标、原MMD与混合公式；已有完整manifest不无故刷新。

源数据、旧模型输出、旧特征及标签保持冻结。当前数据是反复开发的助手标注资料支持场景，不能将这次补充直接称为新独立测试或论文全量复现。
