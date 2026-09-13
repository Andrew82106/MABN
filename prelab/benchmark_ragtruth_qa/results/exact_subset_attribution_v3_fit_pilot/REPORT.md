# Exact subset attribution v3：fit-only pilot

A/B/C 均为我们方法内部消融，不是论文 baseline。全部数字来自固定 256-group fit OOF；同一批 OOF 标签也用于选阈值，因此只能作开发筛查。

| 读出 | 特征数 | 4-BPE window F1 | answer F1 |
|---|---:|---:|---:|
| A_full_nll_citation | 3 | 0.1650 | 0.4716 |
| B_full_empty_delta_citation | 5 | 0.1698 | 0.4877 |
| C_all_29 | 29 | 0.1698 | 0.4966 |
