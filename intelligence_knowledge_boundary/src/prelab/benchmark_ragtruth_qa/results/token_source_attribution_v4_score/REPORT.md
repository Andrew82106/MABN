# Token source attribution v4

选中 `token_attr_full_nll_lr`。fit source-group OOF 窗口/整答 F1：0.509860/0.728997；首次冻结阈值下的 cal：0.535300/0.803738。此后为与公共开发表同口径而做的只读、事后 cal-F1Opt 诊断也仅为0.537714/0.829876，因此未超过当前候选0.690281/0.891089，予以淘汰；事后阈值不得用于选择或部署该模型。

模型和fit阈值冻结后只进行了一次主cal评分；随后只重放已有分数计算类型与公共F1Opt诊断，没有重训或改模型。本模块没有打开旧 official test；baseline 未改。运行后不变量复核确认：fit/cal 材料组交集为0，两部分的答级标签都与其窗口标签最大值完全一致，且本次 calibration started/complete 文件同时存在。

窗口几何是精确4-BPE，但NLI部分仍由词元所属的完整微主张提供，再由该词元的来源归因加权；因此应称“claim-conditioned token特征”，不能称完全局部语义特征。类型诊断显示主要失败仍是Evident Conflict，固定阈值只检出210/1,028个相关风险窗口。

见 `POST_RUN_INVARIANT_AUDIT.json`、`POSTHOC_COMMON_CAL_F1OPT.json`、`TYPE_DIAGNOSTIC.md` 和 `../../research/token_source_attribution_v4_audit/REPORT.md`。
