# Round4 completed; target not reliably achieved

## FINAL STATE — overrides every historical live-state paragraph below

No running experiment or watcher remains. Round4 first100-source test, subsequent50-source confirmation, numerical state replay, independent metric/data audits, reports and all150 example pages are complete. No goal object exists and no subagents were used.

User requested an upgrade toward F1>0.7. Primary metric was explicitly scoped to answer-level error-class F1, not token F1 or AUROC. First-stage validation-selected primary was whole_facts / abc_base_None_0.01: testF1=0.645 (failed). Prespecified alternatives: sentences3/raw_B=0.709; sentences_facts/abc_base_hidden16_balanced_0.01=0.703. Do not hide the failed preselected primary or call the raw score a newly trained probe.

Selected those two alternatives using the now-viewed first100 as development, froze existing weights and thresholds, and independently confirmed on50 entirely new sources excluding all prior684. Llama-only had only16 eligible positive sources remaining; all generators had22. Two infeasible preparation attempts (50Llama/20positive;100mixed/40positive) ended before any data/protocol freeze or scoring. Final50mixed-generator sources20error30clean preserve40% errors within each generator: GPT3.5(2/3),GPT4(4/6),Llama13B(8/12),Llama70B(2/3),Llama7B(4/6). These are existing dataset output origins, not additional model calls.

Confirmation primary learned internal-state probe: F1=0.588235, P=.714286,R=.5,TP10,FP4,FN10,TN26,tokenF1=0.130 rounded. Raw3sentence controlF1=.5625; baseF1=.584615 (TP19,FP26,FN1,TN4). No retuning or reselection after confirmation. Both predeclared targets failed; first alternatives exceeded0.7 only once. Do not claim stable success or pool the two stages as one untouched test. Confirmation50 has different original-generator mix; broad confidence interval [.364,.762]. Localization remains poor.

Diagnosis: selected sentences covered15/20 error answers, facts11/20. Of10 false negatives,5 had no erroneous sentence selected and5 had the erroneous sentence selected but were still missed. Example1627: erroneous claim 'Booker had enlisted in the Army in 2014'; rule only targeted 'Army', missing the critical relation. Further work would need better coverage and full factual-relation checks, plus new independent labelled data rather than tuning on these tests.

Files: round4/README.md links both reports; round4/results/REPORT.md retains all12 first-stage methods,630 validation candidates and frozen primary; round4/confirmation/results/REPORT.md preserves all3 confirmation methods and sampling limitations. Both audit.json files passed and code hashes match final scripts. readout_audit.json replayed6 training targets in exact original batches: all hidden/prob differences0, hook logits unchanged. HTML article counts100/50 and all report links verified; first-stage comparison.png visually inspected. All runs used same local frozen Qwen7B NF4, no external LLM, no free generation, no new dependencies. RootprelabREADME updated. Original prior rounds preserved.

Historical logs below are superseded; do not resume their sessions or rerun data/model selection.

## LATEST LIVE STATE

Watcher **functions.exec cell85** now owns live exec session99293. Use functions.wait(cell_id="85") ONLY; do not concurrently write_stdin on99293. It polls45sec and notifiesreadoutprogress, and returnsmainexitstatus. It does NOT launchaudits. Lastobservedreadouts2900/4252fresh plus490reused =3390/4742saved. Mainpipeline automaticallyfits/evaluatesafterreadouts. No testmetricsviewed.

Allcode nowimplemented: common4,prepare,cache_test,make_queries,readouts,fit,evaluate,audit,audit_readouts,report. Need run audit_readouts.py onlyaftercurrentGPUprocessends, thenaudit.py, thenreport.py. Numericalauditorreplays6train targets intheirexactoriginalfreshbatches andverifiesunchangedhooklogits, cachedhidden/probs tolerances. CPUauditindependentlyuses sklearnF1/PR curves toverifyallvalthresholdmaxima,source/label/OOF/fitmeans/selection/testconfusionmatrices.

Beforefitting/test, addedsentences_facts andall_checks combinationsbecausevalcoverage41/45forthreesentencesversus29/45forthreefacts; train148/175versus83/175. Noadditionalqueriesneeded;protocolamendmentrecorded. Now8scopesx3aggregatorsx24unitmodels+6basecandidates=582. fit.py nowpredictsonlyvalidationunitfeatureswhilechoosing; finaltestunitpredictionscomputedinevaluateafterselectionartifactwritten. Basicindependentthresholdexample passed(2TP1FP=>F1.8 atthreshold.6); allscriptssyntaxcheckedbeforelastreportaddition, rerunASTasneeded. common4appendsoldsrcsothenewfit/evaluatearenotshadowed.

ReportcodeproducesexplicitanswerF1/P/R/confusions, separatestate/localtokenF1, all100exampleheatmaps,graph andlimitations. Stillneedsactualresultsinterpretation, checkedvisualrender,source/links/provenanceaudit,READMEfinalstatus. Needactuallyachieve user'sF1>0.7onthefrozenselectedmethodbeforeclaimingtarget; positiveother-testwinnerdoesnotcount.

## Earlier notes (counts superseded above)

User asks to upgrade factual-span checks and get F1 above0.7. No active goal object (get_goal returnednull); ordinary authorized task, do not invent goal status. No subagents. Work directly underprelab using authorized Python. Doc guide routing,sections1-8/10-11 andPython exception read; no doc modifications required.

Primary clarified to user: ANSWER-level binaryF1, errorpositive, val-only threshold/model choice; not AUROC or wordF1. Need implement actual fact targeting and widercoverage, reportlocalizationseparately andhonestprecision/recall, not just lowerthreshold/cherry-picktest.

Prepared frozenround4 data: old584 distinctsources explicitlydevelopment, stratifiedrepartition464train120val;100 new testgroups entirely excludedfromprior584,40error60clean. Testoriginalgeneratorquotas7B(14error21clean),13B(14/21),70B(12/18), allocationseed20260911withmaxflowenforcingoneoutputpersource. Labelsreleasedhuman EvidentConflict/fullycleanonly. raw943news/geneligibleunseen61positivegroups across3Llama; no score-basedfiltering. Allnew100basefeaturesdone in113.5sec. Previoussession53981terminalexit0.

LIVE session99293: make_queries.py -> readouts.py -> fit.py -> evaluate.py, eachlogsunderround4/logs, stoponerror. Querypreparationcomplete4742queries. BaseattentionLR C.1 trainednew464; strictsource3fold OOF trainselection, fulltrainval/test; additionalattention+supportLRbaseline. Top3sentences, onehighest-risk deterministicname/number/role/risk-wordspanpersentence, pluswhole-summaryquery. No labelsinqueries/prompts. Readouts sameQwenNF4, 0generation, ABC probs and4promptendlayers; exactround3sentencequeriesmayreuseread-onlycachewithprovenance. Otherqueriesfullforward,batch2short/batch1long. Can resume byqueryid.

fit.py andevaluate.py implemented, ASTparsepassedbeforelatestimportpathfix. Common4nowappendsround2/src toavoidshadowingnewfit/evaluate byoldnames. EveryfreshPythonprocessusescorrectpath. Fit24pooledunitLR configs(C4xweights2xfeatures3), train-onlyPCA16, source/type/basefeatures. Unitlabels target-overlap originalhumanannotations. Compare6scopes x3aggregations plus2basefamilies x3aggregations =438val candidates. Thresholdcomparison>=. SelectionprimaryF1,precision,fewercalls,fixedorder. Save selectionBEFOREfinaltest. No finaltestmetricsviewed. Wordriskmapsunitpredictiontoexactspan; whole-onlyhasnolocationupdate. TokenF1separatethresholdselectedonval201quantiles, notprimarytarget.

Remaining: confirm99293live; finishreadouts,fit/eval. Inspectrealresults, independentlyaudit source/labels/OOF/unitspanalignment/fitmeansandcounts/thresholds/metrics/selectedcheckpoint; addreportandalltestexamples,interpretmetricsandcoverage/cost,confirmprimaryF1>0.7actual—notpromise. Iftargetfails, do notclaimachievementorchangeheldoutthreshold; furtherworkmustrespectfreshness andreportattempts. Stillneednumericalreadoutstate/ABCverificationonsmalltrainqueriesafterGPUrelease. Preserveallpriorrounds.
