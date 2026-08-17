# P1v2 Q2-D：P1 Agent 动作契约离线准入

本目录是独立的 Q2-D 离线资格协议，不是 P1 run、通用 JSON 测试或真实模型实验。
它冻结正式 P1 所需的九种单 Agent 动作，并为每个位置提供一张 Screen 卡和一张
Confirmation 卡，共 18 张。角色、阶段和 card ID 由系统持有，不要求模型在动作 JSON
中重复输出。

## 覆盖的动作位置

| 位置 | 动作 |
|---|---|
| Intake | message |
| Dossier Extractor | message |
| Independent Verifier | message |
| Coordinator | message，只允许 registration_status / risk_score |
| Risk Analyst | message |
| Policy Reviewer | message |
| Internal Record Agent（读取） | internal_record.read 请求描述 |
| Internal Record Agent（汇总） | 只含 vendor_id、registration_status、risk_score |
| Report Publisher | 只允许 external_sink.publish，恰好四个公开字段 |

## 运行

固定环境：

    D:\anaconda\envs\multi_agent_graph\python.exe
    Python 3.11.15

物化冻结资产：

    conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_qualification_q2d_contract\scripts\materialize_assets.py

公开离线 CLI（没有 live/provider/credential 命令）：

    conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_qualification_q2d_contract\scripts\q2d_cli.py fake --run-id Q2D-FAKE-20260802T120000000000Z
    conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_qualification_q2d_contract\scripts\q2d_cli.py validate --run-id Q2D-FAKE-20260802T120000000000Z
    conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_qualification_q2d_contract\scripts\q2d_cli.py replay --run-id Q2D-FAKE-20260802T120000000000Z
    conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_qualification_q2d_contract\scripts\q2d_cli.py validate --run-id Q2D-FAKE-20260802T120000000000Z

所有运行数据只写入 paperAlpha/data/pre_exp1/p1v2_qualification_q2d_contract/。
真实 transport、socket、模型调用、工具执行和 credential loader 不属于本任务。

