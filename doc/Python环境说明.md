# 项目统一 Python 环境

> 状态：已创建并验证  
> 环境名称：`multi_agent_graph`  
> Python 版本：`3.11.15`  
> Conda 位置：`D:\anaconda`  
> Python 解释器：`D:\anaconda\envs\multi_agent_graph\python.exe`

> **2026-09-09 用户授权的独立例外：** `prelab` 幻觉监测预实验使用
> `D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe`（Python 3.11.15）。
> 通过 `D:\anaconda\envs\CA\python.exe -m venv --system-site-packages prelab/.venv` 创建，
> 复用 CA 的 PyTorch 2.5.1+cu121，专用依赖安装在 prelab 虚拟环境中。
> 用户已明确允许新建或改造环境；以下统一环境规则对其他项目继续适用。
> 具体依赖和复现步骤见 [prelab](../prelab/README.md)。

## 1. 强制规则

本项目所有 Python 代码、测试、数据处理和实验运行统一使用：

```text
multi_agent_graph
```

任何 Agent 都不得自行：

- 使用 Conda `base`；
- 随机选择现有其他环境；
- 创建新的 Conda 环境；
- 在项目内创建 `.venv` 或 `venv`；
- 使用系统 Python；
- 在未说明的环境中安装依赖或运行测试。

如确需改变 Python 主版本或更换环境，必须先由用户或主 Agent 明确批准，并同步更新本文件、任务书和所有受影响的运行说明。

## 2. 推荐运行方式

自动化 Agent 和非交互 shell 优先使用：

```powershell
conda run -n multi_agent_graph python --version
conda run -n multi_agent_graph python path\to\script.py
conda run -n multi_agent_graph python -m pytest
conda run -n multi_agent_graph python -m pip list
```

这种方式不依赖 shell 是否已正确初始化 Conda。

人工交互时可以使用：

```powershell
conda activate multi_agent_graph
python --version
```

如果 `conda run` 暂时不可用，可以直接使用固定解释器：

```powershell
& 'D:\anaconda\envs\multi_agent_graph\python.exe' --version
```

## 3. 环境验证

Agent 开始实现前必须运行：

```powershell
conda run -n multi_agent_graph python -c "import sys, platform; print(sys.executable); print(platform.python_version())"
```

预期：

```text
D:\anaconda\envs\multi_agent_graph\python.exe
3.11.15
```

最终交付时应再次报告：

- 实际解释器路径；
- Python 版本；
- 测试使用的环境；
- 是否安装了新依赖。

## 4. 依赖安装规则

当前环境初始只包含：

- Python 3.11；
- pip；
- Python/Conda 的基础运行包。

实施 Agent 应：

1. 优先使用标准库；
2. 只安装当前任务确实需要的依赖；
3. 使用以下形式安装：

   ```powershell
   conda run -n multi_agent_graph python -m pip install <package>
   ```

4. 安装前说明新增依赖的用途；
5. 安装后把依赖写入项目的 `pyproject.toml` 或 `environment.yml`；
6. 最终回复中列出新增包和版本；
7. 不安装当前阶段不需要的大型框架；
8. 不静默升级 Python、pip 或已有依赖。

对于可能改变大量包版本、引入 GPU 运行时或影响其他实验的依赖，实施 Agent 应先停止并请求主 Agent确认。

## 5. 项目环境文件

实施 Agent 创建 `paperAlpha` 时，应同时创建：

```text
paperAlpha/environment.yml
```

至少记录：

```yaml
name: multi_agent_graph
channels:
  - defaults
dependencies:
  - python=3.11
  - pip
```

后续增加依赖时同步更新，但不把与项目无关的本机包全部导出进去。

## 6. Prompt 必备环境块

主 Agent 给任何 Python 实施 Agent 的 Prompt 都必须包含：

```text
统一 Python 环境：
- Conda 环境名：multi_agent_graph
- Python：3.11.15
- 解释器：D:\anaconda\envs\multi_agent_graph\python.exe

所有 Python 命令、测试和依赖安装必须在该环境中执行。
自动化命令优先使用：
conda run -n multi_agent_graph python ...

禁止使用 base、系统 Python、其他 Conda 环境或自行创建 venv。
开始前先打印 sys.executable 和 Python 版本；交付时再次报告实际使用的环境。
```

没有这段环境信息的实施 Prompt 视为不完整，不应发送。

## 7. 本机模型执行资源

> 盘点日期：2026-07-31  
> 用途：决定预实验使用本地模型还是远程推理，不作为跨机器性能承诺。

### 7.1 硬件

```text
CPU：Intel Core i5-14600KF，14 核 / 20 线程
内存：约 64 GB
GPU：NVIDIA GeForce RTX 3070
显存：8 GB
```

当前硬件适合：

- 4B–8B 量化模型的单请求、低并发实验；
- 8-Agent 工作流按角色顺序串行调用；
- 小规模工程 shakedown 和 P1 良性任务 Pilot；
- 使用系统内存部分卸载更大模型，但吞吐与时延可能明显下降。

当前硬件不适合默认承担：

- 多个 8B/12B 模型并行常驻；
- 高并发多 episode 批量推理；
- 需要大显存的全精度或超大上下文模型；
- 未经基准测试就直接运行大规模确认性实验。

### 7.2 已安装 Ollama 模型

```text
ministral-3:8b
qwen3-vl:4b
gemma3:12b
deepseek-r1:8b
qwen3-vl:8b
qwen3:8b
gemma3:4b
nomic-embed-text:latest
qwen3:latest
deepseek-r1:latest
llama3.2:latest
```

P1 的默认本地工程 shakedown 选择：

```text
qwen3:8b
```

理由：

- 文本任务不需要视觉模型；
- 约 5.2 GB 的本地模型文件与 8 GB 显存更匹配；
- 相比 12B 模型更适合八角色串行调用；
- 不优先使用带显式推理输出的模型，降低隐藏 reasoning 进入日志的风险。

这只是工程 shakedown 默认值，不自动成为确认性主实验模型。正式 P1 批次前仍需冻结
模型版本、Ollama 版本、解码参数、并发、超时和调用预算。

本地执行规则：

- 并发固定为 1；
- 不在实施任务中自行下载新模型；
- 先做 1 个 episode 的全链时延和结构化输出检查；
- 本地 shakedown 标记为不可用于科学分析；
- 只有主 Agent 验收并明确冻结模型后，才能运行正式 P1 批次。

## 8. 远程模型资源与密钥规则

### 8.1 讯飞星辰 Astron Coding Plan

用户提供的非秘密配置：

```text
Provider：xfyun_astron_coding_plan
OpenAI-compatible Base URL：https://maas-coding-api.cn-huabei-1.xf-yun.com/v2
```

当前登记的模型 ID：

```text
xsparkx2agent
xopglm5
xopglm52
xopdeepseekv4pro
xopdeepseekv4flash
xopkimik26
xopglm51
xopqwen36v35b
xopqwen35397b
```

官方接入文档：

- [讯飞星辰 MaaS · Astron Coding Plan 使用文档](https://www.xfyun.cn/doc/spark/CodingPlan.html)

截至 2026-07-31，官方文档将 Coding Plan 额度限定为交互式编程工具场景，并明确禁止
自动化脚本、批量任务和非编程工具调用。因此：

- 该凭据可以登记为开发辅助资源；
- 不得用于 P1 的自动化 20-episode 批量实验；
- 不得通过伪装成交互式编程请求绕过用途限制；
- 若要用于正式实验，需要用户提供允许自动化推理的常规 MaaS 服务凭据，或取得服务
  方对本研究用途的明确授权。

### 8.2 密钥存储

本机密钥只保存在：

```text
paperAlpha/.env
```

对应变量名：

```text
XFYUN_ASTRON_CODING_BASE_URL
XFYUN_ASTRON_CODING_API_KEY
XFYUN_ASTRON_CODING_MODELS
XFYUN_ASTRON_CODING_USAGE
```

强制规则：

- `.env` 必须被 `.gitignore` 排除；
- `.env` 不进入源码哈希、输入哈希、manifest、日志、报告或 Agent Prompt；
- 代码只记录密钥变量名和“是否存在”，不得记录密钥值或其可逆形式；
- 测试使用虚构 key，不读取真实 `.env`；
- 错误消息不得包含 Authorization header；
- 任何交付回复不得回显密钥；
- 新增远程 provider 时，先核对服务条款是否允许实验用途。

### 8.3 用户指定的 OpenAI-compatible gateway（P1v2-Q2 候选）

用户已明确指定下列本机 gateway 作为后续 Q2 候选配置的提供方：

```text
Base URL：http://127.0.0.1:58661/v1
模型 ID：gpt-5.3-codex-spark
接口形式：OpenAI-compatible
```

2026-08-02 已使用用户提供的凭据成功只读查询 `GET /v1/models`，并确认该模型 ID 出现在列表中。
这只证明 gateway 当前可枚举该模型；不证明模型版本、JSON schema 支持或资格通过。

本候选替代**未来 Q2 资格调用**中的 Ollama，而不改写已经完成的 Q1、本地模型记录或任何历史工件。
Q2-A 已通过独立验收（`ready_for_q2b`），但它只使用虚构凭据和假 transport。用户随后明确授权
Q2-B 在单一冻结协议下逐步执行一次真实调用：2 次 `GET /v1/models`，先 8 张 Screen，门槛通过后才最多
扩展至 32 张 Screen 和 96 张 Confirmation；无重试、无模型切换、无其他 provider。该授权已经消耗，
不延伸到正式 P1、P2、修复后的重跑或任何未另行冻结的新批次。

唯一 Q2-B run `P1V2Q2B-REMOTE-20260802T033140463464Z` 停在首批 8 条 Screen（均被记录为
`tool_calls` 契约失败），原始运行结果为 `not_qualified`。不过独立验收发现总时限和失败记录证据链的
实现缺口，故 Q2-B **整体交付为 `rework`**；它不构成可进入 P1 的已验收资格结论。该 gateway 与模型
仍只能作为未来经新任务书、新离线验收和用户明确授权后才能调用的候选配置。

随后，唯一最终离线修复派生审计 `P1V2Q2B-REPAIR-AUDIT-20260802T044126194259Z` 已通过独立验收：
它证明修复后的**离线** deadline、失败身份、冻结 metadata identity 与本地 DELIVERY 证据链可被验证和
回放。这一结果不重跑、不补写也不升级原始 Q2-B；原始批次仍是 `rework`，P1/P2 仍锁定，也没有产生新的
真实 API 调用授权。

2026-08-02，用户授权的独立 Q2-C Codex 响应行为小型探针已在新代码和数据根中完成唯一真实 run
`P1V2Q2C-PROBE-20260802T061803504801Z`：1 次 `GET /v1/models` 与 4 次串行 completion，零重试、零模型
切换、零其他 provider；四种固定良性格式均记录为 `content_match`，run 结论为 `observed`。其公开
`validation → replay → validation` 与最终独立验收均通过，安全工件位于
`paperAlpha/data/pre_exp1/p1v2_qualification_q2c_probe/runs/P1V2Q2C-PROBE-20260802T061803504801Z/`。

该结果只是一项受限的 gateway 响应兼容性观察：不追溯性修复 Q2-B，不构成模型资格、正式 P1/P2、批量实验
或论文结论。Q2-C 的唯一真实调用授权已经消耗，不得执行第二个 Q2-C run。

该 gateway 的凭据只能保存在 `paperAlpha/.env`，使用以下变量名：

```text
P1V2_Q2_BASE_URL
P1V2_Q2_API_KEY
P1V2_Q2_MODEL
```

变量值不得进入任务书、Prompt、源码哈希、manifest、日志、报告、测试 fixture 或交付回复。实现不得
把它们作为命令行参数，也不得在失败信息中输出 Authorization header。没有明确的用户授权与独立
任务书，不得把该 gateway 用于超出 Q2-B 的自动化真实资格批次或正式实验。
