# tradingagent_hk

基于多智能体（Multi-Agent）架构的**港股 IPO 基石轮投资分析系统**。

灵感来源：[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)，
针对港股 IPO 基石投资场景重新设计 Agent 角色、数据源、决策输出与 token 优化策略。

---

## 一、它能做什么

输入一个拟港股 IPO 公司的基本信息和招股书 PDF，自动产出一份**投决会级别的研究备忘录**，
包括：

- **行业研究**（市场规模、竞争格局、产业链地位）
- **宏观策略**（港股流动性、IPO 窗口、恒指估值/情绪）
- **招股书深度分析**（业务模式、财务质量、风险因素、募资用途、基石条款）
- **可比公司估值**（PE / PS / EV-EBITDA 多维锚定）
- **技术发展趋势**（科技类公司必备）
- **二级市场情绪**（板块动能、暗盘表现、可比公司股价）
- **Bull/Bear 辩论**（多空对辩，自动收敛）
- **风控独立评估**（8 维风险评级 + 否决条件）
- **最终基石认购决议**（认购/观望/不认购 + 估值区间 + 决议条件）

每个 Agent 的完整研究报告独立落盘，最终汇总成 `FINAL_MEMO.md`。

---

## 二、架构

```
                         ┌──────────────────────┐
                         │   招股书 PDF (THS    │
                         │   API / 手动放入)    │
                         └─────────┬────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
                  │   PDF 解析 → 切块 → ChromaDB RAG    │
                  └─────────┬──────────────────────────┘
                            │ search_prospectus(query)
                            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  分析 Agent 群（Sonnet）                                        │
   │  招股书 / 行业 / 宏观 / 可比估值 / 技术 / 情绪                   │
   │      每个 Agent 输出: full_report.md  +  brief (≤500字)         │
   │      下游只读 brief，避免长上下文叠加                           │
   └─────────────────┬───────────────────────────────────────────────┘
                     ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Bull / Bear 辩论 (Sonnet) + Manager 裁判 (Opus)                │
   │      自适应轮数：相似度 > 阈值 即收敛                            │
   └─────────────────┬───────────────────────────────────────────────┘
                     ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  风控委员会 (Sonnet)                                             │
   └─────────────────┬───────────────────────────────────────────────┘
                     ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  最终投决官 (Opus)  →  结构化 JSON + 论述                       │
   └─────────────────────────────────────────────────────────────────┘
```

### Token 优化策略（设计要点）

| 优化项 | 实现位置 | 说明 |
|---|---|---|
| 多 provider 切换 | `src/llm/client.py` | 支持 Anthropic / Kimi / DeepSeek，按需选成本最低的 |
| 分级模型路由 | `src/llm/router.py` | 各 provider 都有 SUMMARIZE / ANALYZE / DECIDE 三档默认模型 |
| Prompt Caching | `src/llm/client.py` | Anthropic 走 cache_control；Kimi/DeepSeek 自动按前缀命中 |
| 中间层压缩 | `src/agents/summarizer.py` | 每个 Agent 输出后用 SUMMARIZE 档压成 ≤500 字 brief |
| RAG 按需检索 | `src/data/rag.py` | 招股书不全文喂入，按主题检索；BGE-zh embedding |
| 确定性工具 | `src/tools/*` | 估值/财务指标 Python 算好再喂，不让 LLM 推数 |
| 辩论早停 | `src/agents/debate.py` | Bull/Bear brief Jaccard 相似度 > 阈值即停 |
| 数据缓存 | `src/data/cache.py` | SQLite 持久化 akshare / THS 调用结果 |

---

## 三、目录结构

```
tradingagent_hk/
├── cli.py                    # CLI 入口
├── pyproject.toml
├── .env.example
├── config/
│   └── settings.py           # 配置中心（模型 tier、路径、API key）
├── src/
│   ├── llm/
│   │   ├── client.py         # Anthropic 客户端 + caching + token 账本
│   │   └── router.py         # tier → model 路由
│   ├── data/
│   │   ├── cache.py          # SQLite 函数级缓存
│   │   ├── akshare_client.py # 港股行情/财务（akshare）
│   │   ├── ths_client.py     # 同花顺 iFinD（骨架，待 SDK 接入）
│   │   ├── hkex_client.py    # 港交所披露易（骨架）
│   │   ├── prospectus.py     # PDF 解析与切块
│   │   └── rag.py            # ChromaDB RAG（每项目独立 collection）
│   ├── tools/
│   │   ├── financials.py     # 财务指标计算
│   │   └── valuation.py      # 可比估值锚定
│   ├── agents/
│   │   ├── base.py           # Agent 基类 + 上下文 + 报告
│   │   ├── _template.py      # 通用模板（子类只写 prompt）
│   │   ├── summarizer.py     # 中间层压缩 (Haiku)
│   │   ├── prospectus_analyst.py   ★ 完整模板示例
│   │   ├── industry.py
│   │   ├── macro.py
│   │   ├── comparable.py
│   │   ├── tech_trend.py
│   │   ├── sentiment.py
│   │   ├── bull.py
│   │   ├── bear.py
│   │   ├── debate.py         # 辩论协调器 + Manager 裁判
│   │   ├── risk.py
│   │   └── decision.py       ★ 最终投决（输出结构化 JSON）
│   ├── graph/
│   │   └── workflow.py       # 主工作流编排
│   └── reports/
│       └── writer.py         # 投决备忘录生成
└── reports/                  # 运行时产物（自动创建）
    └── <ticker>_<timestamp>/
        ├── 01_prospectus_analyst.md
        ├── 02_industry.md
        ├── ...
        ├── 09_decision.md
        ├── _token_usage.md
        └── FINAL_MEMO.md
```

---

## 四、安装与配置

```bash
# 1. 创建虚拟环境（推荐 Python 3.10+）
python -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -e .
# 或 pip install -e ".[dev]"

# 3. 配置 .env
cp .env.example .env
# 至少填: LLM_PROVIDER + 对应的 API Key
```

### LLM Provider 选择

支持三家，按需切换：

| Provider | 适用场景 | 默认 tier 模型 (SUMMARIZE / ANALYZE / DECIDE) |
|---|---|---|
| `anthropic` | 质量最高，token 最贵 | claude-haiku-4-5 / claude-sonnet-4-6 / claude-opus-4-7 |
| `kimi` | 中文+长上下文友好，性价比高，国内合规 | moonshot-v1-8k / moonshot-v1-32k / kimi-latest |
| `deepseek` | 极低成本，自动 prompt caching | deepseek-chat / deepseek-chat / deepseek-reasoner |

**用 Kimi（推荐起步）：**
```env
LLM_PROVIDER=kimi
KIMI_API_KEY=sk-...
# 可选：覆盖默认模型，例如全部用长上下文模型
# MODEL_TIER_ANALYZE_KIMI=moonshot-v1-128k
```

**用 DeepSeek：**
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
# DeepSeek 自动 prompt caching，>1k token 输入会自动命中（计费按 cache hit 价）
```

> Kimi/DeepSeek 都走 OpenAI 兼容协议，本项目用 `openai` SDK 调用。
> 对应的 prompt caching 机制：Anthropic 显式 `cache_control` / Kimi 自动前缀缓存 / DeepSeek 自动前缀缓存。

### 同花顺 iFinD QuantAPI 接入

直接用 HTTP REST API，不需要装 SDK。在 `.env` 填入：

```
THS_REFRESH_TOKEN=<你的 refresh_token>
```

**接入的接口：**
- `get_access_token` / `update_access_token` — token 自动管理（带本地缓存 + 401 自动刷新）
- `report_query` — 公告查询，支持按代码 / 类型 / 日期 / 关键词筛选
- 招股书 PDF 自动下载到 `data/prospectus/<ticker>.pdf`
- `basic_data_service` (THS_BD) — 公司基础信息（中英文名/主营/上市日期/实控人/股东等），喂入招股书 Agent
- `edb_service` (THS_EDB) — 宏观/流动性指标时间序列（HIBOR、HSI PE、USD/HKD、CPI、PMI、IPO 集资额），喂入宏观 Agent
- `data_report` (THS_DR) — 行业研报列表，喂入行业 Agent

> 端点路径可通过 `.env` 的 `THS_ENDPOINT_*` 覆盖；EDB 指标编码集合在 `src/data/ths_client.py` 顶部 `DEFAULT_HK_MACRO_EDB_CODES` 可调整。

**自动获取招股书的流程：**

```
执行 analyze --ticker 09999 时：
  1) 检查 data/prospectus/09999.pdf 是否存在 → 有则直接用
  2) 调用 THS report_query 拉公告流（默认 2023-01-01 至今）
  3) 用标题关键词过滤（"聆讯后资料集" / "招股章程" / "Prospectus" / "PHIP" 等）
  4) 按 prefer 偏好排序（默认 PHIP 优先）
  5) 下载 pdfURL 到本地，落入 RAG 索引
失败时自动 fallback 到无 RAG 模式（依然能跑大部分 Agent）。
```

辅助命令（用于排查匹配问题）：

```bash
# 仅拉招股书
python cli.py fetch-prospectus --ticker 09999

# 列出某 ticker 的所有公告（看标题，调整关键词）
python cli.py list-announcements --ticker 09999 --start 2024-01-01

# 关键词过滤（直接传给同花顺 functionpara）
python cli.py list-announcements --ticker 09999 --keyword 招股
```

**未配置 THS 时**：系统自动跳过 THS，回退到本地 PDF 或 akshare 公开数据。

---

## 五、使用

```bash
# 推荐用法：配置了 THS_REFRESH_TOKEN 后，招股书会自动从同花顺拉取
python cli.py analyze \
    --ticker 09999 \
    --name "示例科技集团" \
    --industry "AI/SaaS"

# 手动指定招股书 PDF
python cli.py analyze --ticker 09999 --name "示例" --industry "TMT" \
    --pdf data/prospectus/09999.pdf

# 禁用自动拉取（仅用本地缓存）
python cli.py analyze --ticker 09999 --name "示例" --industry "TMT" --no-auto-fetch

# 跳过招股书快速测试
python cli.py analyze --ticker 09999 --name "示例" --industry "TMT" --no-prospectus

# 调试用：仅拉招股书 / 列公告
python cli.py fetch-prospectus --ticker 09999
python cli.py list-announcements --ticker 09999 --start 2024-01-01

# 查看当前配置
python cli.py show-config
```

完成后查看：
- `reports/<project_id>/FINAL_MEMO.md` — 投决备忘录（给人看）
- `reports/<project_id>/01_*.md` ~ `09_*.md` — 各 Agent 完整研究报告
- `reports/<project_id>/*.brief.md` — 各 Agent 简报（流转用）
- `reports/<project_id>/_token_usage.md` — token 消耗账本

### 案例：珞石机器人（Loctek Robotics）测试流程

```bash
# 1. 配置 Kimi 和 同花顺
cat >> .env <<EOF
LLM_PROVIDER=kimi
KIMI_API_KEY=sk-你的key
THS_REFRESH_TOKEN=你的refresh_token
EOF

# 2. 执行（招股书会通过 THS 自动拉）
python cli.py analyze \
    --ticker <珞石港股代码> \
    --name "珞石（北京）科技有限公司" \
    --industry "工业机器人/协作机器人"

# 若 THS 暂未收录可手动放 PDF：
# 把招股书丢到 data/prospectus/<ticker>.pdf 即可

# 3. 看 token 消耗
cat reports/<project_id>/_token_usage.md
```

**预估 token 消耗（Kimi 全家桶）：**
- 单次完整跑通约 30-50 万 token（含 RAG 检索 + 8 Agent + 2 轮辩论 + 风控 + 决策）
- Kimi 当前定价约 ¥12/百万 token（输入），单次成本约 ¥4-7
- 同条件用 Anthropic Opus/Sonnet/Haiku 混合约 $3-8（约 ¥20-60）
- 同条件用 DeepSeek 约 ¥0.5-2（cache hit 后更低）

---

## 六、扩展 / 二次开发

### 新增 Agent

1. 在 `src/agents/` 下新建文件，继承 `TemplateAgent`；
2. 声明 `name` / `description` / `tier` / `SYSTEM`；
3. 实现 `build_user_message(ctx)`；
4. 在 `src/graph/workflow.py` 的 `self.steps` 加入实例。

### 修改决策结构

`src/agents/decision.py` 的 `DECISION_SYSTEM` 中定义了输出 JSON Schema，
按需调整字段，对应修改 `parse_decision_json` 与 `src/reports/writer.py`。

### 接入新数据源

在 `src/data/` 加新 client 模块，使用 `disk_cache` 装饰器即可获得磁盘缓存。

---

## 七、当前阶段的 TODO

- [x] 同花顺 QuantAPI 接入：token 管理、`report_query`、招股书 PDF 自动下载
- [x] 同花顺 `THS_BD` / `THS_DR` / `THS_EDB` 接入：公司基础信息、行业研报、宏观指标
- [x] 中文 embedding 切到 BGE-zh-v1.5（可通过 `EMBEDDING_MODEL` 调整为 small/large）
- [ ] 港交所披露易爬虫实现
- [ ] akshare 港股财务接口字段在不同版本的兼容（当前已做容错）
- [ ] LangGraph 替换线性 workflow（如需图状条件分支）
- [ ] 单元测试 + 历史 IPO 回测脚本（按已上市公司回看基石认购效果）

---

## 八、设计取舍

- **没有用 LangGraph**：当前流程是确定性线性的，用 LangGraph 反而增加学习成本。后续如需条件分支再切换。
- **没有用 LangChain**：直接调 Anthropic SDK，控制粒度更细，prompt caching 用得更准。
- **每个 Agent 一个 markdown 文件**：方便人工审查思考过程，符合"完整展示研究分析过程"的需求。
- **brief 强制 JSON-like 结构**：让下游 Agent 解析更稳定，token 上限可控。

---

## License

内部使用，未公开授权。
