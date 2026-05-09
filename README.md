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
| 分级模型路由 | `src/llm/router.py` | Haiku 摘要 / Sonnet 分析 / Opus 决策 |
| Prompt Caching | `src/llm/client.py` | `cached_system_blocks` 把招股书前文缓存 |
| 中间层压缩 | `src/agents/summarizer.py` | 每个 Agent 输出后用 Haiku 压成 ≤500 字 brief，下游只读 brief |
| RAG 按需检索 | `src/data/rag.py` | 招股书不全文喂入，按主题检索 |
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
# 编辑 .env 至少填入 ANTHROPIC_API_KEY
```

### 同花顺 iFinD 接入（可选）

iFinD Python SDK 需单独安装（见同花顺官网）。安装后在 `.env` 填入 `THS_USER` 和 `THS_PASSWORD`，
`src/data/ths_client.py` 会自动加载。当前接入点（招股书下载 / 行业研报 / 宏观指标）为骨架，
按 SDK 实际接口编码补充即可。

未配置 THS 时系统自动回退到 akshare 公开数据。

---

## 五、使用

```bash
# 把招股书放到 data/prospectus/<ticker>.pdf 或用 --pdf 指定路径
python cli.py analyze \
    --ticker 09999 \
    --name "示例科技集团" \
    --industry "AI/SaaS" \
    --pdf data/prospectus/09999.pdf

# 跳过招股书快速测试（仅跑非 RAG 依赖的 Agent）
python cli.py analyze --ticker 09999 --name "示例" --industry "TMT" --no-prospectus

# 查看当前配置
python cli.py show-config
```

完成后查看：
- `reports/<project_id>/FINAL_MEMO.md` — 投决备忘录（给人看）
- `reports/<project_id>/01_*.md` ~ `09_*.md` — 各 Agent 完整研究报告
- `reports/<project_id>/*.brief.md` — 各 Agent 简报（流转用）
- `reports/<project_id>/_token_usage.md` — token 消耗账本

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

- [ ] 同花顺 iFinD SDK 接口完整接入（招股书 URL、行业研报、宏观指标）
- [ ] 港交所披露易爬虫实现
- [ ] akshare 港股财务接口字段在不同版本的兼容（当前已做容错）
- [ ] 中文 embedding 模型替换（默认 MiniLM 中文一般，可换 BGE-zh）
- [ ] LangGraph 替换线性 workflow（如需图状条件分支）
- [ ] 单元测试 + 历史 IPO 回测脚本

---

## 八、设计取舍

- **没有用 LangGraph**：当前流程是确定性线性的，用 LangGraph 反而增加学习成本。后续如需条件分支再切换。
- **没有用 LangChain**：直接调 Anthropic SDK，控制粒度更细，prompt caching 用得更准。
- **每个 Agent 一个 markdown 文件**：方便人工审查思考过程，符合"完整展示研究分析过程"的需求。
- **brief 强制 JSON-like 结构**：让下游 Agent 解析更稳定，token 上限可控。

---

## License

内部使用，未公开授权。
