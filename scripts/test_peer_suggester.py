"""验证 PeerSuggester: 复用 02670 已有 RAG 索引, 不跑 9 agents。"""
from __future__ import annotations

import sys

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from src.agents.peer_suggester import suggest_peers
from src.data.rag import ProspectusRAG
from src.llm import LLMClient

PROJECT_ID = "02670_20260509_135040"
COMPANY = "珞石（山东）智能科技股份有限公司"
INDUSTRY = "工业机器人/协作机器人"


def main() -> int:
    rag = ProspectusRAG(project_id=PROJECT_ID)
    if not rag.is_indexed():
        logger.error(f"RAG {PROJECT_ID} 未索引")
        return 1
    logger.info("RAG loaded, calling suggest_peers ...")
    candidates = suggest_peers(rag, COMPANY, INDUSTRY, LLMClient())
    print()
    print(f"=== 候选 peers ({len(candidates)} 个) ===")
    for c in candidates:
        print(f"  {c.ticker} {c.name:12s} | {c.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
