#!/usr/bin/env python3
"""
Video Intelligence Agent con Keyword Focus
- Manual/Auto keywords extraction
- ±1min intervals intorno keywords trovati
- Full + focused summary
- OpenAI GPT-4o-mini powered
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tools.srt_tools import *
from pathlib import Path
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

# Config
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)


class VideoState(TypedDict):
    video_url: str
    instructions: str
    keywords: Optional[List[str]]  # Manuali O auto-estratte
    webm_path: str
    srt_path: str
    txt_path: str
    txt_content: str
    chunks: List[str]
    keyword_intervals: Optional[List[str]]
    interval_summary: Optional[str]
    full_summary: str
    keyword_summary: str
    actions: List[str]
    sentiment: str


def chunk_transcript(text: str, chunk_size: int = 2000) -> List[str]:
    """Divide transcript in chunk gestibili"""
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(' '.join(current)) > chunk_size:
            chunks.append(' '.join(current))
            current = []
    if current:
        chunks.append(' '.join(current))
    return chunks


def process_video_node(state: VideoState) -> dict:
    """1. Download + transcript + keywords extraction"""
    print(f"🎥 Processing: {state['video_url']}")

    # Download con TUO microservice
    webm_path, srt_path, txt_path = download_video_to_txt(state["video_url"])
    txt_content = Path(txt_path).read_text(encoding='utf-8')
    chunks = chunk_transcript(txt_content)

    # 🔑 KEYWORDS: Manuali PRIORITÀ → Auto-estrazione
    keywords = state.get("keywords")

    if keywords is None or len(keywords) == 0:
        # AUTO-ESTRAZIONE intelligente
        print("🔑 Auto-extracting keywords...")
        keywords_prompt = f"""
        Da queste istruzioni estrai 2-4 parole chiave IMPORTANTI:
        "{state['instructions']}"

        Priorità: nomi propri, concetti chiave, numeri, termini tecnici.
        JSON array: ["budget", "Q1", "meeting"]
        """
        try:
            keywords_resp = llm.invoke(keywords_prompt)
            keywords = json.loads(keywords_resp.content)
            print(f"✅ AUTO keywords: {keywords}")
        except Exception as e:
            print(f"⚠️ Auto-extraction failed: {e}")
            # Fallback regex
            keywords = re.findall(r'\b[A-Z][a-z]{3,}\b', state["instructions"])
            keywords = keywords[:4]

    print(f"🔑 Final keywords: {keywords}")

    return {
        "webm_path": str(webm_path),
        "srt_path": str(srt_path),
        "txt_path": str(txt_path),
        "txt_content": txt_content,
        "chunks": chunks,
        "keywords": keywords
    }


def keyword_interval_node(state: VideoState) -> dict:
    """2. Trova keywords → ±1min testo intorno"""
    print(f"🔍 Keyword search: {state['keywords']}")

    txt_lower = state['txt_content'].lower()
    all_words = txt_lower.split()
    words_per_minute = 150  # Velocità media discorso
    intervals = []

    for keyword in state['keywords']:
        kw_lower = keyword.lower()
        found = False

        for i, word in enumerate(all_words):
            if kw_lower in word:
                # ±1 minuto (300 parole totali)
                start = max(0, i - words_per_minute)
                end = min(len(all_words), i + words_per_minute + 1)
                interval_words = all_words[start:end]
                interval_text = ' '.join(interval_words)

                intervals.append({
                    "keyword": keyword,
                    "position": i,
                    "text": interval_text[:1200]  # LLM limit
                })
                print(f"✅ '{keyword}' trovato alla parola {i} → intervallo {start}-{end}")
                found = True
                break

        if not found:
            print(f"❌ '{keyword}' non trovato")

    # Summary intervalli
    if intervals:
        intervals_text = '\n\n'.join([f"[{i['keyword']}] {i['text'][:400]}..." for i in intervals])
        interval_summary = llm.invoke(
            f"SUMMARIZE SOLO questi intervalli keywords (mantieni contesto):\n\n{intervals_text}"
        ).content
    else:
        interval_summary = "Nessun keyword trovato nel transcript"

    return {
        "keyword_intervals": intervals,
        "interval_summary": interval_summary
    }


def summarize_node(state: VideoState) -> dict:
    """3. Summary completo + keyword focus (safe per entrambi rami)"""
    print("📝 Generating summaries...")

    # Base chunk summaries
    chunk_summaries = []
    for i, chunk in enumerate(state["chunks"][:6]):
        chunk_sum = llm.invoke(f"Summarize chunk {i + 1}:\n{chunk[:1500]}").content
        chunk_summaries.append(chunk_sum)

    # Safe keyword context
    keyword_context = (
        state.get("interval_summary", "Analisi completa video")
        if state.get("keyword_intervals")
        else "Nessun focus specifico"
    )

    final_prompt = f"""
    ISTRUZIONI: {state["instructions"]}

    CHUNK SUMMARIES: {' | '.join(chunk_summaries)}

    KEYWORD FOCUS: {keyword_context}

    Rispondi strutturato:

    ## FULL SUMMARY (3 frasi chiave)

    ## KEYWORD SUMMARY (se applicabile)

    ## ACTION ITEMS
    • item 1
    • item 2

    ## SENTIMENT
    positivo/negativo/neutro
    """

    final_result = llm.invoke(final_prompt)

    # Simple parsing (migliorabile)
    content = final_result.content

    return {
        "full_summary": content[:1000],
        "actions": re.findall(r'• (.*?)(?=\n•|$)', content, re.DOTALL) or ["No actions"],
        "sentiment": re.search(r'(positivo|negativo|neutro)', content.lower()).group(1) if re.search(
            r'(positivo|negativo|neutro)', content.lower()) else "neutro"
    }


def router(state: VideoState) -> str:
    """Decide flusso basato su keywords"""
    has_keywords = bool(state.get("keywords") and len(state["keywords"]) > 0)
    print(f"🧠 Router: keywords={has_keywords}")
    return "keyword_intervals" if has_keywords else "summarize_node"


# ================================
# BUILD GRAPH
# ================================

workflow = StateGraph(VideoState)

# Nodes
workflow.add_node("process_video", process_video_node)
workflow.add_node("keyword_intervals", keyword_interval_node)
workflow.add_node("summarize_node", summarize_node)

# Edges
workflow.set_entry_point("process_video")
workflow.add_conditional_edges(
    "process_video",
    router,
    {
        "keyword_intervals": "keyword_intervals",
        "summarize_node": "summarize_node"
    }
)
workflow.add_edge("keyword_intervals", "summarize_node")
workflow.add_edge("summarize_node", END)

video_agent = workflow.compile()

# ================================
# USAGE EXAMPLES
# ================================

if __name__ == "__main__":
    print("🚀 Video Intelligence Agent Ready!\n")

    # 1. AUTO-KEYWORDS (da instructions)
    print("1️⃣ AUTO-KEYWORDS:")
    result_auto = video_agent.invoke({
        "video_url": "https://www.youtube.com/shorts/d2dJIA1imS8",
        "instructions": "Analizza miso fermentation koji"
    })
    print("Keywords:", result_auto["keywords"])
    print("Interval:", result_auto.get("interval_summary", "N/A")[:200])
    print("Summary:", result_auto["full_summary"], "\n")

    # 2. MANUAL KEYWORDS
    print("2️⃣ MANUAL KEYWORDS:")
    result_manual = video_agent.invoke({
        "video_url": "https://www.youtube.com/shorts/d2dJIA1imS8",
        "keywords": ["miso", "fermentation"],
        "instructions": "Focus specifico"
    })
    print("Manual keywords:", result_manual["keywords"])
    print("Summary:", result_manual["full_summary"], "\n")

    # 3. NO KEYWORDS (full summary)
    print("3️⃣ NO KEYWORDS:")
    result_full = video_agent.invoke({
        "video_url": "https://www.youtube.com/shorts/d2dJIA1imS8",
        "instructions": "Riassunto completo senza focus"
    })
    print("Full summary:", result_full["full_summary"])
