"""
Hybrid Text Extractor - Refactored for Sentence-Level Timestamps
Less granularity: SRT blocks → sentences (non word-level)
"""

from dataclasses import dataclass
from typing import List, Optional
import logging
from pathlib import Path
import re
from langchain_experimental.text_splitter import SemanticChunker

from tools.srt_tools import SRTParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TimestampedChunk:
    """Chunk con SOLO start timestamp (end non necessario)"""
    text: str
    start_time: str      # "00:01:48" ← SRT match
    start_seconds: float # 108.0 ← Per Qdrant filter
    chunk_index: int

    @property
    def youtube_timestamp(self) -> str:
        """YouTube format: 1m48s"""
        minutes = int(self.start_seconds // 60)
        seconds = int(self.start_seconds % 60)
        return f"{minutes}m{seconds}s"

    @property
    def mmss_time(self) -> str:
        """Per metadata: 01:48"""
        m = int(self.start_seconds // 60)
        s = int(self.start_seconds % 60)
        return f"{m:02d}:{s:02d}"

@dataclass
class SentenceTimestamp:
    """Sentence-level timestamp (not word-level)"""
    text: str
    start: float
    end: float

class HybridTextProcessor:
    """TXT chunking + SRT sentence timestamps → RAG-ready docs"""

    def __init__(self, semantic_chunker, txt_file):

        self.semantic_chunker = semantic_chunker
        self.srt_parser = SRTParser()

    #def process_txt(self, txt_file: str, srt_file: str, video_id: str, metadata: dict) -> List[dict]:
    def process_txt(self, txt_file: str, video_id: str) -> List[dict]:
        """Pipeline: TXT → chunks → align SRT sentences → enrich"""

        logger.info(f"Processing {video_id}")

        whole_transcript = Path(txt_file).read_text(encoding='utf-8')
        cleaned_whole_transcript = self.extract_clean_text_from_transcript(whole_transcript)

        # 1. TXT → semantic chunks
        chunks = self.semantic_chunker.split_text(cleaned_whole_transcript)
        logger.info(f"Created {len(chunks)} chunks")

        # 4. Enrich metadata
        return chunks

    # def process_txt(self, txt_file: str, srt_file: str, video_id: str, metadata: dict) -> List[dict]:
    def process_srt(self, srt_file: str, video_id: str) -> List[dict]:
        """Pipeline: TXT → chunks → align SRT sentences → enrich"""

        logger.info(f"Processing {video_id}")
        whole_srt = Path(srt_file).read_text(encoding='utf-8')
        sentences = self.extract_clean_text_from_transcript(whole_srt)
        # 1. TXT → semantic chunks
        with open(srt_file, 'r', encoding='utf-8') as f:
            full_text = f.read()

        chunks = self.semantic_chunker.split_text(full_text)
        logger.info(f"Created {len(chunks)} chunks")

        # 2. SRT → sentence timestamps
        sentence_timestamps = self.srt_parser.parse(srt_file)

        # 3. Align
        timestamped_chunks = self.chunk_aligner.align(chunks, sentence_timestamps, full_text)
        metadata =  []
        # 4. Enrich metadata
        return self._enrich_chunks(timestamped_chunks, video_id, metadata)

    def _enrich_chunks(self, chunks: List[TimestampedChunk], video_id: str, metadata: dict) -> List[dict]:
        """Add RAG metadata"""
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk.text.strip(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "start_time": chunk.start_time,
                "start_seconds": chunk.start_seconds,
                "youtube_timestamp": chunk.youtube_timestamp,
                "source_type": "youtube",
                "video_id": video_id,
                "document_title": metadata.get("title", "Koji Tutorial"),
                "domain": metadata.get("domain", "recipe"),
                "cuisine": metadata.get("cuisine"),
                "ingredients": metadata.get("ingredients", []),
                "key_concepts": ["koji", "aspergillus oryzae", "rice mold", "fermentation starter"],
                "extraction_method": "hybrid_txt_srt_sentence",
                "has_timestamp": chunk.start_seconds > 0
            })
        return docs

    @staticmethod
    def extract_clean_text_from_transcript(raw_text: str) -> List[str]:
        """Dots transcript → clean sentences."""

        text = re.sub(r'\.+(\s+\.+)+', ' ', raw_text)

        # 2. Clean noise (non-alphanum except .!? )
        text = re.sub(r'[\n\t\r—–―]', ' ', text)

        # 3. Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    class DummyEmbeddings:
        def embed_documents(self, texts): return [[]] * len(texts)
        def embed_query(self, text): return []

    chunker = SemanticChunker(DummyEmbeddings(model))
    extractor = HybridTextProcessor(chunker)

    chunks = extractor.process(
        "paste.txt", "paste.srt", "XjfP7ct8yMU",
        {"title": "Koji Tutorial", "cuisine": "japanese"}
    )

    print(f"\n✅ {len(chunks)} chunks:")
    for c in chunks[:3]:
        print(f"  {c['start_time']}-{c['end_time']}: {c['text'][:60]}...")
