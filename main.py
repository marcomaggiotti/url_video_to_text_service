# main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import base64
from pathlib import Path

from tools.HybridTextProcessor import HybridTextProcessor
from tools.srt_tools import *
import uvicorn
from typing import Optional, List
from fastapi import FastAPI, Query
from workflow import video_agent
import tempfile
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ---------- domain logic ----------

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0"}

@app.get("/")
async def root():
    return {"message": "CAD-RAG API ready!"}

class Settings(BaseModel):
    """Application settings"""
    collection_name: str = "personal_embeddings"
    model_name: str = "all-MiniLM-L6-v2"
    vector_size: int = 384
    qdrant_path: str = "./qdrant_storage"  # FIXED: local disk storage
    host: str = "0.0.0.0"
    port: int = 8000

class SentenceTransformerEmbeddings:
    """Wrapper for SemanticChunker compatibility"""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text]).tolist()[0]

class ChunkMetadata(BaseModel):
    source_type: str
    document_id: str
    document_title: str
    domain: str
    cuisine: str
    meal_type: str
    timestamp: Optional[float] = 0.0
    ingredients: List[str] = []
    key_concepts: List[str] = []

class ChunkResponse(BaseModel):
    text: str
    metadata: ChunkMetadata
    chunk_id: str  # UUID o index

class VectorDBReadyResponse(BaseModel):
    chunks: List[ChunkResponse]
    video_id: str
    total_chunks: int

settings = Settings()
embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
        buffer_size = 2,
        min_chunk_size=150
    )
model_mixedBread = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",
                                       cache_folder='./models_cache')

def retrieve_files(name) -> tuple[Path, Path, Path]:
    """
    Your existing function that creates/returns:
    1) .webm file
    2) .srt file
    3) .txt file
    """
    webm_path = Path(f"output/{name}.webm")
    srt_path = Path(f"output/{name}.srt")
    txt_path = Path(f"output/{name}.txt")
    return webm_path, srt_path, txt_path

def encode_file(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------- response models ----------

class FileJson(BaseModel):
    filename: str
    mimetype: str
    content: Optional[str] = Field(default=None)
    content_b64: Optional[str] = Field(default=None)

class FilesResponse(BaseModel):
    webm: FileJson
    srt: FileJson
    text: FileJson

# ---------- API endpoint ----------

@app.get("/files", response_model=FilesResponse)
async def get_files(name:str):
    webm_path, srt_path, txt_path = retrieve_files(name)

    webm = FileJson(
        filename=webm_path.name,
        mimetype="video/webm",
        content_b64=encode_file(webm_path),
    )
    srt = FileJson(
        filename=srt_path.name,
        mimetype="application/x-subrip",
        content_b64=encode_file(srt_path),
    )
    text = FileJson(
        filename=txt_path.name,
        mimetype="text/plain",
        content_b64=encode_file(txt_path),
    )

    return FilesResponse(webm=webm, srt=srt, text=text)

@app.get("/extract-text-from-video", response_model=FilesResponse)
async def extract_video_to_text(
    video_url: str = Query(..., description="YouTube/Direct video URL")):

    """ Processa URL → webm + srt + txt"""

    webm_path, srt_path, txt_path = download_video_to_txt(video_url)

    webm_file = Path(webm_path)
    srt_file = Path(srt_path)
    txt_file = Path(txt_path)

    # Encoding...
    srt = FileJson(filename=srt_file.name, mimetype="application/x-subrip", content=srt_file.read_text(encoding='utf-8'))
    text = FileJson(filename=txt_file.name, mimetype="text/plain", content=txt_file.read_text(encoding='utf-8'))
    webm = FileJson(filename=webm_file.name, mimetype="video/webm", content_b64=encode_file(webm_file))

    return FilesResponse(webm=webm, srt=srt, text=text)

@app.get("/split-transcript-to-chunks", response_model=FilesResponse)
async def get_split_transcript_to_chunks(file_name: str = Query(..., description="YouTube/Direct video URL")):

    chunks_data = []
    downloads_folder = Path("downloads")
    txt_path  = downloads_folder / file_name

    file_base = txt_path.stem
    name_part, video_id = file_base.rsplit("_", 1)
    name_with_underscores = name_part.replace(" ", "_")

    processor = HybridTextProcessor(
        semantic_chunker=semantic_splitter,
        txt_file=file_base
    )

    if txt_path.name.endswith('.txt'):
        # Chunks
        txt_chunks = processor.process_txt(txt_path, video_id)

        name, ext = os.path.splitext(txt_path)
        srt_file_name = f"{name}.srt"

        srt_parser = SRTParser()
        sentence_timestamps = srt_parser.parse(srt_file_name)

        # This need to change as I will retrieve the chunks from the service
        yt_emb_mixedbread = model_mixedBread.encode(txt_chunks)

        previous_timestamp = - 1.0
        index_sentence_timestamps = 0

        for i, chunk in enumerate(txt_chunks):
            timestamp, index_sentence_timestamps = srt_parser.get_timestamps_of_text(chunk, sentence_timestamps,
                                                                                     previous_timestamp,
                                                                                     index_sentence_timestamps)
            previous_timestamp = timestamp
            logger.info(f"timestamp {timestamp}")


            metadata = ChunkMetadata(
                source_type="YouTube",
                document_id=video_id,
                document_title=name_with_underscores,
                domain="recipe",
                cuisine="japanese",
                meal_type="main_dish",
                timestamp=timestamp,
                ingredients=["rice", "koji mold", "salt"],
                key_concepts=["fermentation", "koji making", "umami", "rice preparation"]
            )

            chunk_data = ChunkResponse(
                text=chunk,
                metadata=metadata,
                chunk_id=f"{video_id}_chunk_{i}"  # Unico per Qdrant
            )

            chunks_data.append(chunk_data)

    return VectorDBReadyResponse(
        chunks=chunks_data,
        video_id=video_id,
        total_chunks=len(chunks_data)
    )



@app.post("/agent/analyze")
async def analyze_video_agent(
        video_url: str = Query(..., description="YouTube/URL video"),
        instructions: Optional[str] = Query("Summary + action items", description="Instructions"),
        keywords: Optional[List[str]] = Query(None, description="Keywords manuali (opzionale)")
):
    """Video Intelligence Agent - Keywords + Intervalli"""

    # Invoke con supporto keywords
    result = video_agent.invoke({
        "video_url": video_url,
        "instructions": instructions,
        "keywords": keywords  # ✅ Manual keywords!
    })

    # ✅ BACKWARDS COMPATIBLE + Enhanced
    return {
        # Original API structure (maintained)
        "summary": result["full_summary"],  # ← Mappato
        "actions": result["actions"],
        "sentiment": result["sentiment"],
        "files": {
            "webm": result["webm_path"],
            "srt": result["srt_path"],
            "txt": result["txt_path"]
        },

        # ✅ NUOVO: Keyword intelligence
        "keywords": result["keywords"],
        "keyword_intervals": result.get("keyword_intervals", []),
        "interval_summary": result.get("interval_summary", ""),

        # Debug
        "debug": {
            "total_chunks": len(result["chunks"]),
            "transcript_length": len(result["txt_content"])
        }
    }

if __name__ == "__main__":

    uvicorn.run(app, host=settings.host, port=settings.port)