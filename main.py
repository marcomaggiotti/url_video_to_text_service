# main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import base64
from pathlib import Path
from tools.srt_tools import *
app = FastAPI()

# ---------- domain logic ----------

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0"}

@app.get("/")
async def root():
    return {"message": "CAD-RAG API ready!"}

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
    content_b64: str

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

@app.get("/extract-text-2video", response_model=FilesResponse)
async def process_video(
    video_url: str = Query(..., description="YouTube/Direct video URL")):

    """ Processa URL → webm + srt + txt"""

    webm_path, srt_path, txt_path = download_video_to_txt(video_url)

    # Encoding...
    webm = FileJson(filename=webm_path.name, mimetype="video/webm", content_b64=encode_file(webm_path))
    srt = FileJson(filename=srt_path.name, mimetype="application/x-subrip", content_b64=encode_file(srt_path))
    text = FileJson(filename=txt_path.name, mimetype="text/plain", content_b64=encode_file(txt_path))

    return FilesResponse(webm=webm, srt=srt, text=text)