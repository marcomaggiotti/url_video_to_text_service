# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from pathlib import Path

app = FastAPI()

# ---------- domain logic ----------

def generate_files() -> tuple[Path, Path, Path]:
    """
    Your existing function that creates/returns:
    1) .webm file
    2) .srt file
    3) .txt file
    """
    webm_path = Path("output/video.webm")
    srt_path = Path("output/subtitles.srt")
    txt_path = Path("output/transcript.txt")
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
async def get_files():
    webm_path, srt_path, txt_path = generate_files()

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
