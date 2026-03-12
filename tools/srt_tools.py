"""
YouTube Audio + Whisper STT Downloader (Production Fixed)
✅ deno/FFmpeg safe | ✅ Robust file detection | ✅ Whisper caching
"""

import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple, Optional
from pathlib import Path
import re
import os
import yt_dlp
import whisper
import faster_whisper
from pydantic import BaseModel
import logging

@dataclass
class SRTMatch:
    filename: str
    text: str
    start_time: str
    end_time: str
    seconds: float

class VideoResponse(BaseModel):
    summary: str
    action_items: List[str]
    sentiment: str
    speaker_times: dict
    total_duration: float

# =============================================================================
# CONFIG & UTILS
# =============================================================================

class Config:
    """Centralized configuration."""

    def __init__(self):
        self.audio_dir = Path("downloads")
        self.transcript_dir = Path("transcripts")
        self.min_audio_size = 100_000
        self.finalize_delay = 5
        self.audio_exts = {'.mp3', '.webm', '.m4a', '.opus', '.wav', '.aac'}

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
@dataclass
class SentenceTimestamp:
    """Sentence-level timestamp (not word-level)"""
    filename: str
    text: str
    start_time: str
    end_time: str


class TimeConverter:
    """Time format conversions"""

    @staticmethod
    def to_seconds(time_str: str) -> float:
        """HH:MM:SS,mmm → seconds"""
        try:
            h, m, s = time_str.replace(',', '.').split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        except:
            return 0.0

    @staticmethod
    def to_srt_format(seconds: float) -> str:
        """seconds → HH:MM:SS"""
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

class SRTParser:
    """Parse SRT → sentence-level timestamps (1 timestamp per SRT block)"""

    def __init__(self):
        self.time_converter = TimeConverter()

    def parse(self, srt_file: str) -> List[SentenceTimestamp]:
        """Parse SRT: 1 timestamp per subtitle block (sentence-level)"""
        logger.info(f"Parsing SRT: {srt_file}")

        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
        sentence_timestamps = []

        for block in blocks:
            ts = self._parse_srt_block(block, srt_file)
            if ts:
                sentence_timestamps.append(ts)

        logger.info(f"→ {len(sentence_timestamps)} sentence timestamps")
        return sentence_timestamps

    def _parse_srt_block(self, block: str,  filename: str) -> Optional[SentenceTimestamp]:
        """Parse single SRT block → 1 SentenceTimestamp"""
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            return None

        timestamp_line = lines[1]
        if ' --> ' not in timestamp_line:
            return None

        try:
            start_str, end_str = timestamp_line.split(' --> ')
            start_sec = self.time_converter.to_seconds(start_str)
            end_sec = self.time_converter.to_seconds(end_str)

            # Join subtitle text (multi-line OK)
            text = ' '.join(lines[2:])
            return SentenceTimestamp(filename= filename, text=text, start_time=start_sec, end_time=end_sec)
        except:
            return None

    def get_timestamps_of_text(self, chunk: str,  srt_list: List[SentenceTimestamp], previous_timestamp: float, start_index: int) :

        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())[:2]
        chunk_start = re.sub(r'\s+', ' ', ' '.join(sentences)).strip()
        best_time, best_score = None, 0

        current_index = start_index

        for i, st in enumerate(srt_list[start_index:], start_index):
            srt_text = re.sub(r'\s+', ' ', st.text.strip())

            if (st.text.lower() in chunk.lower() and previous_timestamp < st.start_time):
                return st.start_time, i

            score = SequenceMatcher(None, st.text.lower(), chunk_start.lower()).ratio()

            if score > best_score and score > 0.75:
                best_score = score
                best_time = st.start_time

        return best_time, current_index

        for index in srt_list:
            if chunk in index.text :
                return index.start_time

def extract_video_id(url: str) -> str:
    """Extract video ID (unchanged)."""
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    if not match:
        raise ValueError(f"Invalid URL: {url}")
    return match.group(1)


def safe_filename(name: str) -> str:
    """Sanitize filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def format_timestamp(seconds: float) -> str:
    """Float → SRT timestamp (unchanged)."""
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    ms = int((s % 1) * 1000)
    s = int(s)
    return f"{int(h):02d}:{int(m):02d}:{s:02d},{ms:03d}"


def _time_to_seconds(time_str: str) -> float:
    """SRT timestamp → seconds (unchanged)."""
    h, m, s = time_str.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


# =============================================================================
# ORIGINAL FUNCTIONS (refactored internally)
# =============================================================================

def find_srt_text(query: str, search_folder: str = "downloads") -> List[SRTMatch]:
    """Find query text in SRT files → return timestamps (unchanged signature)."""
    matches = []
    folder = Path(search_folder)

    for srt_file in folder.glob("*.srt"):
        matches.extend(_parse_srt_file(srt_file, query.lower()))

    return sorted(matches, key=lambda m: m.seconds)


def _parse_srt_file(srt_path: Path, query: str) -> List[SRTMatch]:
    """Internal: Parse single SRT."""
    matches = []
    content = srt_path.read_text(encoding='utf-8')
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        timestamp = lines[1]
        text = ' '.join(lines[2:]).lower().strip()

        if query in text:
            start, end = timestamp.split(' --> ')
            matches.append(SRTMatch(
                filename=srt_path.name,
                text=' '.join(lines[2:]),
                start_time=start.strip(),
                end_time=end.strip(),
                seconds=_time_to_seconds(start.strip())
            ))
    return matches


def download_audio(video_url: str, output_dir: str = "downloads") -> str:
    """Download audio - robust detection (unchanged signature)."""
    config = Config()
    config.audio_dir = Path(output_dir)
    video_id = extract_video_id(video_url)
    print(f"🎯 Target: {video_id}")

    # Extracted find_audio_files logic
    def find_audio_files() -> list[Path]:
        return [f for f in config.audio_dir.iterdir()
                if (f.is_file() and
                    f.suffix.lower() in config.audio_exts and
                    f.stat().st_size > config.min_audio_size and
                    video_id.lower() in f.name.lower())]

    # Check existing
    audio_files = find_audio_files()
    if audio_files:
        largest = max(audio_files, key=os.path.getsize)
        print(f"⏭️  Using existing: {largest.name} ({largest.stat().st_size / 1e6:.1f}MB)")
        return str(largest)

    # Download
    print("📥 Downloading...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(config.audio_dir / '%(title)s_%(id)s.%(ext)s'),
        'quiet': True,
        'ignoreerrors': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    time.sleep(config.finalize_delay)

    audio_files = find_audio_files()
    if audio_files:
        largest = max(audio_files, key=os.path.getsize)
        print(f"✅ Downloaded: {largest.name} ({largest.stat().st_size / 1e6:.1f}MB)")
        return str(largest)

    raise FileNotFoundError(f"No audio for {video_id}")


def transcribe_audio(audio_file: str, output_dir: str = "transcripts",
                     model_size: str = "tiny", language: str = "en"):
    """Whisper STT - skip if exists (unchanged signature)."""
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_file}")

    config = Config()
    config.transcript_dir = Path(output_dir)

    safe_name = safe_filename(audio_path.stem)
    srt_file = config.transcript_dir / f"{safe_name}_whisper.srt"
    txt_file = config.transcript_dir / f"{safe_name}_whisper.txt"

    # Skip if exists
    if (srt_file.exists() and txt_file.exists() and
            srt_file.stat().st_size > 1000 and txt_file.stat().st_size > 100):
        print("⏭️  Whisper exists")
        return str(srt_file), str(txt_file)

    print(f"🎤 Whisper: {audio_path.name}")
    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    segments = result["segments"]

    # SRT
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        srt_content.extend([str(i), f"{start} --> {end}", segment['text'].strip(), ""])

    srt_file.write_text('\n'.join(srt_content), encoding='utf-8')

    # TXT
    txt_file.write_text(' '.join(segment['text'] for segment in segments), encoding='utf-8')

    print(f"✅ {len(segments)} segments")
    return str(srt_file), str(txt_file)


def download_video_to_txt(video_url: str, output_dir: str = "downloads",
                              language: str = "en"):

    """Audio + Whisper transcripts (unchanged signature)."""
    audio_file = download_audio(video_url, output_dir)
    whisper_srt, whisper_txt = transcribe_audio(audio_file, output_dir, "base", language)

    print("\n✅ FILES:")
    print(f"Audio: {audio_file}")
    print(f"Whisper SRT: {whisper_srt}")
    print(f"Whisper TXT: {whisper_txt}")

    return audio_file, whisper_srt, whisper_txt

# =============================================================================
# CLI (unchanged)
# =============================================================================

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_video_to_txt(url)
