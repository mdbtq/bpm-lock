"""Audio decoding/encoding through ffmpeg (any format ffmpeg understands)."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class AudioIOError(RuntimeError):
    pass


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # shape (channels, frames), float32 in [-1, 1]
    sample_rate: int

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def frames(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def padded(self, frames: int) -> "Audio":
        """Zero-pad at the end to `frames` (stems of slightly different length)."""
        if frames <= self.frames:
            return self
        return Audio(np.pad(self.samples, ((0, 0), (0, frames - self.frames))), self.sample_rate)


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise AudioIOError(f"'{binary}' not found on PATH")
    return path


def probe(path: Path) -> tuple[int, int]:
    """Return (sample_rate, channels) of the first audio stream."""
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise AudioIOError(f"ffprobe failed for {path}: {out.stderr.strip()}")
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise AudioIOError(f"No audio stream in {path}")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def read(path: Path) -> Audio:
    sample_rate, channels = probe(path)
    out = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, check=False,
    )
    if out.returncode != 0:
        raise AudioIOError(f"ffmpeg decode failed for {path}: {out.stderr.decode().strip()}")
    data = np.frombuffer(out.stdout, dtype=np.float32)
    data = data[: len(data) - len(data) % channels].reshape(-1, channels).T
    return Audio(np.ascontiguousarray(data), sample_rate)


_CODECS = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".aif": ["-c:a", "pcm_s24be"],
    ".aiff": ["-c:a", "pcm_s24be"],
    ".flac": ["-c:a", "flac", "-sample_fmt", "s32"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    ".m4a": ["-c:a", "aac", "-b:a", "320k"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "8"],
}


def write(path: Path, audio: Audio) -> None:
    codec = _CODECS.get(path.suffix.lower())
    if codec is None:
        raise AudioIOError(f"Unsupported output extension: {path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio.samples, -1.0, 1.0).T.astype("<f4").tobytes()
    out = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", "-f", "f32le",
         "-ar", str(audio.sample_rate), "-ac", str(audio.channels), "-i", "-",
         *codec, str(path)],
        input=pcm, capture_output=True, check=False,
    )
    if out.returncode != 0:
        raise AudioIOError(f"ffmpeg encode failed for {path}: {out.stderr.decode().strip()}")
