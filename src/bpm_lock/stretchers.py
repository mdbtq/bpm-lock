"""Variable-rate time-stretching backends (Strategy pattern).

Every backend receives the full time map and must place each source anchor
exactly at its target time, without changing pitch.

Order of preference (see `select`):
  1. rubberband   - Rubber Band CLI with a key-frame map. Best quality, exact.
  2. ffmpeg-rb    - ffmpeg's rubberband filter, tempo switched per beat. Very good.
  3. wsola        - Built-in numpy WSOLA. No dependencies; fine for small ratios.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.io import wavfile

from .audio_io import Audio
from .tempo_map import TimeMap


class StretchError(RuntimeError):
    pass


class Stretcher(Protocol):
    name: str

    @classmethod
    def available(cls) -> bool: ...

    def stretch(self, audio: Audio, time_map: TimeMap) -> Audio: ...


def _fit_length(x: np.ndarray, frames: int) -> np.ndarray:
    if x.shape[1] >= frames:
        return x[:, :frames]
    return np.pad(x, ((0, 0), (0, frames - x.shape[1])))


def _write_wav(path: Path, audio: Audio) -> None:
    wavfile.write(path, audio.sample_rate, audio.samples.T.astype(np.float32))


def _read_wav(path: Path) -> np.ndarray:
    with warnings.catch_warnings():  # Rubber Band adds chunks scipy doesn't parse
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        _, data = wavfile.read(path)
    data = data.astype(np.float32)
    return (data[:, None] if data.ndim == 1 else data).T


# ----------------------------------------------------------------------------
class RubberbandCliStretcher:
    name = "rubberband"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("rubberband") is not None

    def stretch(self, audio: Audio, time_map: TimeMap) -> Audio:
        sr = audio.sample_rate
        out_frames = int(round(time_map.output_duration * sr))
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src, dst, mapf = tmpdir / "in.wav", tmpdir / "out.wav", tmpdir / "map.txt"
            _write_wav(src, audio)
            pairs = np.column_stack([np.round(time_map.source * sr),
                                     np.round(time_map.target * sr)]).astype(np.int64)
            mapf.write_text("\n".join(f"{s} {d}" for s, d in pairs[1:-1]) + "\n")
            base = ["rubberband", "-q", "-D", f"{out_frames / sr:.6f}", "-M", str(mapf)]
            errors = []
            for engine in self._engines():
                res = subprocess.run([*base, *engine, str(src), str(dst)],
                                     capture_output=True, text=True)
                if res.returncode == 0 and dst.exists():
                    out = _read_wav(dst)
                    break
                errors.append(res.stderr.strip())
            else:
                raise StretchError(f"rubberband failed: {' | '.join(errors)}")
        return Audio(_fit_length(out, out_frames), sr)

    @staticmethod
    def _engines() -> list[list[str]]:
        """R3 'fine' engine (v3+) first, R2 with crisp transients as fallback."""
        help_text = subprocess.run(["rubberband", "--help"], capture_output=True,
                                   text=True)
        engines = [["--crisp", "5"]]
        if "--fine" in help_text.stdout + help_text.stderr:
            engines.insert(0, ["--fine"])
        return engines


# ----------------------------------------------------------------------------
class FfmpegRubberbandStretcher:
    """Switches the rubberband filter's tempo at every beat via `asendcmd`."""
    name = "ffmpeg-rb"
    BLOCK = 64  # samples per filter block -> command timing granularity

    @classmethod
    def available(cls) -> bool:
        if shutil.which("ffmpeg") is None:
            return False
        res = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True)
        return " rubberband " in res.stdout

    def stretch(self, audio: Audio, time_map: TimeMap) -> Audio:
        sr = audio.sample_rate
        out_frames = int(round(time_map.output_duration * sr))
        tempi = 1.0 / time_map.ratios()          # rubberband tempo = speed factor
        starts = time_map.source[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src, dst, cmdf = tmpdir / "in.wav", tmpdir / "out.wav", tmpdir / "cmd.txt"
            _write_wav(src, audio)
            cmdf.write_text("".join(
                f"{t:.6f} rubberband tempo {v:.8f};\n" for t, v in zip(starts[1:], tempi[1:])))
            graph = (f"asetnsamples=n={self.BLOCK}:p=0,"
                     f"asendcmd=f='{cmdf}',"
                     f"rubberband=tempo={tempi[0]:.8f}:transients=crisp:detector=compound:"
                     f"phase=laminar:window=standard:pitchq=quality:channels=together")
            res = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", graph,
                 "-c:a", "pcm_f32le", str(dst)],
                capture_output=True, text=True)
            if res.returncode != 0:
                raise StretchError(f"ffmpeg rubberband failed: {res.stderr.strip()}")
            out = _read_wav(dst)
        return Audio(_fit_length(out, out_frames), sr)


# ----------------------------------------------------------------------------
class WsolaStretcher:
    """Waveform-similarity overlap-add with a time-varying rate."""
    name = "wsola"

    def __init__(self, window_ms: float = 46.0, tolerance_ms: float = 12.0) -> None:
        self.window_ms = window_ms
        self.tolerance_ms = tolerance_ms

    @classmethod
    def available(cls) -> bool:
        return True

    def stretch(self, audio: Audio, time_map: TimeMap) -> Audio:
        sr = audio.sample_rate
        win = int(sr * self.window_ms / 1000) & ~1
        hop = win // 2
        tol = int(sr * self.tolerance_ms / 1000)
        window = np.hanning(win).astype(np.float32)

        x = audio.samples
        mono = audio.mono()
        n_in = x.shape[1]
        out_frames = int(round(time_map.output_duration * sr))
        n_blocks = out_frames // hop + 2

        # Source position for each output block via the inverse (target->source) map.
        out_pos = np.arange(n_blocks) * hop
        src_pos = np.interp(out_pos, time_map.target * sr, time_map.source * sr)

        pad = 2 * (win + tol)
        xp = np.pad(x, ((0, 0), (pad, pad)))
        mp = np.pad(mono, (pad, pad))
        y = np.zeros((x.shape[0], n_blocks * hop + win), dtype=np.float32)
        norm = np.zeros(n_blocks * hop + win, dtype=np.float32)
        ones = np.ones(win, dtype=np.float32)
        anchors = np.round(time_map.source * sr).astype(np.int64)

        prev = 0
        for k in range(n_blocks):
            want = int(np.clip(round(src_pos[k]), 0, n_in))
            if k == 0 or self._near_anchor(anchors, want, win):
                start = want  # keep beats exactly on the grid
            else:
                # pick the grain most similar to the natural continuation of the previous one
                ref = mp[pad + prev + hop: pad + prev + hop + win]
                lo = want - tol
                seg = mp[pad + lo: pad + lo + win + 2 * tol]
                corr = np.correlate(seg, ref, mode="valid")
                energy = np.sqrt(np.convolve(seg ** 2, ones, mode="valid")) + 1e-9
                start = lo + int(np.argmax(corr / energy))
            grain = xp[:, pad + start: pad + start + win] * window
            o = k * hop
            y[:, o:o + win] += grain
            norm[o:o + win] += window
            prev = start
        y /= np.maximum(norm, 1e-3)
        return Audio(_fit_length(y, out_frames), sr)

    @staticmethod
    def _near_anchor(anchors: np.ndarray, pos: int, win: int) -> bool:
        i = np.searchsorted(anchors, pos - win)
        return i < len(anchors) and anchors[i] <= pos + 2 * win


# ----------------------------------------------------------------------------
BACKENDS: dict[str, type] = {
    RubberbandCliStretcher.name: RubberbandCliStretcher,
    FfmpegRubberbandStretcher.name: FfmpegRubberbandStretcher,
    WsolaStretcher.name: WsolaStretcher,
}


def select(name: str = "auto") -> Stretcher:
    if name != "auto":
        cls = BACKENDS.get(name)
        if cls is None:
            raise StretchError(f"Unknown backend '{name}'. Options: {', '.join(BACKENDS)}")
        if not cls.available():
            raise StretchError(f"Backend '{name}' is not available on this system")
        return cls()
    for cls in BACKENDS.values():
        if cls.available():
            return cls()
    raise StretchError("No stretch backend available")
