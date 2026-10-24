"""Synthetic drum tracks with a drifting tempo, for tests."""
from __future__ import annotations

import numpy as np

SR = 44100


def drifting_track(base_bpm: float, duration: float = 40.0, wobble: float = 0.008,
                   sr: int = SR, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Kick on 1+3, snare on 2+4, 8th hats, offbeat bass. Returns (stereo, beat_times)."""
    beats, t = [], 0.3
    while t < duration - 0.5:
        beats.append(t)
        t += 60 / (base_bpm * (1 + wobble * np.sin(2 * np.pi * t / 20)))
    return render(np.array(beats), duration, sr, seed), np.array(beats)


def stepped_track(changes: list[tuple[float, float]], duration: float = 60.0,
                  wobble: float = 0.006, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Intentional tempo changes: `changes` = [(start_s, bpm), ...] with slight drift."""
    beats, t = [], 0.3
    while t < duration - 0.5:
        bpm = [b for start, b in changes if t >= start][-1]
        beats.append(t)
        t += 60 / (bpm * (1 + wobble * np.sin(2 * np.pi * t / 9)))
    return render(np.array(beats), duration, sr), np.array(beats)


def render(beats: np.ndarray, duration: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    parts = render_stems(beats, duration, sr, seed)
    return parts["drums"] + parts["bass"]


def render_stems(beats: np.ndarray, duration: float, sr: int = SR,
                 seed: int = 0) -> dict[str, np.ndarray]:
    """Separate stereo stems: drums (kick/snare/hats), bass (offbeats), pad (no transients)."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)
    drums = np.zeros(n_samples, np.float32)
    low = np.zeros(n_samples, np.float32)
    n = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-n * 30)) * n) * np.exp(-n * 12)
    snare = rng.standard_normal(len(n)) * np.exp(-n * 25) * 0.5 \
        + np.sin(2 * np.pi * 190 * n) * np.exp(-n * 20) * 0.4
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 120) * 0.2
    bass = np.sin(2 * np.pi * 55 * n) * np.exp(-n * 8) * 0.4

    def add(y: np.ndarray, x: np.ndarray, at: float) -> None:
        s = int(at * sr)
        m = min(len(x), len(y) - s)
        y[s:s + m] += x[:m]

    for i, b in enumerate(beats):
        add(drums, kick if i % 2 == 0 else snare, b)
        if i + 1 < len(beats):
            half = b + (beats[i + 1] - b) / 2
            add(drums, hat, b)
            add(drums, hat, half)
            add(low, bass, half)
    t = np.arange(n_samples) / sr
    pad = (0.08 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t)))
    stereo = lambda y: np.stack([y, y]).astype(np.float32) * 0.6
    return {"drums": stereo(drums), "bass": stereo(low), "pad": stereo(pad.astype(np.float32))}


def kick_times(mono: np.ndarray, sr: int = SR) -> np.ndarray:
    """Locate kicks/snares independently of the library (matched-filter peaks)."""
    from scipy import signal
    band = signal.sosfiltfilt(signal.butter(4, [120, 400], "bandpass", fs=sr, output="sos"), mono)
    env = signal.sosfiltfilt(signal.butter(2, 80, fs=sr, output="sos"), np.abs(band))
    onset = np.maximum(np.diff(env, prepend=env[0]), 0)
    peaks, _ = signal.find_peaks(onset, height=0.25 * onset.max(), distance=int(0.25 * sr))
    return peaks / sr


def bass_onsets(mono: np.ndarray, sr: int = SR) -> np.ndarray:
    """Onsets of the 55 Hz offbeat bass notes."""
    from scipy import signal
    band = signal.sosfiltfilt(signal.butter(4, [35, 90], "bandpass", fs=sr, output="sos"), mono)
    env = signal.sosfiltfilt(signal.butter(2, 40, fs=sr, output="sos"), np.abs(band))
    onset = np.maximum(np.diff(env, prepend=env[0]), 0)
    peaks, _ = signal.find_peaks(onset, height=0.3 * onset.max(), distance=int(0.2 * sr))
    return peaks / sr
