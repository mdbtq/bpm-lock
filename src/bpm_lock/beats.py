"""Beat tracking: onset envelope + global tempo estimate + dynamic-programming
beat tracker (Ellis 2007), followed by sub-frame refinement and robust smoothing.

Only numpy/scipy are required. Implementations follow the `BeatTracker` protocol
so an alternative (e.g. librosa/madmom based) tracker can be plugged in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import signal

ANALYSIS_SR = 22050
HOP = 128          # 5.8 ms frames at 22.05 kHz
N_FFT = 2048
N_BANDS = 64
HALF_TIME_BELOW = 100.0  # BPM; slower estimates are re-checked at double tempo


class BeatTracker(Protocol):
    def track(self, mono: np.ndarray, sample_rate: int) -> "BeatResult": ...


@dataclass(frozen=True)
class BeatResult:
    beats: np.ndarray       # beat times in seconds (smoothed)
    raw_beats: np.ndarray   # beat times before smoothing
    tempo: float            # global tempo estimate (BPM) at the tracked metrical level


@dataclass(frozen=True)
class TrackerConfig:
    bpm_hint: float = 120.0
    bpm_min: float = 60.0
    bpm_max: float = 240.0
    tightness: float = 100.0
    smoothing_beats: int = 9   # odd window for drift smoothing; 0 disables


class DynamicProgrammingBeatTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

    # -- public ------------------------------------------------------------
    def track(self, mono: np.ndarray, sample_rate: int) -> BeatResult:
        y = _resample(mono, sample_rate, ANALYSIS_SR)
        env = onset_envelope(y)
        fps = ANALYSIS_SR / HOP
        tempo = self._estimate_tempo(env, fps, self.config.bpm_min, self.config.bpm_max)
        if tempo < HALF_TIME_BELOW and 2 * tempo <= self.config.bpm_max:
            # Half-time tracking can hop between kick and snare phases; the
            # beat level is more stable because every beat carries an onset.
            tempo = self._estimate_tempo(env, fps, 1.8 * tempo, 2.2 * tempo)
        frames = self._dp_beats(env, fps, tempo)
        if len(frames) < 4:
            raise ValueError("Too few beats detected; is this rhythmic material?")
        raw = _refine(env, frames) / fps
        smoothed = _robust_smooth(raw, self.config.smoothing_beats)
        return BeatResult(beats=smoothed, raw_beats=raw, tempo=tempo)

    # -- tempo ---------------------------------------------------------------
    def _estimate_tempo(self, env: np.ndarray, fps: float,
                        bpm_min: float, bpm_max: float) -> float:
        c = self.config
        x = env - env.mean()
        n = len(x)
        spec = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        min_lag = int(np.floor(60 * fps / bpm_max))
        max_lag = min(int(np.ceil(60 * fps / bpm_min)), n - 1)
        lags = np.arange(min_lag, max_lag + 1)
        bpms = 60 * fps / lags
        prior = np.exp(-0.5 * (np.log2(bpms / c.bpm_hint) / 1.0) ** 2)
        # Duple-meter salience: a real beat period also repeats at 2x/4x (bars) and
        # is subdivided at 1/2 (8ths), while a dotted/triplet period (3/2 beat) is
        # subdivided in thirds. This suppresses 2/3-tempo errors.
        ac = np.maximum(ac, 0.0) / (ac[0] + 1e-12)

        def at(m: float) -> np.ndarray:
            return ac[np.minimum(np.round(lags * m).astype(np.int64), n - 1)]

        salience = at(1) + at(2) + at(4) + at(0.5) - at(1 / 3)
        score = np.maximum(salience, 0.0) * prior
        i = int(np.argmax(score))
        # parabolic refinement of the lag
        if 0 < i < len(score) - 1:
            a, b, d = score[i - 1], score[i], score[i + 1]
            denom = a - 2 * b + d
            shift = 0.5 * (a - d) / denom if denom != 0 else 0.0
        else:
            shift = 0.0
        return float(60 * fps / (lags[i] + shift))

    # -- dynamic programming ------------------------------------------------
    def _dp_beats(self, env: np.ndarray, fps: float, tempo: float) -> np.ndarray:
        period = 60 * fps / tempo
        norm = env / (env.std() + 1e-9)
        win = np.arange(-int(period), int(period) + 1)
        gauss = np.exp(-0.5 * (win * 32.0 / period) ** 2)
        local = np.convolve(norm, gauss, mode="same")

        lo, hi = int(round(-2 * period)), int(round(-period / 2))
        offsets = np.arange(lo, hi + 1)
        txcost = -self.config.tightness * np.log(-offsets / period) ** 2

        n = len(local)
        cum = np.zeros(n)
        back = np.full(n, -1, dtype=np.int64)
        first = True
        thresh = 0.01 * local.max()
        for i in range(n):
            idx = i + offsets
            valid = idx >= 0
            if not valid.any():
                cum[i] = local[i]
                continue
            cand = cum[idx[valid]] + txcost[valid]
            j = int(np.argmax(cand))
            cum[i] = local[i] + cand[j]
            back[i] = idx[valid][j]
            if first and local[i] < thresh:
                back[i] = -1
            else:
                first = False

        peaks = signal.argrelmax(cum)[0]
        if len(peaks) == 0:
            return np.array([], dtype=np.int64)
        good = peaks[cum[peaks] >= 0.5 * np.median(cum[peaks])]
        beat = int(good[-1])
        beats = [beat]
        while back[beat] >= 0:
            beat = int(back[beat])
            beats.append(beat)
        beats = np.array(beats[::-1])
        return _trim_weak(beats, local)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _band_filterbank(n_fft: int, sr: int, n_bands: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(30.0, 11000.0, n_bands + 2)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[b] = np.clip(np.minimum(up, down), 0, None)
    fb /= fb.sum(axis=1, keepdims=True) + 1e-12
    return fb


def onset_envelope(y: np.ndarray) -> np.ndarray:
    """Log-band spectral flux, one value per HOP samples."""
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad))
    n_frames = 1 + (len(yp) - N_FFT) // HOP
    window = signal.get_window("hann", N_FFT).astype(np.float32)
    fb = _band_filterbank(N_FFT, ANALYSIS_SR, N_BANDS)
    bands = np.empty((n_frames, N_BANDS), dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(yp, N_FFT)[::HOP]
    chunk = 4096
    for s in range(0, n_frames, chunk):
        mag = np.abs(np.fft.rfft(frames[s:s + chunk] * window, axis=1))
        bands[s:s + chunk] = mag @ fb.T
    logb = np.log1p(1000.0 * bands)
    flux = np.maximum(0.0, np.diff(logb, axis=0, prepend=logb[:1])).sum(axis=1)
    # remove slow trend (local mean) and half-wave rectify
    trend = signal.filtfilt(*signal.butter(1, 0.4 / (ANALYSIS_SR / HOP / 2)), flux)
    env = np.maximum(0.0, flux - trend)
    edge = N_FFT // HOP  # file start/end boundaries are not musical onsets
    env[:edge] = 0.0
    env[-edge:] = 0.0
    return env / (env.max() + 1e-9)


def _trim_weak(beats: np.ndarray, local: np.ndarray) -> np.ndarray:
    strength = local[beats]
    thresh = 0.5 * np.sqrt(np.mean(strength ** 2))
    strong = np.nonzero(strength >= thresh)[0]
    if len(strong) == 0:
        return beats
    return beats[strong[0]: strong[-1] + 1]


def _refine(env: np.ndarray, frames: np.ndarray, radius: int = 3) -> np.ndarray:
    """Move each beat to the nearest onset peak with parabolic sub-frame accuracy."""
    out = frames.astype(np.float64)
    n = len(env)
    for k, f in enumerate(frames):
        lo, hi = max(1, f - radius), min(n - 2, f + radius)
        if hi <= lo:
            continue
        p = lo + int(np.argmax(env[lo:hi + 1]))
        a, b, c = env[p - 1], env[p], env[p + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom < 0 else 0.0
        out[k] = p + np.clip(shift, -0.5, 0.5)
    return out


def _robust_smooth(times: np.ndarray, window: int, iterations: int = 3) -> np.ndarray:
    """Local quadratic fit of beat time vs. beat index with outlier rejection.

    Removes per-beat detection jitter while following slow tempo drift.
    """
    if window <= 2 or len(times) < window:
        return times.copy()
    window += 1 - window % 2
    x = times.copy()
    for _ in range(iterations):
        fit = signal.savgol_filter(x, window, polyorder=2, mode="interp")
        resid = times - fit
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        outlier = np.abs(resid) > 4 * 1.4826 * mad
        x = np.where(outlier, fit, times)
    return signal.savgol_filter(x, window, polyorder=2, mode="interp")
