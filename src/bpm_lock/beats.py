"""Beat tracking: onset envelope + global tempo estimate + dynamic-programming
beat tracker (Ellis 2007), followed by sub-frame refinement and robust smoothing.

Only numpy/scipy are required. Implementations follow the `BeatTracker` protocol
so an alternative (e.g. librosa/madmom based) tracker can be plugged in.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Protocol

import numpy as np
from scipy import signal

ANALYSIS_SR = 22050
HOP = 128          # 5.8 ms frames at 22.05 kHz
N_FFT = 2048
N_BANDS = 64
HALF_TIME_BELOW = 100.0  # BPM; slower estimates are re-checked at double tempo
MAX_INTERVAL_JUMP = 0.08  # a beat interval deviating more than this from its
                          # neighbourhood marks unreliable tracking (fills, breaks)
MAX_PHASE_OFFSET = 0.10   # fraction of a beat a hit may sit off the reference curve
PHASE_WINDOW = 33         # beats in the reference curve for the phase check


class BeatTracker(Protocol):
    def track(self, mono: np.ndarray, sample_rate: int) -> "BeatResult": ...


@dataclass(frozen=True)
class BeatResult:
    beats: np.ndarray       # beat times in seconds: repaired, then smoothed
    raw_beats: np.ndarray   # beat times as detected
    repaired: np.ndarray    # raw beats with unreliable stretches rebuilt (may differ in count)
    reliable: np.ndarray    # bool per repaired beat; False where it was rebuilt
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
        repaired, reliable = repair(raw, reliable_mask(raw))
        smoothed = robust_smooth(repaired, self.config.smoothing_beats)
        return BeatResult(beats=smoothed, raw_beats=raw, repaired=repaired,
                          reliable=reliable, tempo=tempo)

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


def reliable_mask(beats: np.ndarray, side: int = 8,
                  max_jump: float = MAX_INTERVAL_JUMP, bridge: int = 2) -> np.ndarray:
    """Flag beats the tracker placed implausibly (fills, breaks, unclear beat).

    An interval is suspect when it deviates from the typical interval both
    before and after it; a real tempo change matches one of the two sides and is
    left alone. Positions are then checked against a long, robust reference
    curve, built separately on each side of real tempo changes, because in a
    fill the tracker can slip behind with ordinary-looking intervals (or drop
    beats) and catch up later.
    """
    ibi = np.diff(beats)
    before = _side_median(ibi, side, past=True)
    after = _side_median(ibi, side, past=False)
    with np.errstate(invalid="ignore"):
        off_before = np.abs(ibi / before - 1.0) > max_jump
        off_after = np.abs(ibi / after - 1.0) > max_jump
    bad_interval = np.where(np.isnan(before), off_after,
                            np.where(np.isnan(after), off_before, off_before & off_after))
    # close short gaps between bad intervals: fills alternate good and bad hits
    idx = np.nonzero(bad_interval)[0]
    for a, b in zip(idx, idx[1:]):
        if b - a <= bridge + 1:
            bad_interval[a:b] = True
    reliable = np.ones(len(beats), dtype=bool)
    reliable[1:][bad_interval] = False
    reliable[:-1][bad_interval] = False

    step_times = beats[tempo_steps(ibi, side, max_jump) + 1]
    max_offset = MAX_PHASE_OFFSET * float(np.median(ibi))
    for _ in range(6):
        repaired, _ = repair(beats, reliable)
        reference = _piecewise_smooth(repaired, step_times, PHASE_WINDOW)
        updated = reliable & (nearest_distance(beats, reference) <= max_offset)
        if (updated == reliable).all():
            break
        reliable = updated
    return reliable


def tempo_steps(ibi: np.ndarray, side: int, min_change: float) -> np.ndarray:
    """Interval indices where the typical interval changes by more than
    `min_change` between the `side` intervals before and after: real tempo changes."""
    before = _side_median(ibi, side, past=True)
    after = _side_median(ibi, side, past=False)
    with np.errstate(invalid="ignore"):
        change = np.abs(np.log(after / before))
    change = np.nan_to_num(change)
    above = change > np.log1p(min_change)
    steps = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            steps.append(i + int(np.argmax(change[i:j])))
            i = j
        else:
            i += 1
    return np.asarray(steps, dtype=np.int64)


def _side_median(x: np.ndarray, side: int, past: bool) -> np.ndarray:
    """Median of the `side` values before (past) or after each element, excluding
    the element itself; NaN where fewer than half of them exist."""
    pad = np.full(side, np.nan)
    padded = np.concatenate([pad, x, pad])
    windows = np.lib.stride_tricks.sliding_window_view(padded, side)
    start = np.arange(len(x)) + (0 if past else side + 1)
    w = windows[start]
    with warnings.catch_warnings():   # all-NaN windows at the edges are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(w, axis=1)
    med[np.sum(~np.isnan(w), axis=1) < side // 2] = np.nan
    return med


def _piecewise_smooth(beats: np.ndarray, boundaries: np.ndarray, window: int) -> np.ndarray:
    """robust_smooth applied separately between boundary times (tempo changes)."""
    cuts = np.searchsorted(beats, boundaries)
    edges = [0, *[int(c) for c in cuts if 0 < c < len(beats)], len(beats)]
    return np.concatenate([robust_smooth(beats[a:b], window)
                           for a, b in zip(edges, edges[1:]) if b > a])


def repair(beats: np.ndarray, reliable: np.ndarray,
           context: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild unreliable stretches at constant tempo between their reliable
    neighbours. The number of beats in a stretch is re-counted from the tempo of
    the surrounding reliable beats, so beats the tracker dropped or doubled in a
    fill are restored. Returns (beats, reliable) for the rebuilt sequence."""
    good = np.nonzero(reliable)[0]
    if reliable.all() or len(good) < 2:
        return beats.copy(), reliable.copy()

    # intervals between directly adjacent reliable beats, for local tempo
    pair = good[np.nonzero(np.diff(good) == 1)[0]]
    pair_ibi = beats[pair + 1] - beats[pair]
    global_period = float(np.median(pair_ibi)) if len(pair) else float(np.median(np.diff(beats)))

    def period_near(a: int, b: int) -> float:
        near = (pair >= a - context) & (pair <= b + context)
        return float(np.median(pair_ibi[near])) if near.sum() >= 4 else global_period

    out: list[float] = []
    mask: list[bool] = []
    first, last = good[0], good[-1]
    p_in = period_near(first, first)
    out.extend(beats[first] - p_in * np.arange(first, 0, -1))      # lead-in, count kept
    mask.extend([False] * first)
    for g0, g1 in zip(good, good[1:]):
        out.append(beats[g0])
        mask.append(True)
        if g1 - g0 > 1:
            span = beats[g1] - beats[g0]
            n = max(1, int(round(span / period_near(g0, g1))))
            out.extend(beats[g0] + span * np.arange(1, n) / n)
            mask.extend([False] * (n - 1))
    out.append(beats[last])
    mask.append(True)
    p_out = period_near(last, last)
    tail = len(beats) - 1 - last
    out.extend(beats[last] + p_out * np.arange(1, tail + 1))         # outro, count kept
    mask.extend([False] * tail)
    return np.asarray(out), np.asarray(mask)


def nearest_distance(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Distance from each time to the closest point of a sorted grid."""
    i = np.clip(np.searchsorted(grid, times), 1, len(grid) - 1)
    return np.minimum(np.abs(times - grid[i - 1]), np.abs(times - grid[i]))


def robust_smooth(times: np.ndarray, window: int, iterations: int = 3) -> np.ndarray:
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
