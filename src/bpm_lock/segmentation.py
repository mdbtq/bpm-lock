"""Split a track into tempo sections.

Within a section the tempo is locked to one baseline. When the tempo leaves
`baseline ± tolerance` and stays out for `hold_beats` consecutive beats, a new
section starts at the (refined) change point with a fresh baseline measured
there, checked against the same tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beats import robust_smooth
from .tempo_map import ANALYSIS_WINDOW, round_bpm

BASELINE_BEATS = 16   # beats used to measure a section's baseline tempo
STEP_WINDOW = 8       # beats on each side when locating the exact change point


@dataclass(frozen=True)
class TempoSection:
    start: int            # first beat index (shared with the previous section's end)
    end: int              # last beat index
    detected_bpm: float   # least-squares tempo of the section
    bpm: float            # target tempo the section is locked to


def single(beats: np.ndarray, bpm: float) -> list[TempoSection]:
    return [TempoSection(0, len(beats) - 1, _fit_bpm(beats), bpm)]


class TempoSegmenter:
    def __init__(self, tolerance: float, hold_beats: int = 8, min_beats: int = 32,
                 rounding: float | None = 1.0, fixed_bpm: float | None = None) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        self.tolerance = tolerance
        self.hold = max(2, hold_beats)
        self.min_beats = max(BASELINE_BEATS, min_beats)
        self.rounding = rounding
        self.fixed_bpm = fixed_bpm

    def split(self, raw_beats: np.ndarray) -> list[TempoSection]:
        curve = robust_smooth(raw_beats, ANALYSIS_WINDOW)
        local = 60.0 / np.diff(curve)          # local[i]: tempo between beat i and i+1
        n = len(local)
        bounds = [0]
        start = 0
        while True:
            cut = self._find_change(local, start)
            if cut is None:
                break
            bounds.append(cut)
            start = cut
        bounds.append(n)
        bounds = self._absorb_short(local, bounds)
        sections = [self._section(curve, a, b) for a, b in zip(bounds, bounds[1:])]
        return self._merge_equal(curve, sections)

    # -- internals -------------------------------------------------------------
    def _find_change(self, local: np.ndarray, start: int) -> int | None:
        n = len(local)
        if n - start < 2 * BASELINE_BEATS:
            return None
        baseline = float(np.median(local[start:start + BASELINE_BEATS]))
        outside = np.abs(local - baseline) > self.tolerance
        run = 0
        for i in range(start + BASELINE_BEATS, n):
            run = run + 1 if outside[i] else 0
            if run >= self.hold:
                cut = self._refine(local, start, i - self.hold + 1)
                # leave room for the next baseline; otherwise keep the tail in this section
                return cut if n - cut >= BASELINE_BEATS else None
        return None

    def _refine(self, local: np.ndarray, start: int, exit_at: int) -> int:
        """The smoothed curve lags the real change; pick the beat with the
        largest tempo step between the windows before and after it."""
        w = STEP_WINDOW
        lo = max(start + BASELINE_BEATS // 2, exit_at - 2 * self.hold)
        hi = min(len(local) - w, exit_at + self.hold)
        if hi <= lo:
            return exit_at
        steps = [abs(np.median(local[k:k + w]) - np.median(local[max(start, k - w):k]))
                 for k in range(lo, hi)]
        return lo + int(np.argmax(steps))

    def _absorb_short(self, local: np.ndarray, bounds: list[int]) -> list[int]:
        """Sections shorter than `min_beats` are usually fills or breaks where the
        beat is unclear, not real tempo changes: fold them into the neighbour
        whose tempo is closest."""
        bounds = list(bounds)
        while len(bounds) > 2:
            lengths = np.diff(bounds)
            k = int(np.argmin(lengths))
            if lengths[k] >= self.min_beats:
                break
            tempo = lambda j: float(np.median(local[bounds[j]:bounds[j + 1]]))
            here = tempo(k)
            left = abs(tempo(k - 1) - here) if k > 0 else np.inf
            right = abs(tempo(k + 1) - here) if k + 1 < len(lengths) else np.inf
            del bounds[k if left <= right else k + 1]
        return bounds

    def _section(self, curve: np.ndarray, a: int, b: int) -> TempoSection:
        detected = _fit_bpm(curve[a:b + 1])
        if self.fixed_bpm is not None and abs(detected - self.fixed_bpm) <= self.tolerance:
            target = self.fixed_bpm
        else:
            target = round_bpm(detected, self.rounding)
        return TempoSection(a, b, detected, target)

    def _merge_equal(self, curve: np.ndarray,
                     sections: list[TempoSection]) -> list[TempoSection]:
        """Neighbours that end up on the same target tempo become one section."""
        merged = [sections[0]]
        for s in sections[1:]:
            prev = merged[-1]
            if abs(prev.bpm - s.bpm) < 1e-6:
                merged[-1] = TempoSection(prev.start, s.end,
                                          _fit_bpm(curve[prev.start:s.end + 1]), prev.bpm)
            else:
                merged.append(s)
        return merged


def _fit_bpm(beats: np.ndarray) -> float:
    return float(60.0 / np.polyfit(np.arange(len(beats)), beats, 1)[0])
