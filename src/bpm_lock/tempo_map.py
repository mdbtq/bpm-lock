"""Turn detected beats into a source->target time map on a fixed-tempo grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeMap:
    source: np.ndarray   # anchor times in the input (seconds), strictly increasing
    target: np.ndarray   # corresponding times in the output (seconds)
    target_bpm: float

    @property
    def output_duration(self) -> float:
        return float(self.target[-1])

    def ratios(self) -> np.ndarray:
        """Stretch factor per segment (>1 = longer/slower)."""
        return np.diff(self.target) / np.diff(self.source)


@dataclass(frozen=True)
class TempoStats:
    detected_bpm: float
    min_bpm: float
    max_bpm: float
    max_drift_ms: float   # largest deviation of a beat from the ideal fixed grid


def octave_factor(bpm: float, low: float = 100.0) -> float:
    """Power of two that lifts a half-time beat level (< `low`) to its musical
    tempo. Fast tempi (200+) are left alone."""
    return float(2.0 ** np.ceil(np.log2(low / bpm))) if bpm < low else 1.0


def choose_target_bpm(beats: np.ndarray, rounding: float | None) -> float:
    """Detected tempo at the tracked beat level; rounding is applied at the
    musical level (>= 100 BPM) so half-time tracking still lands on whole BPMs."""
    # least-squares constant-tempo fit = the tempo that needs the least stretching overall
    slope = np.polyfit(np.arange(len(beats)), beats, 1)[0]
    bpm = 60.0 / float(slope)
    if rounding:
        f = octave_factor(bpm)
        bpm = round(bpm * f / rounding) * rounding / f
    return bpm


def analyse(beats: np.ndarray) -> TempoStats:
    ibi = np.diff(beats)
    local = 60.0 / ibi
    # best straight-line (constant tempo) fit: deviation shows accumulated drift
    idx = np.arange(len(beats))
    slope, offset = np.polyfit(idx, beats, 1)
    drift = beats - (slope * idx + offset)
    return TempoStats(
        detected_bpm=60.0 / slope,
        min_bpm=float(np.percentile(local, 2)),
        max_bpm=float(np.percentile(local, 98)),
        max_drift_ms=float(np.max(np.abs(drift)) * 1000),
    )


def build(beats: np.ndarray, duration: float, target_bpm: float,
          edge_beats: int = 4) -> TimeMap:
    """Map every detected beat onto a uniform grid of `target_bpm`.

    Material before the first and after the last beat is stretched with the
    local ratio of the nearest beats so intros/outros keep their feel.
    """
    period = 60.0 / target_bpm
    ibi = np.diff(beats)
    r_in = period / float(np.median(ibi[:edge_beats]))
    r_out = period / float(np.median(ibi[-edge_beats:]))

    first = beats[0] * r_in
    grid = first + period * np.arange(len(beats))
    tail = grid[-1] + (duration - beats[-1]) * r_out

    source = np.concatenate([[0.0], beats, [duration]])
    target = np.concatenate([[0.0], grid, [tail]])
    keep = np.concatenate([[True], np.diff(source) > 1e-6])
    return TimeMap(source[keep], target[keep], target_bpm)
