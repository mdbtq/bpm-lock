"""Turn detected beats into a source->target time map on a fixed-tempo grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .beats import nearest_distance, robust_smooth

if TYPE_CHECKING:
    from .segmentation import TempoSection

ANALYSIS_WINDOW = 15  # beats; separates slow drift from per-hit timing
EDGE_BEATS = 8


@dataclass(frozen=True)
class TimeMap:
    source: np.ndarray   # anchor times in the input (seconds), strictly increasing
    target: np.ndarray   # corresponding times in the output (seconds)
    beat_targets: np.ndarray  # output time of every detected beat

    @property
    def output_duration(self) -> float:
        return float(self.target[-1])

    def ratios(self) -> np.ndarray:
        """Stretch factor per segment (>1 = longer/slower)."""
        return np.diff(self.target) / np.diff(self.source)


@dataclass(frozen=True)
class TempoStats:
    detected_bpm: float
    min_bpm: float          # slowest local tempo (smoothed, 2nd percentile)
    max_bpm: float          # fastest local tempo (smoothed, 98th percentile)
    max_drift_ms: float     # deviation of the tempo curve from a fixed grid (p98)
    spread_ms: float        # per-hit timing spread around the tempo curve (p95)


def octave_factor(bpm: float, low: float = 100.0) -> float:
    """Power of two that lifts a half-time beat level (< `low`) to its musical
    tempo. Fast tempi (200+) are left alone."""
    return float(2.0 ** np.ceil(np.log2(low / bpm))) if bpm < low else 1.0


def round_bpm(bpm: float, rounding: float | None) -> float:
    """Round at the musical level (>= 100 BPM) so half-time tracking still lands
    on whole BPMs."""
    if not rounding:
        return bpm
    f = octave_factor(bpm)
    return round(bpm * f / rounding) * rounding / f


def choose_target_bpm(beats: np.ndarray, rounding: float | None) -> float:
    """Least-squares constant tempo = the tempo that needs the least stretching overall."""
    slope = np.polyfit(np.arange(len(beats)), beats, 1)[0]
    return round_bpm(60.0 / float(slope), rounding)


def analyse(raw_beats: np.ndarray, repaired: np.ndarray | None = None,
            window: int = ANALYSIS_WINDOW) -> TempoStats:
    """Separate tempo drift (slow, what bpm-lock removes) from per-hit timing
    spread (fast, the groove/feel or detection noise, which is preserved).

    The tempo curve is built from `repaired` beats (unreliable stretches such as
    fills interpolated) when given; spread is always measured on the raw hits."""
    curve = robust_smooth(raw_beats if repaired is None else repaired, window)
    # The first/last beats sit in intros/outros/fades where tracking is least
    # reliable and smoothing has no context; keep them out of the statistics.
    core = slice(EDGE_BEATS, -EDGE_BEATS) if len(curve) > 4 * EDGE_BEATS else slice(None)
    beats = curve[core]
    raw = raw_beats[(raw_beats >= beats[0]) & (raw_beats <= beats[-1])]
    local = 60.0 / np.diff(beats)
    idx = np.arange(len(beats))
    slope, offset = np.polyfit(idx, beats, 1)
    drift = np.abs(beats - (slope * idx + offset))
    return TempoStats(
        detected_bpm=float(60.0 / slope),
        min_bpm=float(np.percentile(local, 2)),
        max_bpm=float(np.percentile(local, 98)),
        max_drift_ms=float(np.percentile(drift, 98) * 1000),
        spread_ms=float(np.percentile(nearest_distance(raw, beats), 95) * 1000),
    )


def build(beats: np.ndarray, duration: float, sections: Sequence["TempoSection"],
          edge_beats: int = 4) -> TimeMap:
    """Map every detected beat onto a grid that is uniform within each section.

    Sections are contiguous beat ranges, each with its own target tempo; the grid
    is continuous across section boundaries. Material before the first and after
    the last beat is stretched with the local ratio of the nearest beats so
    intros/outros keep their feel.
    """
    n = len(beats)
    period = np.empty(n - 1)               # target length of each beat interval
    for s in sections:
        period[s.start:s.end] = 60.0 / s.bpm
    ibi = np.diff(beats)
    r_in = period[0] / float(np.median(ibi[:edge_beats]))
    r_out = period[-1] / float(np.median(ibi[-edge_beats:]))

    grid = beats[0] * r_in + np.concatenate([[0.0], np.cumsum(period)])
    tail = grid[-1] + (duration - beats[-1]) * r_out

    source = np.concatenate([[0.0], beats, [duration]])
    target = np.concatenate([[0.0], grid, [tail]])
    keep = np.concatenate([[True], np.diff(source) > 1e-6])
    return TimeMap(source[keep], target[keep], grid)
