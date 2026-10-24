"""Orchestrates read -> beat tracking -> sections -> time map -> stretch -> verify -> write."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audio_io, segmentation, tempo_map
from .beats import BeatTracker, DynamicProgrammingBeatTracker, TrackerConfig
from .stretchers import Stretcher

NARROW = 1.10   # tracking range around a known tempo (single lock)
WIDE = 1.40     # tracking range when sections may move away from it


@dataclass(frozen=True)
class Options:
    target_bpm: float | None = None   # None = derive from the track
    rounding: float | None = 1.0      # round derived BPM to this step
    bpm_hint: float = 120.0
    smoothing_beats: int = 9
    tolerance: float | None = None    # BPM; None = one fixed tempo for the whole track
    hold_beats: int = 8               # beats the tempo must stay outside the tolerance
    min_section_beats: int = 32       # shorter sections are folded into a neighbour
    min_drift_ms: float = 0.0         # skip stretching when drift is below this
    verify: bool = True


@dataclass(frozen=True)
class SectionResult:
    start_s: float                    # section start in the input
    end_s: float
    out_start_s: float                # section start in the output
    beats: int
    detected_bpm: float
    bpm: float                        # locked tempo
    after: tempo_map.TempoStats | None


@dataclass(frozen=True)
class Report:
    input: str
    output: str | None
    backend: str
    beats: int
    before: tempo_map.TempoStats
    sections: list[SectionResult]
    skipped: bool

    @property
    def target_bpm(self) -> float:
        return self.sections[0].bpm

    def to_dict(self) -> dict:
        return asdict(self)


class BpmLocker:
    def __init__(self, stretcher: Stretcher, options: Options,
                 tracker_factory=None) -> None:
        self.stretcher = stretcher
        self.options = options
        self._tracker_factory = tracker_factory or self._default_tracker

    def _default_tracker(self, hint: float, spread: float | None) -> BeatTracker:
        if spread:   # known tempo: search only around it
            cfg = TrackerConfig(bpm_hint=hint, bpm_min=hint / spread, bpm_max=hint * spread,
                                smoothing_beats=self.options.smoothing_beats)
        else:
            cfg = TrackerConfig(bpm_hint=hint, smoothing_beats=self.options.smoothing_beats)
        return DynamicProgrammingBeatTracker(cfg)

    def process(self, src: Path, dst: Path) -> Report:
        o = self.options
        audio = audio_io.read(src)
        spread = (WIDE if o.tolerance else NARROW) if o.target_bpm else None
        result = self._tracker_factory(o.target_bpm or o.bpm_hint, spread).track(
            audio.mono(), audio.sample_rate)
        beats = result.beats
        before = tempo_map.analyse(result.raw_beats)
        sections = self._sections(result.raw_beats, beats)

        if len(sections) == 1 and (before.max_drift_ms < o.min_drift_ms
                                   and abs(before.detected_bpm - sections[0].bpm) < 0.01):
            return Report(str(src), None, self.stretcher.name, len(beats), before,
                          self._results(beats, sections, None, None), skipped=True)

        tmap = tempo_map.build(beats, audio.duration, sections)
        out = self.stretcher.stretch(audio, tmap)
        audio_io.write(dst, out)

        check = None
        if o.verify:
            verify_spread = WIDE if len(sections) > 1 else NARROW
            check = self._tracker_factory(sections[0].bpm, verify_spread).track(
                out.mono(), out.sample_rate).raw_beats
        return Report(str(src), str(dst), self.stretcher.name, len(beats), before,
                      self._results(beats, sections, tmap, check), skipped=False)

    # -- internals -------------------------------------------------------------
    def _sections(self, raw: np.ndarray, beats: np.ndarray) -> list[segmentation.TempoSection]:
        o = self.options
        if o.tolerance:
            return segmentation.TempoSegmenter(
                o.tolerance, o.hold_beats, o.min_section_beats, o.rounding,
                o.target_bpm).split(raw)
        target = o.target_bpm or tempo_map.choose_target_bpm(beats, o.rounding)
        return segmentation.single(beats, target)

    @staticmethod
    def _results(beats: np.ndarray, sections: list[segmentation.TempoSection],
                 tmap: tempo_map.TimeMap | None, check: np.ndarray | None) -> list[SectionResult]:
        results = []
        for s in sections:
            after = None
            out_start = out_end = 0.0
            if tmap is not None:
                out_start = float(tmap.beat_targets[s.start])
                out_end = float(tmap.beat_targets[s.end])
                if check is not None:
                    inside = check[(check >= out_start) & (check <= out_end)]
                    if len(inside) >= 2 * tempo_map.ANALYSIS_WINDOW:
                        after = tempo_map.analyse(inside)
            results.append(SectionResult(float(beats[s.start]), float(beats[s.end]), out_start,
                                         s.end - s.start, s.detected_bpm, s.bpm, after))
        return results
