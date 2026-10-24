"""Orchestrates read -> beat tracking -> sections -> time map -> stretch -> verify -> write.

A single file and a set of stems go through the same path: one reference signal
is analysed (the file itself, a chosen stem, or the sum of all stems) and the
resulting time map is applied identically to every stem, so they stay
sample-aligned.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audio_io, segmentation, tempo_map
from .audio_io import Audio
from .beats import BeatResult, BeatTracker, DynamicProgrammingBeatTracker, TrackerConfig
from .stretchers import Stretcher

NARROW = 1.10   # tracking range around a known tempo (single lock)
WIDE = 1.40     # tracking range when sections may move away from it
MIX = "mix of all stems"


class StemError(ValueError):
    pass


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
    name: str                         # file name, or folder name for a stem set
    inputs: list[str]
    outputs: list[str]
    reference: str                    # what was analysed
    backend: str
    beats: int
    before: tempo_map.TempoStats
    sections: list[SectionResult]
    skipped: bool

    @property
    def target_bpm(self) -> float:
        return self.sections[0].bpm

    @property
    def is_stem_set(self) -> bool:
        return len(self.inputs) > 1

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

    # -- public ----------------------------------------------------------------
    def process(self, src: Path, dst: Path) -> Report:
        return self.process_stems([src], [dst], name=src.name)

    def process_stems(self, srcs: list[Path], dsts: list[Path], reference: Path | None = None,
                      name: str | None = None, jobs: int = 1) -> Report:
        """Lock a set of stems with one shared time map.

        `reference` (one of `srcs`) is analysed; without it the sum of all stems is.
        """
        if len(srcs) != len(dsts) or not srcs:
            raise StemError("need one output path per input")
        if reference is not None and reference not in srcs:
            raise StemError(f"reference {reference.name} is not one of the stems")
        stems = [audio_io.read(p) for p in srcs]
        sample_rate = _common_sample_rate(stems, srcs)
        frames = max(a.frames for a in stems)
        stems = [a.padded(frames) for a in stems]

        if reference is not None:
            ref_mono = stems[srcs.index(reference)].mono()
            ref_label = reference.name
        else:
            ref_mono = np.sum([a.mono() for a in stems], axis=0)
            ref_label = srcs[0].name if len(srcs) == 1 else MIX
        name = name or srcs[0].name
        inputs = [str(p) for p in srcs]

        o = self.options
        spread = (WIDE if o.tolerance else NARROW) if o.target_bpm else None
        result = self._tracker_factory(o.target_bpm or o.bpm_hint, spread).track(
            ref_mono, sample_rate)
        beats = result.beats
        before = tempo_map.analyse(result.raw_beats, result.repaired)
        sections = self._sections(result.repaired, beats)

        if len(sections) == 1 and (before.max_drift_ms < o.min_drift_ms
                                   and abs(before.detected_bpm - sections[0].bpm) < 0.01):
            return Report(name, inputs, [], ref_label, self.stretcher.name, len(beats), before,
                          self._results(beats, sections, None, None), skipped=True)

        tmap = tempo_map.build(beats, frames / sample_rate, sections)

        def render(i: int) -> np.ndarray | None:
            out = self.stretcher.stretch(stems[i], tmap)
            audio_io.write(dsts[i], out)
            return out.mono() if reference is None or srcs[i] == reference else None

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            rendered = list(pool.map(render, range(len(stems))))

        check = None
        if o.verify:
            if reference is not None:
                out_mono = rendered[srcs.index(reference)]
            else:
                out_mono = np.sum([m for m in rendered if m is not None], axis=0)
            verify_spread = WIDE if len(sections) > 1 else NARROW
            check = self._tracker_factory(sections[0].bpm, verify_spread).track(
                out_mono, sample_rate)
        return Report(name, inputs, [str(d) for d in dsts], ref_label, self.stretcher.name,
                      len(beats), before, self._results(beats, sections, tmap, check),
                      skipped=False)

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
                 tmap: tempo_map.TimeMap | None, check: BeatResult | None) -> list[SectionResult]:
        results = []
        for s in sections:
            after = None
            out_start = out_end = 0.0
            if tmap is not None:
                out_start = float(tmap.beat_targets[s.start])
                out_end = float(tmap.beat_targets[s.end])
                if check is not None:
                    raw = _within(check.raw_beats, out_start, out_end)
                    repaired = _within(check.repaired, out_start, out_end)
                    if len(repaired) >= 2 * tempo_map.ANALYSIS_WINDOW:
                        after = tempo_map.analyse(raw, repaired)
            results.append(SectionResult(float(beats[s.start]), float(beats[s.end]), out_start,
                                         s.end - s.start, s.detected_bpm, s.bpm, after))
        return results


def _common_sample_rate(stems: list[Audio], paths: list[Path]) -> int:
    rates = {a.sample_rate for a in stems}
    if len(rates) > 1:
        detail = ", ".join(f"{p.name}: {a.sample_rate} Hz" for p, a in zip(paths, stems))
        raise StemError(f"stems must share one sample rate ({detail})")
    return rates.pop()


def _within(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return times[(times >= start) & (times <= end)]
