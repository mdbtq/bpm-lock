"""Orchestrates read -> beat tracking -> time map -> stretch -> verify -> write."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from . import audio_io, tempo_map
from .beats import BeatTracker, DynamicProgrammingBeatTracker, TrackerConfig
from .stretchers import Stretcher


@dataclass(frozen=True)
class Options:
    target_bpm: float | None = None   # None = derive from the track
    rounding: float | None = 1.0      # round derived BPM to this step
    bpm_hint: float = 120.0
    smoothing_beats: int = 9
    min_drift_ms: float = 0.0         # skip stretching when drift is below this
    verify: bool = True


@dataclass(frozen=True)
class Report:
    input: str
    output: str | None
    backend: str
    beats: int
    target_bpm: float
    before: tempo_map.TempoStats
    after: tempo_map.TempoStats | None
    skipped: bool

    def to_dict(self) -> dict:
        return asdict(self)


class BpmLocker:
    def __init__(self, stretcher: Stretcher, options: Options,
                 tracker_factory=None) -> None:
        self.stretcher = stretcher
        self.options = options
        self._tracker_factory = tracker_factory or self._default_tracker

    def _default_tracker(self, hint: float, narrow: bool) -> BeatTracker:
        if narrow:   # known target tempo: search only around it
            cfg = TrackerConfig(bpm_hint=hint, bpm_min=hint * 0.9, bpm_max=hint * 1.1,
                                smoothing_beats=self.options.smoothing_beats)
        else:
            cfg = TrackerConfig(bpm_hint=hint, smoothing_beats=self.options.smoothing_beats)
        return DynamicProgrammingBeatTracker(cfg)

    def process(self, src: Path, dst: Path) -> Report:
        o = self.options
        audio = audio_io.read(src)
        hint = o.target_bpm or o.bpm_hint
        result = self._tracker_factory(hint, o.target_bpm is not None).track(
            audio.mono(), audio.sample_rate)
        before = tempo_map.analyse(result.raw_beats)
        target = o.target_bpm or tempo_map.choose_target_bpm(result.beats, o.rounding)

        already_fixed = (before.max_drift_ms < o.min_drift_ms
                         and abs(before.detected_bpm - target) < 0.01)
        if already_fixed:
            return Report(str(src), None, self.stretcher.name, len(result.beats),
                          target, before, None, skipped=True)

        tmap = tempo_map.build(result.beats, audio.duration, target)
        out = self.stretcher.stretch(audio, tmap)
        audio_io.write(dst, out)

        after = None
        if o.verify:
            check = self._tracker_factory(target, True).track(out.mono(), out.sample_rate)
            after = tempo_map.analyse(check.raw_beats)
        return Report(str(src), str(dst), self.stretcher.name, len(result.beats),
                      target, before, after, skipped=False)

