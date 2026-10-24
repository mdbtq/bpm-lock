import tempfile
import unittest
from pathlib import Path

import numpy as np

from bpm_lock import audio_io, stretchers, tempo_map
from bpm_lock.audio_io import Audio
from bpm_lock.beats import DynamicProgrammingBeatTracker
from bpm_lock.pipeline import BpmLocker, Options

from fixtures import SR, drifting_track, kick_times


def _grid_error_ms(times: np.ndarray) -> tuple[float, float]:
    idx = np.arange(len(times))
    slope, offset = np.polyfit(idx, times, 1)
    return 60 / slope, float(np.max(np.abs(times - (slope * idx + offset))) * 1000)


class TempoDetection(unittest.TestCase):
    def test_detects_tempo_without_hint(self):
        for bpm in (128, 140, 174, 190):
            with self.subTest(bpm=bpm):
                y, _ = drifting_track(bpm)
                r = DynamicProgrammingBeatTracker().track(y.mean(0), SR)
                self.assertAlmostEqual(r.tempo, bpm, delta=bpm * 0.02)

    def test_beats_match_ground_truth(self):
        y, truth = drifting_track(174)
        r = DynamicProgrammingBeatTracker().track(y.mean(0), SR)
        self.assertEqual(len(r.beats), len(truth))
        err = (r.beats - truth) * 1000
        self.assertLess(np.std(err), 2.0)   # constant latency is fine; jitter is not


class TimeMap(unittest.TestCase):
    def test_uniform_grid(self):
        beats = np.cumsum(np.full(32, 60 / 173.5)) + 0.2
        tm = tempo_map.build(beats, beats[-1] + 1, 174)
        self.assertTrue(np.allclose(np.diff(tm.target[1:-1]), 60 / 174))
        self.assertTrue(np.all(np.diff(tm.source) > 0))

    def test_half_time_rounding(self):
        beats = np.arange(64) * 60 / 86.86
        self.assertAlmostEqual(tempo_map.choose_target_bpm(beats, 1.0), 87.0)


class Stretchers(unittest.TestCase):
    def _check(self, backend: str):
        y, truth = drifting_track(150, duration=25)
        tm = tempo_map.build(truth, y.shape[1] / SR, 150)
        out = stretchers.select(backend).stretch(Audio(y.astype(np.float32), SR), tm)
        self.assertEqual(out.frames, int(round(tm.output_duration * SR)))
        bpm, max_dev = _grid_error_ms(kick_times(out.mono()))
        self.assertAlmostEqual(bpm, 150, delta=0.02)
        self.assertLess(max_dev, 3.0)

    def test_wsola(self):
        self._check("wsola")

    @unittest.skipUnless(stretchers.FfmpegRubberbandStretcher.available(), "no ffmpeg rubberband")
    def test_ffmpeg_rubberband(self):
        self._check("ffmpeg-rb")

    @unittest.skipUnless(stretchers.RubberbandCliStretcher.available(), "no rubberband CLI")
    def test_rubberband_cli(self):
        self._check("rubberband")


class EndToEnd(unittest.TestCase):
    def test_file_roundtrip(self):
        y, _ = drifting_track(174, duration=40)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.flac"
            audio_io.write(src, Audio(y.astype(np.float32), SR))
            report = BpmLocker(stretchers.select("auto"), Options()).process(src, dst)
            self.assertEqual(report.target_bpm, 174)
            out = audio_io.read(dst)
            bpm, max_dev = _grid_error_ms(kick_times(out.mono(), out.sample_rate))
            self.assertAlmostEqual(bpm, 174, delta=0.02)
            self.assertLess(max_dev, 3.0)


if __name__ == "__main__":
    unittest.main()
