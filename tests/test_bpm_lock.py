import tempfile
import unittest
from pathlib import Path

import numpy as np

from bpm_lock import audio_io, segmentation, stretchers, tempo_map
from bpm_lock import beats as beats_mod
from bpm_lock.audio_io import Audio
from bpm_lock.beats import DynamicProgrammingBeatTracker
from bpm_lock.pipeline import BpmLocker, Options

from fixtures import SR, bass_onsets, drifting_track, kick_times, render_stems, stepped_track


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


class FillRepair(unittest.TestCase):
    def test_dropped_and_misplaced_beats_in_fill_are_rebuilt(self):
        _, truth = drifting_track(162, duration=60, wobble=0.0)
        noise = np.random.default_rng(3).normal(0, 0.008, len(truth) - 2)  # played feel
        k = int(np.argmax(truth >= 30))
        messy = truth.copy()
        messy[k:k + 10] += 0.06 * np.sin(np.linspace(0, np.pi, 10))   # tracker slips
        messy = np.delete(messy, [k + 3, k + 7]) + noise               # and drops 2 beats
        baseline = np.delete(truth, [0, 1]) + noise                    # same feel, no fill

        def tempo(x):
            repaired, reliable = beats_mod.repair(x, beats_mod.reliable_mask(x))
            return repaired, reliable, 60 / np.diff(beats_mod.robust_smooth(repaired, 9))

        repaired, reliable, t_fill = tempo(messy)
        _, _, t_base = tempo(baseline)
        self.assertEqual(len(repaired), len(truth))                    # dropped beats restored
        self.assertFalse(reliable[k:k + 10].all())
        j = int(np.argmax(repaired >= 30))
        fill = t_fill[j - 5:j + 15]
        self.assertAlmostEqual(fill.mean(), 162, delta=0.5)
        self.assertLessEqual(np.abs(fill - 162).max(), np.abs(t_base - 162).max() + 0.5)

    def test_clean_track_is_left_alone(self):
        _, truth = drifting_track(174, duration=40)
        repaired, reliable = beats_mod.repair(truth, beats_mod.reliable_mask(truth))
        self.assertTrue(reliable.all())
        self.assertTrue(np.array_equal(repaired, truth))


class TimeMap(unittest.TestCase):
    def test_uniform_grid(self):
        beats = np.cumsum(np.full(32, 60 / 173.5)) + 0.2
        tm = tempo_map.build(beats, beats[-1] + 1, segmentation.single(beats, 174))
        self.assertTrue(np.allclose(np.diff(tm.target[1:-1]), 60 / 174))
        self.assertTrue(np.all(np.diff(tm.source) > 0))

    def test_half_time_rounding(self):
        beats = np.arange(64) * 60 / 86.86
        self.assertAlmostEqual(tempo_map.choose_target_bpm(beats, 1.0), 87.0)


class Sections(unittest.TestCase):
    CHANGES = [(0, 150), (20, 170), (40, 160)]

    def test_finds_intentional_tempo_changes(self):
        _, truth = stepped_track(self.CHANGES)
        sections = segmentation.TempoSegmenter(tolerance=3).split(truth)
        self.assertEqual([s.bpm for s in sections], [150, 170, 160])
        for s, (start, _) in zip(sections[1:], self.CHANGES[1:]):
            self.assertLessEqual(abs(int(np.argmax(truth >= start)) - s.start), 1)

    def test_drift_within_tolerance_stays_one_section(self):
        _, truth = drifting_track(174, duration=60)
        sections = segmentation.TempoSegmenter(tolerance=3).split(truth)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].bpm, 174)

    def test_fixed_bpm_applies_to_sections_within_tolerance(self):
        _, truth = stepped_track(self.CHANGES)
        sections = segmentation.TempoSegmenter(tolerance=3, fixed_bpm=152).split(truth)
        self.assertEqual([s.bpm for s in sections], [152, 170, 160])

    def test_short_irregular_passage_is_absorbed(self):
        _, truth = stepped_track([(0, 160)], duration=60)
        messy = truth.copy()
        k = np.argmax(truth >= 30)
        messy[k:k + 12] += np.linspace(0, 0.25, 12)          # 12 beats dragging behind
        messy[k + 12:] += 0.25
        sections = segmentation.TempoSegmenter(tolerance=3).split(messy)
        self.assertEqual(len(sections), 1)

    def test_grid_is_continuous_across_sections(self):
        _, truth = stepped_track(self.CHANGES)
        sections = segmentation.TempoSegmenter(tolerance=3).split(truth)
        tm = tempo_map.build(truth, 60.0, sections)
        for s in sections:
            ibi = np.diff(tm.beat_targets[s.start:s.end + 1])
            self.assertTrue(np.allclose(ibi, 60 / s.bpm))


class Stretchers(unittest.TestCase):
    def _check(self, backend: str):
        y, truth = drifting_track(150, duration=25)
        tm = tempo_map.build(truth, y.shape[1] / SR, segmentation.single(truth, 150))
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

    def test_sections_roundtrip(self):
        y, _ = stepped_track(Sections.CHANGES)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            audio_io.write(src, Audio(y.astype(np.float32), SR))
            report = BpmLocker(stretchers.select("auto"), Options(tolerance=3)).process(src, dst)
            self.assertEqual([s.bpm for s in report.sections], [150, 170, 160])
            for s in report.sections:
                self.assertIsNotNone(s.after)
                self.assertAlmostEqual(s.after.detected_bpm, s.bpm, delta=0.05)
                self.assertLess(s.after.max_drift_ms, 3.0)



class Stems(unittest.TestCase):
    def _make_set(self, tmp: Path) -> list[Path]:
        _, beats = drifting_track(174, duration=40)
        paths = []
        for name, audio in render_stems(beats, 40).items():
            frames = audio.shape[1] - (SR // 10 if name == "pad" else 0)  # pad 0.1 s shorter
            paths.append(tmp / f"{name}.wav")
            audio_io.write(paths[-1], Audio(audio[:, :frames], SR))
        return paths

    def test_stems_share_one_time_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            srcs = self._make_set(Path(tmp))
            dsts = [p.with_name(p.stem + "_locked.wav") for p in srcs]
            report = BpmLocker(stretchers.select("auto"), Options(target_bpm=174)).process_stems(
                srcs, dsts, reference=srcs[0], name="set", jobs=3)
            outs = [audio_io.read(d) for d in dsts]
            self.assertEqual(len({o.frames for o in outs}), 1)        # identical lengths
            self.assertEqual(report.reference, "drums.wav")
            kicks = kick_times(outs[0].mono())
            bass = bass_onsets(outs[1].mono())
            bpm, max_dev = _grid_error_ms(kicks)
            self.assertAlmostEqual(bpm, 174, delta=0.02)
            # every bass note still sits exactly halfway between two drum hits
            mids = (kicks[:-1] + kicks[1:]) / 2
            offset = np.array([np.min(np.abs(mids - b)) for b in bass[2:-2]]) * 1000
            self.assertLess(np.median(offset), 3.0)
            self.assertLess(np.max(offset), 6.0)

    def test_sample_rate_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            srcs = self._make_set(Path(tmp))
            odd = Path(tmp) / "odd.wav"
            a = audio_io.read(srcs[2])
            audio_io.write(odd, Audio(a.samples, 48000))
            with self.assertRaises(ValueError):
                BpmLocker(stretchers.select("auto"), Options()).process_stems(
                    srcs + [odd], [p.with_name(p.stem + "_x.wav") for p in srcs + [odd]])


if __name__ == "__main__":
    unittest.main()
