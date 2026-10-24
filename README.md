# bpm-lock

Locks pre-rendered music to one fixed tempo. It detects every beat, builds a
source→target time map onto a uniform grid, and time-stretches each beat
segment to fit — pitch unchanged.

## Requirements

- Python ≥ 3.10 with numpy and scipy (installed automatically)
- `ffmpeg` / `ffprobe` on PATH (decoding/encoding any format)
- Recommended: `rubberband` CLI (best quality). `brew install rubberband`

## Install

```bash
pipx install ./bpm-lock        # or: pip install ./bpm-lock
```

## Usage

```bash
bpm-lock track.wav                         # detect tempo, round to whole BPM
bpm-lock track.wav --bpm 174               # force target tempo (most reliable)
bpm-lock ~/Music/in -o ~/Music/out -j 4    # whole folder, 4 files in parallel
bpm-lock in/ --bpm 174 -f flac --json report.json
```

Output per file:

```
track.wav: 173.73 BPM (range 172.39-175.15, drift 88.7 ms) -> 174 BPM [rubberband], residual drift 1.2 ms
```

`drift` is the largest deviation of a beat from a perfect fixed grid before
processing; `residual drift` is re-measured on the output.

| Option | Default | Meaning |
|---|---|---|
| `-b, --bpm` | detected | Target tempo. Also narrows detection to ±10 % |
| `--round` | `1` | Rounding step for the detected tempo (`0` = none) |
| `--hint` | `120` | Detection prior when `--bpm` is not given |
| `--smooth` | `9` | Beats in the drift-smoothing window (`0` = follow every beat) |
| `--skip-below MS` | `0` | Leave files untouched when drift < MS and BPM already matches |
| `-o, --out-dir` | next to input | Output folder |
| `--suffix` | `_locked` | Output filename suffix (inputs containing it are skipped) |
| `-f, --format` | same as input | `wav`, `flac`, `aiff`, `mp3`, `m4a`, `ogg` |
| `--backend` | `auto` | `rubberband`, `ffmpeg-rb`, `wsola` |
| `-j, --jobs` | `1` | Parallel files |
| `--no-verify` | off | Skip re-analysis of the output |
| `--json FILE` | – | Machine-readable report |

## Stretch backends

`auto` picks the first available:

1. **rubberband** – Rubber Band CLI with a key-frame map (R3 `--fine`, R2 fallback). Exact and highest quality.
2. **ffmpeg-rb** – ffmpeg's rubberband filter, tempo switched at every beat.
3. **wsola** – built-in numpy WSOLA; beats are snapped exactly, fine for the small ratios typical here.

## Architecture

```
cli ─► pipeline.BpmLocker ─► audio_io (ffmpeg)
                         ├─► beats.BeatTracker      (protocol; DP tracker by default)
                         ├─► tempo_map.build        (beats → uniform-grid time map)
                         └─► stretchers.Stretcher   (strategy: rubberband | ffmpeg-rb | wsola)
```

Tracker and stretcher are injectable, so alternative implementations
(e.g. madmom beat tracking) can be plugged in without touching the pipeline.

## Notes

- Tempo detection assumes a steady duple meter (most electronic music).
  For tracks with breaks, tempo changes on purpose or swing, pass `--bpm`
  and check the output.
- Half-time detections (< 100 BPM) are re-checked at double tempo.
  Tempi of 200+ are kept as detected.

## Tests

```bash
cd tests && PYTHONPATH=../src python3 -m unittest -v
```

Synthetic tracks with drifting tempo; each backend must land every beat within
3 ms of a perfect grid. The Rubber Band test runs only when the CLI is installed.
