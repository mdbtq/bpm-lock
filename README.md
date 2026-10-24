# bpm-lock

Locks pre-rendered music to a fixed tempo. It detects every beat, builds a
source→target time map onto a uniform grid, and time-stretches each beat
interval to fit — pitch unchanged. Optionally it keeps intentional tempo
changes, locking each section to its own tempo.

## Requirements

- Python ≥ 3.10 with numpy and scipy (installed automatically)
- `ffmpeg` / `ffprobe` on PATH (decoding/encoding any format)
- Recommended: `rubberband` CLI (best quality). `brew install rubberband`

## Install

```bash
pipx install git+https://github.com/mdbtq/bpm-lock@v1.1.1
```

Or from a local checkout: `pipx install .`

## Usage

```bash
bpm-lock track.wav                         # detect tempo, round to whole BPM
bpm-lock track.wav --bpm 174               # force target tempo (most reliable)
bpm-lock ~/Music/in -o ~/Music/out -j 4    # whole folder, 4 files in parallel
bpm-lock in/ --bpm 174 -f flac --json report.json
bpm-lock track.wav --tolerance 3           # keep intentional tempo changes (sections)
```

Output per file:

```
track.wav: 159.78 BPM (range 153.45-163.88, drift 795.3 ms) -> 155 BPM [rubberband], now 155.00 BPM, drift 13.6 ms (per-hit spread 22.8 ms, kept as-is)
```

- `range` – slowest/fastest local tempo of the smoothed tempo curve.
- `drift` – how far the tempo curve wanders from a fixed grid (98th percentile).
  This is what bpm-lock removes.
- `per-hit spread` – how far individual hits sit from that curve (95th
  percentile): the groove of a played performance plus detection noise on dense
  material. It is deliberately preserved; bpm-lock fixes the tempo, it does not
  quantize.

## Tempo sections (`--tolerance`)

Without `--tolerance` the whole track is locked to one tempo. With
`--tolerance BPM`, a baseline is measured at the start; the tempo is locked to it
as long as it stays within ±BPM. When it stays outside that band for `--hold`
beats, a new section starts at the change point (refined to the exact beat) with
a fresh baseline, checked with the same tolerance. Each section is locked to its
own rounded tempo and the grid is continuous across boundaries.

```
track.wav: 163.03 BPM (range 148.85-170.95, drift 677.6 ms) -> 3 sections [rubberband]
  #      in     out  beats  detected  locked     now   drift  spread
  1  0:00.3  0:00.3     50    150.03     150  150.00  1.6 ms  1.0 ms
  2  0:20.2  0:20.3     56    169.89     170  169.99  0.8 ms  1.5 ms
  3  0:40.0  0:40.0     51    160.03     160  160.00  1.1 ms  1.5 ms
```

- `in` / `out` – section start in the input and in the output file (use `out`
  to place tempo markers in your DAW).
- `detected` – measured tempo of the section; `locked` – tempo it was locked to;
  `now` – re-measured on the output.
- Sections shorter than `--min-section` beats (fills, breaks with an unclear
  beat) are merged into the neighbour with the closest tempo.
- With `--bpm`, sections within the tolerance of that tempo are locked to it;
  others get their own rounded tempo.
- A gradual tempo ramp is split into steps of roughly the tolerance. Use a
  larger tolerance (or none) to treat it as drift instead.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-b, --bpm` | detected | Target tempo. Also narrows detection (±10 %, ±40 % with `--tolerance`) |
| `-t, --tolerance BPM` | off | Keep intentional tempo changes: new section when the tempo leaves ±BPM |
| `--hold BEATS` | `8` | Beats the tempo must stay outside the tolerance before a new section |
| `--min-section BEATS` | `32` | Shorter sections are merged into a neighbour |
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
                         ├─► segmentation           (tempo sections; one section by default)
                         ├─► tempo_map.build        (beats + sections → time map)
                         └─► stretchers.Stretcher   (strategy: rubberband | ffmpeg-rb | wsola)
```

Tracker and stretcher are injectable, so alternative implementations
(e.g. madmom beat tracking) can be plugged in without touching the pipeline.

## Notes

- Tempo detection assumes a duple meter (most electronic and rock music).
  For tracks with swing or an unusual meter, pass `--bpm` and check the output.
- In fills and breaks the beat tracker can misplace or drop beats. Such
  stretches are detected and rebuilt at the surrounding tempo (with the beat
  count re-counted), so they are not sped up or slowed down. Real tempo changes
  are kept.
- Half-time detections (< 100 BPM) are re-checked at double tempo.
  Tempi of 200+ are kept as detected.

## Tests

```bash
cd tests && PYTHONPATH=../src python3 -m unittest -v
```

Synthetic tracks with drifting tempo and intentional tempo changes; each backend
must land every beat within 3 ms of a perfect grid, and section boundaries must
be found within one beat. The Rubber Band test runs only when the CLI is installed.

## License

MIT — see [LICENSE](LICENSE).
