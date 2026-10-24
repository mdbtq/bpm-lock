# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-10-24

### Added
- `--tolerance BPM`: keep intentional tempo changes. The tempo is locked to a
  baseline while it stays within ±BPM; when it stays outside for `--hold`
  beats, a new section with a fresh baseline starts at the refined change point.
- `--hold BEATS` and `--min-section BEATS` to tune section detection; short
  sections (fills, breaks) are merged into the closest neighbour.
- Section table in the output (input/output start, beats, detected, locked and
  re-measured tempo, drift, spread) and a `sections` list in the JSON report.

### Changed
- `tempo_map.build` takes a list of tempo sections instead of one target BPM.

## [1.0.1] - 2026-10-24

### Changed
- Report separates tempo drift (removed) from per-hit timing spread (preserved
  groove/feel). Drift is the 98th percentile of the smoothed tempo curve's
  deviation from a fixed grid, excluding the first/last 8 beats, so a single
  loose hit or an outro no longer reads as residual drift.
- Tempo range is reported from the smoothed tempo curve.

### Fixed
- Silenced scipy `WavFileWarning` on Rubber Band output (extra WAV chunks).

## [1.0.0] - 2026-10-24

### Added
- Beat tracking (onset envelope, duple-meter tempo estimate, dynamic-programming
  tracker) with sub-frame refinement and robust drift smoothing.
- Half-time detections re-checked at double tempo; 2/3-tempo errors suppressed.
- Uniform-grid time map; target tempo from `--bpm` or least-squares fit, rounded.
- Stretch backends: Rubber Band CLI (key-frame map), ffmpeg rubberband filter,
  built-in WSOLA with beat snapping.
- CLI with batch folders, parallel jobs, output format selection, JSON report
  and post-processing verification.

[1.1.0]: https://github.com/mdbtq/bpm-lock/releases/tag/v1.1.0
[1.0.1]: https://github.com/mdbtq/bpm-lock/releases/tag/v1.0.1
[1.0.0]: https://github.com/mdbtq/bpm-lock/releases/tag/v1.0.0
