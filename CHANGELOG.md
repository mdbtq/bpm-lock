# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

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

[1.0.0]: https://github.com/mdbtq/bpm-lock/releases/tag/v1.0.0
