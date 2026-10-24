"""Command line interface: bpm-lock <files/dirs> [options]."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import __version__, stretchers
from .pipeline import BpmLocker, Options, Report
from .tempo_map import octave_factor

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg"}


def _collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip: {p} not found", file=sys.stderr)
    return files


def _output_path(src: Path, out_dir: Path | None, suffix: str, fmt: str | None) -> Path:
    ext = f".{fmt.lstrip('.')}" if fmt else src.suffix
    return (out_dir or src.parent) / f"{src.stem}{suffix}{ext}"


def _run_one(src: Path, dst: Path, backend: str, options: Options) -> Report:
    return BpmLocker(stretchers.select(backend), options).process(src, dst)


def _clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def _format(r: Report) -> str:
    b = r.before
    f = octave_factor(r.target_bpm)
    line = (f"{Path(r.input).name}: {b.detected_bpm * f:.2f} BPM "
            f"(range {b.min_bpm * f:.2f}-{b.max_bpm * f:.2f}, drift {b.max_drift_ms:.1f} ms)")
    if r.skipped:
        return line + " -> already fixed, skipped"
    if len(r.sections) == 1:
        s = r.sections[0]
        line += f" -> {s.bpm * f:g} BPM [{r.backend}]"
        if s.after:
            a = s.after
            line += (f", now {a.detected_bpm * f:.2f} BPM, drift {a.max_drift_ms:.1f} ms"
                     f" (per-hit spread {a.spread_ms:.1f} ms, kept as-is)")
        return line
    return "\n".join([line + f" -> {len(r.sections)} sections [{r.backend}]",
                      *_section_table(r, f)])


def _section_table(r: Report, f: float) -> list[str]:
    rows = [("#", "in", "out", "beats", "detected", "locked", "now", "drift", "spread")]
    for i, s in enumerate(r.sections, 1):
        a = s.after
        rows.append((
            str(i), _clock(s.start_s), _clock(s.out_start_s), str(s.beats),
            f"{s.detected_bpm * f:.2f}", f"{s.bpm * f:g}",
            f"{a.detected_bpm * f:.2f}" if a else "-",
            f"{a.max_drift_ms:.1f} ms" if a else "-",
            f"{a.spread_ms:.1f} ms" if a else "-",
        ))
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    return ["  " + "  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in rows]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bpm-lock",
        description="Lock pre-rendered music to one fixed tempo by beat-wise time-stretching.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("inputs", nargs="+", type=Path, help="audio files and/or folders")
    p.add_argument("-b", "--bpm", type=float, help="target BPM (default: detected, rounded)")
    p.add_argument("--round", type=float, default=1.0, dest="rounding",
                   help="rounding step for detected BPM (default 1; 0 = no rounding)")
    p.add_argument("--hint", type=float, default=120.0,
                   help="tempo hint for detection when --bpm is not given (default 120)")
    p.add_argument("-t", "--tolerance", type=float, metavar="BPM",
                   help="allow intentional tempo changes: lock within ±BPM of a baseline and "
                        "start a new section (new baseline) when the tempo stays outside it")
    p.add_argument("--hold", type=int, default=8, metavar="BEATS",
                   help="beats the tempo must stay outside the tolerance before a new section "
                        "starts (default 8)")
    p.add_argument("--min-section", type=int, default=32, metavar="BEATS",
                   help="shortest allowed section; shorter ones (fills, breaks) are merged "
                        "into a neighbour (default 32)")
    p.add_argument("--smooth", type=int, default=9,
                   help="beats in drift-smoothing window (default 9; 0 = follow every beat)")
    p.add_argument("--skip-below", type=float, default=0.0, metavar="MS",
                   help="leave files untouched when drift is below MS and BPM already matches")
    p.add_argument("-o", "--out-dir", type=Path, help="output folder (default: next to input)")
    p.add_argument("--suffix", default="_locked", help="output filename suffix")
    p.add_argument("-f", "--format", help="output format: wav, flac, aiff, mp3 … (default: same)")
    p.add_argument("--backend", default="auto", choices=["auto", *stretchers.BACKENDS])
    p.add_argument("-j", "--jobs", type=int, default=1, help="files processed in parallel")
    p.add_argument("--no-verify", action="store_true", help="skip re-analysis of the output")
    p.add_argument("--json", type=Path, help="write a JSON report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backend = stretchers.select(args.backend).name
    except stretchers.StretchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    options = Options(target_bpm=args.bpm, rounding=args.rounding or None,
                      bpm_hint=args.hint, smoothing_beats=args.smooth,
                      tolerance=args.tolerance, hold_beats=args.hold,
                      min_section_beats=args.min_section,
                      min_drift_ms=args.skip_below, verify=not args.no_verify)
    jobs = [(f, _output_path(f, args.out_dir, args.suffix, args.format))
            for f in _collect(args.inputs) if args.suffix not in f.stem]
    if not jobs:
        print("error: no audio files found", file=sys.stderr)
        return 2

    reports: list[Report] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(_run_one, s, d, backend, options): s for s, d in jobs}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                r = fut.result()
                reports.append(r)
                print(_format(r))
            except Exception as e:  # report and continue with the batch
                failures += 1
                print(f"{src.name}: FAILED - {e}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
