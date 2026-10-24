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


def _collect(paths: list[Path], recursive: bool = True) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            found = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in found if f.is_file() and f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip: {p} not found", file=sys.stderr)
    return files


def _stem_groups(paths: list[Path], suffix: str) -> list[tuple[str, list[Path]]]:
    """Each folder is one stem set (its files, not subfolders); loose files form one set."""
    groups = []
    loose = [p for p in paths if p.is_file()]
    for d in (p for p in paths if p.is_dir()):
        files = [f for f in _collect([d], recursive=False) if suffix not in f.stem]
        if files:
            groups.append((d.name, files))
    if loose:
        files = [f for f in loose if f.suffix.lower() in AUDIO_EXT and suffix not in f.stem]
        if files:
            groups.append((files[0].parent.name, files))
    return groups


def _resolve_reference(name: str, stems: list[Path]) -> Path:
    """Match by file name, by stem name, or by a unique part of the stem name."""
    key = name.lower()
    for test in (lambda f: f.name.lower() == key,
                 lambda f: f.stem.lower() == key,
                 lambda f: key in f.stem.lower()):
        hits = [f for f in stems if test(f)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"reference '{name}' matches several stems: "
                             + ", ".join(f.name for f in hits))
    raise ValueError(f"reference '{name}' matches none of: " + ", ".join(f.name for f in stems))


def _output_path(src: Path, out_dir: Path | None, suffix: str, fmt: str | None) -> Path:
    ext = f".{fmt.lstrip('.')}" if fmt else src.suffix
    return (out_dir or src.parent) / f"{src.stem}{suffix}{ext}"


def _run_one(src: Path, dst: Path, backend: str, options: Options) -> Report:
    return BpmLocker(stretchers.select(backend), options).process(src, dst)


def _run_stems(name: str, srcs: list[Path], dsts: list[Path], reference: Path | None,
               backend: str, options: Options, jobs: int) -> Report:
    return BpmLocker(stretchers.select(backend), options).process_stems(
        srcs, dsts, reference=reference, name=name, jobs=jobs)


def _clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def _format(r: Report) -> str:
    text = _format_tempo(r)
    if r.is_stem_set and not r.skipped:
        text += f"\n  {len(r.outputs)} stems written to {Path(r.outputs[0]).parent}"
    return text


def _format_tempo(r: Report) -> str:
    b = r.before
    f = octave_factor(r.target_bpm)
    title = f"{r.name} ({len(r.inputs)} stems, reference: {r.reference})" if r.is_stem_set \
        else r.name
    line = (f"{title}: {b.detected_bpm * f:.2f} BPM "
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
    p.add_argument("-s", "--stems", action="store_true",
                   help="treat each folder (and all loose files together) as one set of stems: "
                        "analysed once, every stem gets the identical time map")
    p.add_argument("-r", "--reference", metavar="NAME",
                   help="stem that defines the tempo (file name or part of it, e.g. 'drums'); "
                        "implies --stems. Default: the sum of all stems")
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
    if args.stems or args.reference:
        reports, failures = _main_stems(args, backend, options)
    else:
        reports, failures = _main_files(args, backend, options)
    if reports is None:
        return 2

    if args.json:
        args.json.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    return 1 if failures else 0


def _main_files(args, backend: str, options: Options) -> tuple[list[Report] | None, int]:
    jobs = [(f, _output_path(f, args.out_dir, args.suffix, args.format))
            for f in _collect(args.inputs) if args.suffix not in f.stem]
    if not jobs:
        print("error: no audio files found", file=sys.stderr)
        return None, 0
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
    return reports, failures


def _main_stems(args, backend: str, options: Options) -> tuple[list[Report] | None, int]:
    groups = _stem_groups(args.inputs, args.suffix)
    if not groups:
        print("error: no audio files found", file=sys.stderr)
        return None, 0
    reports: list[Report] = []
    failures = 0
    for name, srcs in groups:   # stems within a set are rendered in parallel (--jobs)
        try:
            reference = _resolve_reference(args.reference, srcs) if args.reference else None
            dsts = [_output_path(f, args.out_dir, args.suffix, args.format) for f in srcs]
            r = _run_stems(name, srcs, dsts, reference, backend, options, args.jobs)
            reports.append(r)
            print(_format(r))
        except Exception as e:
            failures += 1
            print(f"{name}: FAILED - {e}", file=sys.stderr)
    return reports, failures


if __name__ == "__main__":
    raise SystemExit(main())
