#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

_NATURAL_PARTS = re.compile(r"(\d+)")


def _natural_sort_key(path: Path):
    parts = _NATURAL_PARTS.split(path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble PNG frames into a video using OpenCV.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory with frame_*.png files.")
    parser.add_argument("--output", required=True, help="Output video path (e.g., out.mp4).")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second.")
    parser.add_argument(
        "--pattern",
        default="frame_*.png",
        help="Glob pattern for frames (default: frame_*.png).",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is required for video export fallback.", file=sys.stderr)
        return 6

    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    frames = sorted(input_dir.glob(args.pattern), key=_natural_sort_key)
    if not frames:
        print(f"No frames found in {input_dir} with pattern {args.pattern}", file=sys.stderr)
        return 3

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        print(f"Failed to read first frame: {frames[0]}", file=sys.stderr)
        return 4

    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v" if output.suffix.lower() == ".mp4" else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, float(args.fps), (width, height))
    if not writer.isOpened():
        print(f"Failed to open video writer for {output}", file=sys.stderr)
        return 5

    for idx, path in enumerate(frames):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Skipping unreadable frame: {path}", file=sys.stderr)
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        if idx > 0 and idx % 200 == 0:
            print(f"Wrote {idx} frames...")

    writer.release()
    print(f"Wrote video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
