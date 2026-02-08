"""Extract frames from videos stored in the `videos` subdirectory.

Usage:
    python extract_frames.py VIDEO_FILENAME INTERVAL_SECONDS

The script looks for the video inside the `videos` directory located next to
this script. Extracted frames are written to `videos/<video_stem>/`.
"""
# python extract_frames.py 1A_3.mp4 1
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose a video into still images at a fixed interval."
    )
    parser.add_argument(
        "video_filename",
        type=str,
        help="Name of the video file inside the videos directory (e.g. 1A_1.MP4).",
    )
    parser.add_argument(
        "interval_seconds",
        type=float,
        help="Interval between extracted frames, in seconds (e.g. 0.5 for 2 fps).",
    )
    return parser.parse_args()


def extract_frames(video_filename: str, interval_seconds: float) -> Path:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    base_dir = Path(__file__).resolve().parent
    videos_dir = base_dir / "videos"
    video_path = videos_dir / video_filename

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = videos_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("Unable to determine FPS from video stream.")

    frame_interval = max(1, math.floor(fps * interval_seconds))
    frame_idx = 0
    saved_count = 0

    success, frame = capture.read()
    while success:
        if frame_idx % frame_interval == 0:
            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        frame_idx += 1
        success, frame = capture.read()

    capture.release()
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = extract_frames(args.video_filename, args.interval_seconds)
    print(f"Frames saved to: {output_dir}")


if __name__ == "__main__":
    main()


