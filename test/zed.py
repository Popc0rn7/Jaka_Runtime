#!/usr/bin/env python3
"""RGB-only ZED 2i smoke test through Linux V4L2 (no ZED SDK or CUDA).

The ZED 2i exposes a side-by-side UVC video stream under /dev/video*. This
script reads it with OpenCV's V4L2 backend and, by default, takes only the
left RGB image. It does not use pyzed.sl or request depth, tracking,
calibration, or CUDA processing.

Examples:
    v4l2-ctl --list-devices
    python test/zed.py --device /dev/video0
    python test/zed.py --device /dev/video0 --frames 120 --headless
    python test/zed.py --device /dev/video0 --save /tmp/zed_rgb.jpg
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a V4L2 RGB stream from a ZED 2i."
    )
    parser.add_argument(
        "--device",
        default="/dev/video4",
        help="V4L2 device path (default: /dev/video0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1344,
        help="requested side-by-side width (default: 1344)",
    )
    parser.add_argument(
        "--height", type=int, default=376, help="requested height (default: 376)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="requested FPS (default: 30)"
    )
    parser.add_argument(
        "--fourcc",
        default="",
        help="requested pixel format, e.g. UYVY or MJPG; empty keeps device default",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop after this many frames; 0 runs until q/Ctrl-C (default: 0)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="do not open a preview window"
    )
    parser.add_argument(
        "--save", type=Path, help="save the first color frame to this PNG/JPEG path"
    )
    parser.add_argument(
        "--eye",
        choices=("left", "right", "stereo"),
        default="left",
        help="choose one camera from the side-by-side stream (default: left)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print available /dev/video* paths and exit",
    )
    return parser.parse_args()


def load_opencv():
    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "OpenCV is required. Install it with: pip install opencv-python"
        ) from error
    return cv2


def main() -> int:
    args = parse_args()
    if args.list_devices:
        devices = sorted(glob.glob("/dev/video*"))
        print("\n".join(devices) if devices else "No V4L2 video devices found.")
        return 0
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.frames < 0:
        print(
            "Width, height and FPS must be positive; --frames cannot be negative.",
            file=sys.stderr,
        )
        return 2
    if not Path(args.device).exists():
        print(f"V4L2 device does not exist: {args.device}", file=sys.stderr)
        print("Find the ZED node with: v4l2-ctl --list-devices", file=sys.stderr)
        return 1

    cv2 = load_opencv()
    # CAP_V4L2 forces the kernel UVC/V4L2 path rather than any ZED SDK backend.
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        print(
            f"Could not open {args.device}. Is the ZED connected and permitted?",
            file=sys.stderr,
        )
        return 1

    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        if args.fourcc:
            if len(args.fourcc) != 4:
                print(
                    "--fourcc must be exactly four characters, e.g. MJPG.",
                    file=sys.stderr,
                )
                return 2
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        print(
            f"Opened {args.device}: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )
        print(
            f"V4L2 RGB check ({args.eye} view); press q in the preview window to stop."
        )

        frame_count = 0
        saved = False
        start_time = time.monotonic()
        while args.frames == 0 or frame_count < args.frames:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                print("Failed to read a video frame.", file=sys.stderr)
                return 1
            frame_count += 1

            # ZED UVC frames are side-by-side. OpenCV decodes color into BGR;
            # crop one eye to obtain a normal RGB image without SDK processing.
            if args.eye == "stereo":
                output_bgr = frame_bgr
            else:
                width = frame_bgr.shape[1]
                if width % 2:
                    print(
                        f"Expected an even side-by-side width, got {width}.",
                        file=sys.stderr,
                    )
                    return 1
                midpoint = width // 2
                output_bgr = (
                    frame_bgr[:, :midpoint]
                    if args.eye == "left"
                    else frame_bgr[:, midpoint:]
                )

            if args.save and not saved:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), output_bgr):
                    print(f"Could not save frame to {args.save}", file=sys.stderr)
                    return 1
                print(f"Saved color frame: {args.save}")
                saved = True

            if not args.headless:
                cv2.imshow("ZED 2i V4L2 RGB", output_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        elapsed = time.monotonic() - start_time
        measured_fps = frame_count / elapsed if elapsed else 0.0
        print(f"RGB check passed: {frame_count} frame(s), {measured_fps:.1f} FPS")
        return 0
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
