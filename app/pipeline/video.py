from pathlib import Path

import cv2


def extract_representative_frame(video_path: Path, frame_path: Path) -> None:
    """v1 video support: grab a single frame from the middle of the clip and run
    the rest of the pipeline on it as if it were a photo. Good enough for
    'what happened in this clip' on typical short camera clips; a future
    iteration could sample several frames or use a video-native model instead."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count // 2, 0))
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read any frame from {video_path}")
        cv2.imwrite(str(frame_path), frame)
    finally:
        capture.release()
