import os
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

import src.config as config
from src.frame_utils import preprocess_camera_frame

VideoSource = Union[int, str]


def camera_busy_pids(device: str = "/dev/video0") -> List[int]:
    """Return PIDs holding the camera device, if any."""
    if not os.path.exists(device):
        return []
    try:
        result = subprocess.run(
            ["fuser", device],
            capture_output=True,
            text=True,
            check=False,
        )
        line = (result.stdout or "") + (result.stderr or "")
        # fuser prints: /dev/video0:         12345m
        pids = []
        for token in line.replace(device, "").replace(":", " ").split():
            digits = "".join(c for c in token if c.isdigit())
            if digits:
                pids.append(int(digits))
        return pids
    except (FileNotFoundError, ValueError, OSError):
        return []


def _configure_capture(cap: cv2.VideoCapture) -> cv2.VideoCapture:
    if config.CAMERA_MAX_WIDTH > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_MAX_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.CAMERA_MAX_WIDTH * 0.75))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _can_read_frame(cap: cv2.VideoCapture, attempts: int = 8) -> bool:
    if not cap.isOpened():
        return False
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
        time.sleep(0.05)
    return False


def open_video_capture(
    video_source: Optional[VideoSource] = None,
    probe_indices: Tuple[int, ...] = (0, 1, 2),
) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    """
    Open a camera or video file. On Linux, tries V4L2 and probes multiple indices
    when the default index fails (common on laptops with video0 + video1 nodes).

    Returns (capture, label) or (None, None).
    """
    if isinstance(video_source, str) and not video_source.isdigit():
        print(f"[Camera] Opening file: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if _can_read_frame(cap):
            return _configure_capture(cap), str(video_source)
        cap.release()
        return None, None

    if video_source is not None:
        primary = int(video_source)
        indices = [primary] + [i for i in probe_indices if i != primary]
    else:
        indices = list(probe_indices)

    backends = (cv2.CAP_V4L2, cv2.CAP_ANY) if sys.platform.startswith("linux") else (cv2.CAP_ANY,)

    sources: List[Union[int, str]] = []
    for idx in indices:
        sources.append(idx)
        dev_path = f"/dev/video{idx}"
        if os.path.exists(dev_path):
            sources.append(dev_path)

    seen = set()
    for src in sources:
        key = str(src)
        if key in seen:
            continue
        seen.add(key)

        for backend in backends:
            backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "default"
            print(f"[Camera] Trying {src} ({backend_name})...")
            cap = cv2.VideoCapture(src, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(src)
            if _can_read_frame(cap):
                print(f"[Camera] Opened {src} ({backend_name})")
                return _configure_capture(cap), str(src)
            cap.release()

    return None, None


def camera_open_error_message(video_source: Optional[VideoSource] = None) -> str:
    """Human-readable error when no camera could be opened."""
    lines = [
        "Could not open camera.",
        f"  • Requested source: {video_source if video_source is not None else '0 (default)'}",
    ]
    busy = camera_busy_pids("/dev/video0")
    if busy:
        pid_list = ", ".join(str(p) for p in busy)
        lines.append(f"  • Camera is in use by process(es): {pid_list}")
        lines.append(f"    Stop them: kill {busy[0]}")
        lines.append("    (often a previous python main.py that did not exit cleanly)")
    else:
        lines.append("  • Try: python main.py --source 1")
    lines.append("  • Simulation: python main.py --mock")
    lines.append("  • Close browser tabs / Zoom / Cheese using the webcam")
    return "\n".join(lines)


class LatestFrameGrabber:
    """
    Background thread that continuously reads from a camera and keeps only
    the most recent frame. Prevents buffer backlog when inference is slow.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = preprocess_camera_frame(frame)
            with self._lock:
                self._latest = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
