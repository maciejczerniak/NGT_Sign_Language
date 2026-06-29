"""
Live camera test for the sign language API.

Captures webcam frames, streams them to /ws/predict, and overlays
the predictions on the video feed in real time.

Requirements:
    pip install opencv-python websockets

Usage:
    python scripts/live_test.py                  # default ws://localhost:8000
    python scripts/live_test.py --url ws://localhost:8000
    python scripts/live_test.py --camera 1       # use a different camera index

Controls (press while the video window is focused):
    Q  — quit
    R  — reset smoother + sequence on the server
    S  — toggle frame sending (pause/resume)
    D  — toggle skeleton debug overlay
"""

import typer
from typing import Any, Optional
import asyncio
import base64
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import websockets
import websockets.exceptions


# ---------------------------------------------------------------------------
# Shared state between the camera thread and the WebSocket thread
# ---------------------------------------------------------------------------


@dataclass
class SharedState:
    """Mutable state shared between the camera capture thread and the WebSocket thread.

    All fields that are read or written from multiple threads are protected by
    their corresponding lock. Boolean control flags are written atomically and
    do not require separate locking.

    Args:
        latest_frame: JPEG-encoded bytes of the most recent captured frame.
            Written by the camera thread; read by the WebSocket thread.
        frame_lock: Lock protecting ``latest_frame``.
        prediction: Most recent prediction response dict received from the server.
            Written by the WebSocket thread; read by the display loop.
        pred_lock: Lock protecting ``prediction``.
        running: Global run flag. Setting to ``False`` stops all threads.
        sending: When ``False``, the WebSocket thread skips frame sending.
        do_reset: When ``True``, the WebSocket thread sends a reset action
            to the server on the next iteration.
        show_skeleton: When ``True``, the landmark skeleton overlay is drawn
            on the video frame.
        latencies: Ring buffer of recent round-trip times in milliseconds,
            used to compute average RTT and estimated FPS.
    """

    latest_frame: Optional[bytes] = None
    frame_lock: threading.Lock = field(default_factory=threading.Lock)

    prediction: dict = field(default_factory=dict)
    pred_lock: threading.Lock = field(default_factory=threading.Lock)

    running: bool = True
    sending: bool = True
    do_reset: bool = False
    show_skeleton: bool = True

    latencies: deque = field(default_factory=lambda: deque(maxlen=30))


STATE = SharedState()


def _reset_state() -> None:
    """Reset the global shared state to a fresh :class:`SharedState` instance.

    Called at the start of :func:`run` to ensure a clean state when the live
    test is restarted within the same process.
    """
    global STATE
    STATE = SharedState()


# ---------------------------------------------------------------------------
# WebSocket thread — runs the async event loop in a background thread
# ---------------------------------------------------------------------------


async def _ws_loop(url: str, attempts: int = 3) -> None:
    """Async WebSocket loop that sends frames and receives predictions.

    Connects to the given WebSocket URL with exponential backoff retries.
    Once connected, runs until ``STATE.running`` is ``False``, handling
    reset requests, frame sending, and prediction receipt on each iteration.

    Args:
        url: WebSocket endpoint URL, e.g. ``ws://localhost:8000/ws/predict``.
        attempts: Maximum number of connection attempts before raising.
            Uses exponential backoff with a cap of 5 seconds between retries.

    Raises:
        OSError: If the connection cannot be established after all attempts.
    """
    print(f"[ws] Connecting to {url} …")
    try:
        ws = None
        for attempt in range(1, attempts + 1):
            try:
                ws = await websockets.connect(url, ping_interval=20)
                break
            except OSError as exc:
                if attempt == attempts:
                    raise
                delay = min(2**attempt, 5)
                print(
                    f"[ws] Could not connect to {url}: {exc}; "
                    f"retrying in {delay}s ({attempt}/{attempts})"
                )
                await asyncio.sleep(delay)

        assert ws is not None
        async with ws:
            print("[ws] Connected.")
            while STATE.running:
                if STATE.do_reset:
                    await ws.send(json.dumps({"action": "reset"}))
                    resp = json.loads(await ws.recv())
                    print(f"[ws] Reset → {resp}")
                    STATE.do_reset = False
                    continue

                if not STATE.sending:
                    await asyncio.sleep(0.05)
                    continue

                with STATE.frame_lock:
                    frame_bytes = STATE.latest_frame

                if frame_bytes is None:
                    await asyncio.sleep(0.01)
                    continue

                b64 = base64.b64encode(frame_bytes).decode()
                t0 = time.perf_counter()
                await ws.send(json.dumps({"image": b64}))

                raw = await ws.recv()
                rtt_ms = (time.perf_counter() - t0) * 1000

                pred = json.loads(raw)
                with STATE.pred_lock:
                    STATE.prediction = pred
                STATE.latencies.append(rtt_ms)

    except websockets.exceptions.ConnectionClosedError as exc:
        print(f"[ws] Connection closed: {exc}")
    except OSError as exc:
        print(f"[ws] Could not connect to {url}: {exc}")
    finally:
        STATE.running = False


def _start_ws_thread(url: str) -> threading.Thread:
    """Start the WebSocket event loop in a background daemon thread.

    Args:
        url: WebSocket endpoint URL passed through to :func:`_ws_loop`.

    Returns:
        The started daemon :class:`~threading.Thread` running the
            async WebSocket loop.
    """

    def _run() -> None:
        asyncio.run(_ws_loop(url))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Skeleton overlay helpers
# ---------------------------------------------------------------------------

_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # palm cross-bar
]

_FINGER_LABELS = {
    0: "WRIST",
    4: "THB",
    8: "IDX",
    12: "MID",
    16: "RNG",
    20: "PNK",
    5: "I-MCP",
    9: "M-MCP",
    13: "R-MCP",
    17: "P-MCP",
}

_SKEL_LINE = (0, 220, 100)
_SKEL_JOINT = (255, 255, 255)
_SKEL_TIP = (0, 255, 255)


def _draw_skeleton(frame: "cv2.Mat", landmarks: list[dict[str, Any]]) -> None:
    """Draw MediaPipe hand skeleton from server-returned landmark list.

    ``landmarks`` is a list of 21 dicts each with keys ``x``, ``y`` (and
    optionally ``z``) normalised to [0, 1]. The server already ran
    MediaPipe during inference — we just re-use the coordinates so we
    don't pay the detection cost twice.

    Args:
        frame: OpenCV BGR frame to draw onto, modified in place.
        landmarks: List of 21 landmark dicts with normalised ``x`` and ``y``
            coordinates. If the list is not exactly 21 elements, drawing is skipped.
    """
    if not landmarks or len(landmarks) != 21:
        return

    h, w = frame.shape[:2]
    pts = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in landmarks]

    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _SKEL_LINE, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        radius = 5 if i in _FINGER_LABELS else 3
        cv2.circle(frame, (x, y), radius, _SKEL_JOINT, -1, cv2.LINE_AA)

    for idx, label in _FINGER_LABELS.items():
        x, y = pts[idx]
        cv2.putText(
            frame,
            label,
            (x + 6, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _SKEL_TIP,
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# OpenCV overlay helpers
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 220, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 60, 220)
_WHITE = (255, 255, 255)
_DARK = (30, 30, 30)


def _text(
    img: Any,
    text: str,
    pos: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Render text with a dark drop shadow for readability on any background.

    Draws the text twice: once offset by one pixel in dark colour as a shadow,
    then again at the original position in the requested colour.

    Args:
        img: OpenCV BGR image to draw onto, modified in place.
        text: The string to render.
        pos: ``(x, y)`` pixel position of the text baseline.
        scale: Font scale factor passed to :func:`cv2.putText`.
        color: BGR text colour as a three-element tuple.
        thickness: Line thickness for the foreground text. The shadow is
            drawn at ``thickness + 1``.
    """
    cv2.putText(
        img,
        text,
        (pos[0] + 1, pos[1] + 1),
        _FONT,
        scale,
        _DARK,
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(img, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_overlay(
    frame: Any,
    pred: dict[str, Any],
    latencies: deque[float],
    sending: bool,
    show_skeleton: bool,
) -> None:
    """Draw the prediction HUD overlay onto the video frame in place.

    Renders the following elements:

    - Optional MediaPipe skeleton if ``show_skeleton`` is ``True`` and
      landmarks are present in the prediction.
    - Semi-transparent bottom bar with raw prediction, stable letter,
      current word, and sentence.
    - Top-right confidence bars for the top-3 predicted letters.
    - Top-left RTT and FPS indicator.
    - Skeleton toggle indicator and paused banner when applicable.
    - Controls hint along the bottom edge.

    Args:
        frame: OpenCV BGR frame to draw onto, modified in place.
        pred: Most recent prediction response dict from the server.
            Expected keys include ``predicted_letter``, ``confidence``,
            ``hand_detected``, ``stable_letter``, ``stable_confidence``,
            ``current_word``, ``sentence``, ``top_3``, and ``landmarks``.
        latencies: Ring buffer of recent round-trip times in milliseconds.
        sending: Whether frame sending is currently active. When ``False``,
            a paused banner is rendered.
        show_skeleton: Whether the skeleton overlay is currently enabled.
    """
    h, w = frame.shape[:2]

    if show_skeleton and pred:
        landmarks = pred.get("landmarks")
        if isinstance(landmarks, list):
            _draw_skeleton(frame, landmarks)

    bar_h = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), _DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if not pred:
        _text(frame, "Waiting for prediction …", (10, h - bar_h + 25), 0.55, _WHITE)
        return

    if "error" in pred:
        _text(frame, f"Error: {pred['error']}", (10, h - bar_h + 25), 0.55, _RED)
        return

    letter = pred.get("predicted_letter") or "—"
    conf = pred.get("confidence", 0.0)
    hand = "hand ✓" if pred.get("hand_detected") else "no hand"
    _text(
        frame,
        f"Raw:  {letter}  ({conf:.0%})  [{hand}]",
        (10, h - bar_h + 22),
        0.58,
        _WHITE,
    )

    stable = pred.get("stable_letter") or "…"
    stable_conf = pred.get("stable_confidence") or 0.0
    color = _GREEN if pred.get("stable_letter") else _YELLOW
    _text(
        frame,
        f"Stable: {stable}  ({stable_conf:.0%})",
        (10, h - bar_h + 48),
        0.65,
        color,
        2,
    )

    word = pred.get("current_word") or ""
    sentence = pred.get("sentence") or ""
    _text(
        frame,
        f"Word: {word}   Sentence: {sentence}",
        (10, h - bar_h + 76),
        0.55,
        _WHITE,
    )

    for i, item in enumerate(pred.get("top_3", [])[:3]):
        bar_w = int(item["confidence"] * 120)
        y = 18 + i * 22
        cv2.rectangle(frame, (w - 135, y - 12), (w - 135 + bar_w, y + 4), _GREEN, -1)
        _text(
            frame,
            f"{item['letter']} {item['confidence']:.0%}",
            (w - 130, y),
            0.45,
            _WHITE,
        )

    if latencies:
        avg_rtt = sum(latencies) / len(latencies)
        fps = 1000 / avg_rtt if avg_rtt > 0 else 0
        _text(frame, f"{avg_rtt:.0f}ms  {fps:.1f}fps", (8, 20), 0.48, _YELLOW)

    skel_label = "skel ON" if show_skeleton else "skel OFF"
    skel_color = _GREEN if show_skeleton else _YELLOW
    _text(frame, skel_label, (w - 75, 20), 0.4, skel_color)

    if not sending:
        _text(
            frame,
            "[ PAUSED — press S to resume ]",
            (w // 2 - 160, h // 2),
            0.7,
            _RED,
            2,
        )

    _text(frame, "Q quit  R reset  S pause  D skeleton", (8, h - 8), 0.4, _WHITE)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(url: str, camera_index: int, target_fps: int) -> None:
    """Open the webcam, start the WebSocket thread, and run the display loop.

    Resets global state, opens the camera at ``camera_index``, starts the
    background WebSocket thread, and enters the main capture-and-display loop.
    Frames are captured, mirrored, JPEG-encoded, and shared with the WebSocket
    thread. The display loop renders the prediction overlay and handles keyboard
    input until the user quits or ``STATE.running`` becomes ``False``.

    Args:
        url: WebSocket endpoint URL to connect to.
        camera_index: OpenCV camera index to open (0 for the default webcam).
        target_fps: Target frame rate for sending frames to the server.
            The loop sleeps to maintain this rate.

    Raises:
        SystemExit: If the camera at ``camera_index`` cannot be opened.
    """
    _reset_state()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[cam] Cannot open camera index {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _start_ws_thread(url)
    frame_interval = 1.0 / target_fps
    print(
        "[cam] Camera open. Press Q to quit, R to reset, S to pause, D to toggle skeleton."
    )

    try:
        while STATE.running:
            t_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("[cam] Frame read failed — retrying …")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with STATE.frame_lock:
                STATE.latest_frame = jpg.tobytes()

            with STATE.pred_lock:
                pred_copy = dict(STATE.prediction)

            _draw_overlay(
                frame, pred_copy, STATE.latencies, STATE.sending, STATE.show_skeleton
            )
            cv2.imshow("Sign Language — Live Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[cam] Quitting …")
                STATE.running = False
            elif key == ord("r"):
                STATE.do_reset = True
                print("[cam] Reset requested.")
            elif key == ord("s"):
                STATE.sending = not STATE.sending
                print(f"[cam] Sending {'resumed' if STATE.sending else 'paused'}.")
            elif key == ord("d"):
                STATE.show_skeleton = not STATE.show_skeleton
                print(
                    f"[cam] Skeleton overlay {'enabled' if STATE.show_skeleton else 'disabled'}."
                )

            elapsed = time.perf_counter() - t_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[cam] Interrupted.")
    finally:
        STATE.running = False
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    url: str = typer.Option(
        "ws://localhost:8000/ws/predict",
        "--url",
        help="WebSocket URL.",
    ),
    camera: int = typer.Option(
        0,
        "--camera",
        help="OpenCV camera index.",
        min=0,
    ),
    fps: int = typer.Option(
        15,
        "--fps",
        help="Target frames per second to send.",
        min=1,
        max=60,
    ),
) -> None:
    """Run the live camera WebSocket prediction test.

    Launches the webcam capture and WebSocket streaming loop. Frames are
    sent to the sign language prediction API at the given URL and predictions
    are overlaid on the live video feed.

    Args:
        url: WebSocket endpoint URL to connect to.
        camera: OpenCV camera index to open. Use ``0`` for the default
            webcam, or a higher index for additional connected cameras.
        fps: Target frames per second to capture and send to the server.
            Higher values increase responsiveness but also network load.
    """
    run(url=url, camera_index=camera, target_fps=fps)


if __name__ == "__main__":
    typer.run(main)
