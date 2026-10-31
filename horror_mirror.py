"""
Horror Mirror -- Halloween Kinect v2 scare booth.

Modes (keys 1-4):
  1  Shadow Demon  - you become a black void with glowing red eyes
  2  VHS Nightmare  - desaturated, grainy, flickering found-footage
  3  Ghost Trails  - green afterimages trail your movement
  4  Face Warp     - your face pulses and bulges

Jump scare: fires automatically every 20-40 s, or press SPACE to trigger.
  - extreme face bulge + inverted colors + red flash + static + shake

Keys: 1-4 modes, SPACE = scare now, s = save snapshot, q = quit.
"""
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import FrameSourceTypes_Color, FrameSourceTypes_Depth

PROC_W = 960          # processing width (16:9 -> 540)
SCARE_DUR = 0.5       # seconds a jump scare lasts

MODEL_DIR = Path(__file__).parent / "models"
_detector = cv2.FaceDetectorYN.create(
    str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
    "",
    (PROC_W, 540),
    score_threshold=0.55,
    nms_threshold=0.3,
    top_k=5000,
)

MODE_NAMES = {
    "1": "Shadow Demon",
    "2": "VHS Nightmare",
    "3": "Ghost Trails",
    "4": "Face Warp",
}


# ---------------------------------------------------------------- helpers
def grab(kinect):
    """Return (color_bgr, depth_mm) or (None, None)."""
    color = depth = None
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        bgra = frame.reshape(
            (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)
        ).copy()
        color = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()
        depth = frame.reshape(
            (kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)
        ).copy()
    return color, depth


def resize_color(color):
    h, w = color.shape[:2]
    if w != PROC_W:
        return cv2.resize(color, (PROC_W, int(h * PROC_W / w)))
    return color


def depth_resized(depth, h, w):
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def detect_faces(color_bgr):
    """YuNet face detection -> list of dicts: box, eyes, score."""
    h, w = color_bgr.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(color_bgr)
    if faces is None:
        return []
    out = []
    for f in faces:
        out.append({
            "box": (int(f[0]), int(f[1]), int(f[2]), int(f[3])),
            "eyes": [(int(f[4]), int(f[5])), (int(f[6]), int(f[7]))],
            "score": float(f[14]),
        })
    return out


def bulge(img, face_box, amount=0.6):
    """Radially magnify the face region (horror bulge)."""
    h, w = img.shape[:2]
    x, y, fw, fh = face_box
    cx, cy = x + fw / 2.0, y + fh / 2.0
    r = max(fw, fh) * 0.9
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    k = np.clip(1.0 - d / r, 0.0, 1.0)
    mag = 1.0 + amount * (k ** 2)
    src_x = (cx + dx / mag).astype(np.float32)
    src_y = (cy + dy / mag).astype(np.float32)
    return cv2.remap(img, src_x, src_y, cv2.INTER_LINEAR)


def vignette(img, strength=0.6):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
    mask = np.clip(1.0 - strength * (d ** 2), 0.0, 1.0)[..., None]
    return (img.astype(np.float32) * mask).astype(np.uint8)


def add_grain(img, sigma=14):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def scanlines(img, step=3, alpha=0.18):
    out = img.copy()
    out[::step] = (out[::step].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return out


def red_tint(img):
    out = img.astype(np.float32)
    out[..., 1] *= 0.35
    out[..., 2] *= 1.15
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- effects
def shadow_demon(color, depth, faces):
    d = depth_resized(depth, *color.shape[:2])
    mask = ((d > 400) & (d < 3500)).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    grayc = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(grayc, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bg *= 0.30
    bg[:, :, 0] *= 1.30  # cold blue tint

    m = (mask / 255.0)[:, :, None]
    out = (bg * (1.0 - m)).astype(np.uint8)  # person region -> black void

    # glowing eyes on the void (YuNet gives exact eye landmarks)
    if faces:
        face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        eyes = sorted(face["eyes"], key=lambda p: p[0])
        (lx, ly), (rx, ry) = eyes
        eye_dist = max(6, abs(rx - lx))
        r = max(3, eye_dist // 8)
        glow = np.zeros_like(out)
        cv2.circle(glow, (lx, ly), r * 2, (0, 0, 255), -1)
        cv2.circle(glow, (rx, ry), r * 2, (0, 0, 255), -1)
        glow = cv2.GaussianBlur(glow, (0, 0), 15)
        out = cv2.addWeighted(out, 1.0, glow, 1.2, 0)
        cv2.circle(out, (lx, ly), r, (180, 180, 255), -1)
        cv2.circle(out, (rx, ry), r, (180, 180, 255), -1)
    return out


def vhs(color, state):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.35
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = add_grain(out, 16)
    if random.random() < 0.2:
        out = np.clip(out.astype(np.float32) * random.uniform(0.75, 1.25),
                      0, 255).astype(np.uint8)
    out = scanlines(out)
    if random.random() < 0.08:
        M = np.float32([[1, 0, random.randint(-10, 10)], [0, 1, 0]])
        out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]))
    cv2.putText(out, "REC " + time.strftime("%m/%d/%Y %H:%M:%S"),
                (12, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (out.shape[1] - 24, 24), 9, (0, 0, 255), -1)
    return out


def ghost_trails(color, state):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if state.bg is None:
        state.bg = gray
        state.trail = np.zeros_like(gray)
    state.bg = state.bg * 0.97 + gray * 0.03
    diff = cv2.absdiff(gray.astype(np.uint8), state.bg.astype(np.uint8))
    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    state.trail = state.trail * 0.96
    state.trail = np.maximum(state.trail, motion.astype(np.float32))

    dark = (color.astype(np.float32) * 0.6).astype(np.uint8)
    ghost = np.zeros_like(color)
    ghost[..., 1] = np.clip(state.trail * 0.9, 0, 255).astype(np.uint8)
    return cv2.addWeighted(dark, 1.0, ghost, 1.0, 0)


def face_warp(color, faces, t):
    out = color.copy()
    if faces:
        amount = 0.45 + 0.35 * np.sin(t * 3.0)
        for f in faces:
            out = bulge(out, f["box"], amount)
    return vignette(red_tint(out), 0.5)


def scare(color, faces, t):
    out = color.copy()
    if faces:
        face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        out = bulge(out, face["box"], 0.9)
    out = cv2.bitwise_not(out)
    out = red_tint(out)
    shift = int(np.sin(t * 55.0) * 12)
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]))
    noise = np.random.randint(0, 256, out.shape, np.uint8)
    out = cv2.addWeighted(out, 0.7, noise, 0.3, 0)
    return vignette(out, 1.0)


# ---------------------------------------------------------------- main
def main():
    print("Opening Kinect sensor...")
    kinect = PyKinectRuntime.PyKinectRuntime(
        FrameSourceTypes_Color | FrameSourceTypes_Depth)

    state = SimpleNamespace(bg=None, trail=None, static_left=0)
    mode = "1"
    scare_until = 0.0
    next_scare = time.time() + random.uniform(20, 40)

    print("Horror Mirror running. Keys: 1-4 modes, SPACE scare, s save, q quit")
    try:
        while True:
            color, depth = grab(kinect)
            now = time.time()
            out = None
            if color is not None:
                color = resize_color(color)
                faces = detect_faces(color)

                if state.static_left > 0:
                    out = np.random.randint(0, 256, color.shape, np.uint8)
                    state.static_left -= 1
                elif now < scare_until:
                    out = scare(color, faces, now)
                elif mode == "1":
                    out = shadow_demon(color, depth, faces) if depth is not None else color
                elif mode == "2":
                    out = vhs(color, state)
                    if random.random() < 0.008:
                        state.static_left = random.randint(2, 6)
                elif mode == "3":
                    out = ghost_trails(color, state)
                elif mode == "4":
                    out = face_warp(color, faces, now)
                else:
                    out = color

            # auto jump scare
            if now > next_scare and now > scare_until:
                scare_until = now + SCARE_DUR
                next_scare = now + random.uniform(20, 40)

            if out is not None:
                label = "SCARE!" if now < scare_until else f"[{mode}] {MODE_NAMES[mode]}"
                cv2.putText(out, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, "SPACE scare | 1-4 mode | s save | q quit",
                            (12, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow("Horror Mirror", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                scare_until = time.time() + SCARE_DUR
            elif key == ord("s") and out is not None:
                path = f"snapshot_{MODE_NAMES[mode].lower().replace(' ', '_')}.png"
                cv2.imwrite(path, out)
                print("saved", path)
            elif chr(key) in MODE_NAMES:
                mode = chr(key)
                print("mode ->", MODE_NAMES[mode])
    finally:
        cv2.destroyAllWindows()
        kinect.close()


if __name__ == "__main__":
    main()
