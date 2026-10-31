"""
Halloween Kinect v2 image-processing playground.

Run live (window opens, switch effects with keys 1-4, 's' saves, 'q' quits):
    uv run python halloween.py

Or grab a single processed snapshot (no window):
    uv run python halloween.py --snap

Effects:
    1  Ghost     - depth-based foreground glows pale over a dark world
    2  Sketch    - Canny edges as white chalk lines on black
    3  Thermal   - colorized depth map with edge highlights
    4  Negative  - inverted creepy color
"""
import sys
import time

import cv2
import numpy as np

from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import FrameSourceTypes_Color, FrameSourceTypes_Depth

DISPLAY_WIDTH = 1280


def make_runtime():
    return PyKinectRuntime.PyKinectRuntime(
        FrameSourceTypes_Color | FrameSourceTypes_Depth
    )


def grab(kinect):
    """Return (color_bgr, depth_mm) or (None, None) if no fresh frame."""
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


def depth_resized(depth, h, w):
    """Resize the depth map to the color frame size (rough alignment)."""
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def effect_ghost(color, depth):
    d = depth_resized(depth, *color.shape[:2])
    # Foreground = things between 0.4 m and 3.5 m away (i.e. you!)
    mask = ((d > 400) & (d < 3500)).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    # Dark, desaturated, cold background
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bg *= 0.35
    bg[:, :, 0] *= 1.25  # blue tint

    # Pale, glowing ghost foreground
    fg = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR).astype(np.float32)
    fg = fg * 1.5 + 70.0

    m = (mask / 255.0)[:, :, None]
    out = bg * (1.0 - m) + fg * m
    return np.clip(out, 0, 255).astype(np.uint8)


def effect_sketch(color, depth=None):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # white chalk on black


def effect_thermal(color, depth):
    d = depth.astype(np.float32)
    d[d == 0] = np.nan
    # Normalize to 0.5-8 m range
    lo, hi = 500.0, 8000.0
    d = np.clip(d, lo, hi)
    vis = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis = depth_resized(vis, *color.shape[:2])

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(vis, 0.75, edges_bgr, 0.35, 0)


def effect_negative(color, depth=None):
    return cv2.bitwise_not(color)


EFFECTS = {
    "1": ("Ghost", effect_ghost),
    "2": ("Sketch", effect_sketch),
    "3": ("Thermal", effect_thermal),
    "4": ("Negative", effect_negative),
}


def scale_for_display(img):
    h, w = img.shape[:2]
    if w > DISPLAY_WIDTH:
        s = DISPLAY_WIDTH / w
        return cv2.resize(img, (DISPLAY_WIDTH, int(h * s)))
    return img


def run_live():
    kinect = make_runtime()
    mode = "1"
    print("Halloween Kinect playground. Keys: 1-4 effects, s=save, q=quit")
    try:
        while True:
            color, depth = grab(kinect)
            if color is not None:
                name, fn = EFFECTS[mode]
                if depth is not None or mode in ("2", "4"):
                    out = fn(color, depth)
                else:
                    out = color
                view = scale_for_display(out)
                cv2.putText(view, f"[{mode}] {name}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Halloween Kinect", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and color is not None:
                name, fn = EFFECTS[mode]
                out = fn(color, depth) if depth is not None else color
                path = f"snapshot_{name.lower()}.png"
                cv2.imwrite(path, out)
                print("saved", path)
            elif chr(key) in EFFECTS:
                mode = chr(key)
                print("effect ->", EFFECTS[mode][0])
    finally:
        cv2.destroyAllWindows()
        kinect.close()


def run_snap():
    kinect = make_runtime()
    color = depth = None
    deadline = time.time() + 15
    while time.time() < deadline and color is None:
        color, depth = grab(kinect)
        time.sleep(0.03)
    if color is None:
        print("ERROR: no color frame")
        kinect.close()
        sys.exit(1)
    out = effect_ghost(color, depth) if depth is not None else color
    cv2.imwrite("halloween_ghost.png", out)
    print("saved halloween_ghost.png", out.shape)
    kinect.close()


if __name__ == "__main__":
    if "--snap" in sys.argv:
        run_snap()
    else:
        run_live()
