"""Quick test: grab a color + depth frame from the Kinect v2 and save PNGs."""
import time

import cv2
import numpy as np

from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import FrameSourceTypes_Color, FrameSourceTypes_Depth


def main():
    print("Opening Kinect sensor...")
    kinect = PyKinectRuntime.PyKinectRuntime(
        FrameSourceTypes_Color | FrameSourceTypes_Depth
    )

    color = None
    depth = None
    deadline = time.time() + 15
    while time.time() < deadline and (color is None or depth is None):
        if color is None and kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            color = frame.reshape(
                (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)
            ).copy()
        if depth is None and kinect.has_new_depth_frame():
            frame = kinect.get_last_depth_frame()
            depth = frame.reshape(
                (kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)
            ).copy()
        time.sleep(0.03)

    if color is None:
        print("ERROR: no color frame received")
    else:
        bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        cv2.imwrite("color.png", bgr)
        print(f"Saved color.png   shape={bgr.shape}")

    if depth is None:
        print("ERROR: no depth frame received")
    else:
        # Depth is uint16 millimeters (0 = unknown/far). Map 0..8 m to 0..255.
        vis = np.where(depth == 0, 8000, depth).astype(np.uint16)
        vis = cv2.convertScaleAbs(vis, alpha=255.0 / 8000.0)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imwrite("depth.png", vis)
        valid = depth[depth > 0]
        print(
            f"Saved depth.png   shape={depth.shape}  "
            f"min={valid.min()}mm max={valid.max()}mm"
        )

    kinect.close()
    print("Done.")


if __name__ == "__main__":
    main()
