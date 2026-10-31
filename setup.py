"""One-time setup for the Haunted Mirror.

- Patches pykinect2 for 64-bit / Python 3.11+ (three fixes).
- Downloads the YuNet face-detector model if missing.

Run with:  uv run python setup.py
"""
import importlib.util
import pathlib
import urllib.request

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def find_pykinect2_dir():
    spec = importlib.util.find_spec("pykinect2")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("pykinect2 is not installed — run: uv pip install pykinect2")
    return pathlib.Path(spec.submodule_search_locations[0])


def patch_pykinect2(pkg):
    v2 = pkg / "PyKinectV2.py"
    s = v2.read_text(encoding="utf-8")
    s = s.replace(
        "assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)",
        "assert sizeof(tagSTATSTG) == 80, sizeof(tagSTATSTG)",
    )
    s = s.replace(
        "from comtypes import _check_version; _check_version('')",
        "# version check disabled (comtypes 1.4.x vs legacy bindings)",
    )
    v2.write_text(s, encoding="utf-8")

    rt = pkg / "PyKinectRuntime.py"
    s = rt.read_text(encoding="utf-8")
    s = s.replace("time.clock()", "time.perf_counter()")
    rt.write_text(s, encoding="utf-8")
    print("patched pykinect2 OK")


def download_yunet():
    dst = pathlib.Path("models") / "face_detection_yunet_2023mar.onnx"
    if dst.exists():
        print("YuNet model already present:", dst)
        return
    dst.parent.mkdir(exist_ok=True)
    print("downloading YuNet model ...")
    urllib.request.urlretrieve(YUNET_URL, dst)
    print("downloaded:", dst)


if __name__ == "__main__":
    patch_pykinect2(find_pykinect2_dir())
    download_yunet()
    print("setup complete")
