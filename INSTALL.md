# Haunted Mirror — install guide

Step-by-step to set up the Kinect v2 "Haunted Mirror" on a fresh Windows 10/11
laptop (same specs). Written to be followed exactly, by an agent or a human.

## 0. Prerequisites

- Windows 10/11 (64-bit)
- A **Kinect v2** sensor ("Kinect for Xbox One") + the **Kinect Adapter for
  Windows** (converts the sensor's proprietary connector to USB 3.0 + 12 V
  power). The adapter's power brick MUST be plugged into the wall.
- A **USB 3.0** port (blue). USB 2.0 (black) will NOT work — the sensor
  enumerates but fails to start with Code 10.
- Internet access.

## 1. Connect the hardware

1. Plug the adapter's power brick into mains.
2. Connect the adapter to the Kinect sensor (proprietary connector).
3. Plug the adapter's USB cable into a **blue USB 3.0** port on the laptop.
   (AMD Ryzen USB 3.1 controllers are fine — confirmed working.)

## 2. Install the Kinect for Windows SDK 2.0 (driver + runtime)

**Required** — it provides the sensor driver (`kinectsensor.inf`) and the
`Kinect20.dll` runtime (installed to `C:\Windows\System32`).

Download (Microsoft, ~276 MB):
```
https://download.microsoft.com/download/F/2/D/F2D1012E-3BC6-49C5-B8B3-5ACFF58AF7B8/KinectSDK-v2.0_1409-Setup.exe
```

Install silently (needs elevation — a UAC prompt will appear):
```
KinectSDK-v2.0_1409-Setup.exe /passive /norestart
```

## 3. Verify the sensor

After install, **unplug and re-plug** the sensor's USB cable. Open Device
Manager and confirm all three appear with **no warning icon**:

- **Kinect sensor devices** → `WDF KinectSensor Interface 0` (OK)
- **Sound, video and game controllers** → `Xbox NUI Sensor` (OK)
- **Universal Serial Bus devices** → `USB Composite Device` (OK)

> If instead you see a single **"USB Composite Device"** with a yellow warning
> (Code 10), you are on a **USB 2.0** port. Move to a blue **USB 3.0** port.

Optional full report (all green = good):
```
C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Tools\ConfigurationVerifier\KinectV2ConfigurationVerifier.exe
```

## 4. Install tools

- **uv** (Python env/package manager):
  `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  (or `pip install uv`)
- **ffmpeg + ffplay** — required for audio (converting mp3/m4a to wav, and the
  background loop). Must be on `PATH`.
  - `winget install Gyan.FFmpeg` (recommended), or
  - download from https://www.gyan.dev/ffmpeg/builds/ and add its `bin` to `PATH`.
  - Verify: `ffmpeg -version` and `ffplay -version`.

## 5. Get the project

```
git clone <repo-url>
cd haunted-mirror
```

These must be present (they are tracked in the repo):

- `haunted_mirror.py`
- `mask.jpg` — the jump-scare mask
- `sounds/The_Iron_Latch.mp3` — background music
- `sounds/sppoky_cat.m4a` — scare sound
- `models/face_detection_yunet_2023mar.onnx` — face detector

## 6. Python environment + patching

```
uv venv --python 3.11
uv pip install numpy opencv-python comtypes pykinect2 pillow
uv run python setup.py
```

`setup.py`:
1. patches `pykinect2` (three fixes needed for 64-bit + Python 3.11+), and
2. downloads the YuNet face model if missing.

## 7. Run it

```
uv run python haunted_mirror.py
```

Controls: `1-4` creatures · `g` ghost · `s` spider · `b` bat · `f` fullscreen ·
`SPACE` random · `l` lightning · `q` quit.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "USB Composite Device" Code 10 | Wrong port — use a blue **USB 3.0** port |
| Sensor opens but no image | Adapter power brick not plugged in / sensor light off |
| No audio | ffmpeg/ffplay not on `PATH` |
| `pykinect2` import error (assertion / `time.clock`) | re-run `uv run python setup.py` |
| Everything slow in HD | Don't raise resolution — it runs at 960×540 and scales up fullscreen (`f`) |

## pykinect2 patches (for reference — `setup.py` applies these)

`pykinect2==0.1.0` is from 2014. It needs three fixes on modern systems:

1. `PyKinectV2.py`: `assert sizeof(tagSTATSTG) == 72` → `== 80` (64-bit size)
2. `PyKinectV2.py`: remove `from comtypes import _check_version; _check_version('')`
   (stale comtypes version gate)
3. `PyKinectRuntime.py`: replace all `time.clock()` → `time.perf_counter()`
   (`time.clock` was removed in Python 3.8)
