# Haunted Mirror (Kinect v2)

A live Halloween "haunted mirror" for the Microsoft Kinect v2. A visitor steps
up and the screen comes alive: ghosts, wraiths, spiders, bats, lightning,
glitches, a close-up jump-scare mask, and synthesized + provided audio — all
procedural, no faces or photos needed.

## Requirements

- Windows 10/11 (64-bit)
- Kinect v2 + Kinect Adapter for Windows, plugged into a **USB 3.0** port
- Kinect for Windows SDK 2.0
- `uv` and `ffmpeg`/`ffplay` on `PATH`

Full setup: see [INSTALL.md](INSTALL.md).

## Run

```
uv venv --python 3.11
uv pip install numpy opencv-python comtypes pykinect2 pillow
uv run python setup.py        # patch pykinect2 + fetch the face model
uv run python haunted_mirror.py
```

## Controls

`1-4` creatures · `g` ghost · `s` spider · `b` bat · `f` fullscreen ·
`SPACE` random · `l` lightning · `q` quit

## Files

- `haunted_mirror.py` — the main scare booth
- `halloween.py`, `horror_mirror.py` — earlier playground iterations
- `test_kinect.py` — simple color+depth grabber (diagnostic)
- `setup.py` — pykinect2 patching + model download
- `mask.jpg` — jump-scare mask overlay
- `sounds/` — background (`The_Iron_Latch.mp3`), scare (`sppoky_cat.m4a`)
- `models/` — YuNet face detector
