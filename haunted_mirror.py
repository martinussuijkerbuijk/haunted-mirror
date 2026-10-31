"""
The Haunting -- a live haunted mirror for the Kinect v2.

A visitor stands in front of the mirror. The scene goes dark and foggy while
creatures materialize around them. Creatures render in two layers:

  FRONT (always visible, fly around):
    Ghost    - a translucent sheet ghost with hollow eyes drifts across the frame
    Spirit   - the visitor's own silhouette detaches and floats up, fading
    Eyes     - pairs of glowing eyes blink open in the darkness
    Shoulder - a pair of eyes peeks over the top of the visitor's head

  BEHIND (occluded correctly by the body):
    Wraith   - a tall dark hooded figure with big red eyes looms up behind you

Ambient: drifting fog, random lightning (flash -> near-dark beat), desaturation,
vignette, and a faint glowing outline around the visitor.

No faces, no photos -- everything is depth-silhouette and motion based.

Keys: 1 wraith, 2 spirit, 3 eyes, 4 shoulder, g ghost, SPACE random,
      l lightning, q quit.
"""
import math
import random
import subprocess
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import winsound
from PIL import Image, ImageDraw, ImageFont

from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import FrameSourceTypes_Color, FrameSourceTypes_Depth

PROC_W, PROC_H = 960, 540
DEPTH_NEAR, DEPTH_FAR = 400, 3500   # mm range that counts as "the visitor"

# --- creature tuning ------------------------------------------------------
WRAITH_DUR = 4.0
SPIRIT_DUR = 4.0
EYES_DUR = 1.8
SHOULDER_DUR = 2.2
GHOST_DUR = 6.0
SPIDER_DUR = 6.0
CREATURE_MIN, CREATURE_MAX = 10.0, 16.0   # seconds between scares
GHOST_MIN, GHOST_MAX = 4.0, 7.0           # seconds between flying ghosts
SPIDER_MIN, SPIDER_MAX = 2.5, 4.5         # seconds between spiders
BAT_DUR = 5.0
BAT_MIN, BAT_MAX = 3.0, 6.0               # seconds between bats
LIGHTNING_MIN, LIGHTNING_MAX = 6.0, 15.0
FACE_CLOSE_MM = 800            # face within this distance triggers the mask scare
MASK_SCARE_DUR = 2.0           # seconds the scare takes over the screen
MASK_CD_MIN, MASK_CD_MAX = 8.0, 15.0   # cooldown between mask scares

# --- face detector + scary mask overlay -----------------------------------
MODEL_DIR = Path(__file__).parent / "models"
_face_det = cv2.FaceDetectorYN.create(
    str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"), "",
    (PROC_W, PROC_H), score_threshold=0.55, nms_threshold=0.3, top_k=5000)

_scary_mask = None
_scary_alpha = None
_mask_path = Path(__file__).parent / "mask.jpg"
if _mask_path.exists():
    img = cv2.imread(str(_mask_path))
    if img is not None:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(g > 10)
        if len(xs):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        _scary_mask = img
        ga = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        _scary_alpha = np.clip((ga - 12.0) / 50.0, 0.0, 1.0)
else:
    print("WARNING: mask.jpg not found; mask scare disabled")

# --- decorative font (Chiller) --------------------------------------------
SCARY_FONT_PATH = r"C:\Windows\Fonts\CHILLER.TTF"
_font_cache = {}


def _get_font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(SCARY_FONT_PATH, size)
        except Exception:
            _font_cache[size] = None
    return _font_cache[size]


def draw_texts_scary(img, items, color_rgb=(190, 25, 35), shadow_rgb=(12, 2, 6)):
    if _get_font(72) is None:
        for text, center, size in items:
            cv2.putText(img, text, center, cv2.FONT_HERSHEY_SIMPLEX, size / 60.0,
                        (color_rgb[2], color_rgb[1], color_rgb[0]), 3, cv2.LINE_AA)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, center, size in items:
        font = _get_font(size)
        draw.text((center[0] + 3, center[1] + 4), text, font=font,
                  fill=shadow_rgb, anchor="mm")
        draw.text(center, text, font=font, fill=color_rgb, anchor="mm")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# --- audio (synthesized scare sting + thunder) ----------------------------
def _write_wav(path, samples, rate):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _synthesize_sounds():
    sounds_dir = Path(__file__).parent / "sounds"
    sounds_dir.mkdir(exist_ok=True)
    rate = 44100

    tpath = sounds_dir / "thunder.wav"
    if not tpath.exists():
        dur = 2.2
        n = int(rate * dur)
        rng = np.random.default_rng(7)
        brown = np.cumsum(rng.standard_normal(n))
        brown /= (np.max(np.abs(brown)) + 1e-6)
        rumble = np.convolve(brown, np.ones(60) / 60, mode="same")
        t = np.arange(n) / rate
        low = 0.5 * np.sin(2 * np.pi * 42 * t) + 0.3 * np.sin(2 * np.pi * 58 * t)
        s = (rumble * 0.7 + low) * np.exp(-t * 2.2)
        s = s / (np.max(np.abs(s)) + 1e-6) * 0.85
        _write_wav(tpath, s, rate)

    spath = sounds_dir / "sting.wav"
    if not spath.exists():
        dur = 1.4
        n = int(rate * dur)
        t = np.arange(n) / rate
        rng = np.random.default_rng(11)
        f_inst = 750 * np.exp(-t * 6.0) + 180.0
        phase = 2 * np.pi * np.cumsum(f_inst) / rate
        screech = np.sin(phase) * np.exp(-t * 2.2)
        boom = 0.7 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 5.0)
        noise = rng.standard_normal(n) * 0.35 * np.exp(-t * 4.0)
        s = 0.55 * screech + boom + noise
        s = s / (np.max(np.abs(s)) + 1e-6) * 0.9
        _write_wav(spath, s, rate)

    return str(tpath), str(spath)


THUNDER_WAV, STING_WAV = _synthesize_sounds()

# --- convert user-provided sounds (mp3/m4a) to wav for winsound -----------
SOUNDS_DIR = Path(__file__).parent / "sounds"


def _run_ffmpeg(args):
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args,
                           capture_output=True, timeout=180)
        return r.returncode == 0
    except Exception:
        return False


def _convert_audio(src_name, dst_name, max_seconds=None, volume=None):
    dst = SOUNDS_DIR / dst_name
    if dst.exists():
        return str(dst)
    src = SOUNDS_DIR / src_name
    if not src.exists():
        return None
    args = ["-i", str(src)]
    if max_seconds:
        args += ["-t", str(max_seconds)]
    if volume:
        args += ["-af", f"volume={volume}"]
    args += ["-ac", "1", "-ar", "44100", str(dst)]
    if _run_ffmpeg(args) and dst.exists():
        return str(dst)
    return None


BACKGROUND_WAV = _convert_audio("The_Iron_Latch.mp3", "background.wav", 60)
SCARE_WAV = _convert_audio("sppoky_cat.m4a", "scare.wav", volume=0.6)


def play_thunder():
    try:
        winsound.PlaySound(THUNDER_WAV, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


def play_scare():
    path = SCARE_WAV or STING_WAV
    if path:
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass


def start_background():
    """Loop the 60s background track in a separate ffplay process."""
    if not BACKGROUND_WAV:
        return None
    try:
        return subprocess.Popen(
            ["ffplay", "-nodisp", "-loop", "0", "-volume", "70",
             "-loglevel", "quiet", BACKGROUND_WAV],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None


# ---------------------------------------------------------------- frames
def grab(kinect):
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


# ---------------------------------------------------------------- person
def person_mask(depth):
    d = depth_resized(depth, PROC_H, PROC_W)
    mask = ((d > DEPTH_NEAR) & (d < DEPTH_FAR)).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return cv2.GaussianBlur(mask, (21, 21), 0)


def person_present(mask):
    return (mask > 100).sum() > 4000


def head_position(mask):
    ys, xs = np.where(mask > 100)
    if len(xs) < 500:
        return None
    top = int(ys.min())
    band = ys < top + 40
    x = int(np.median(xs[band]))
    return (x, top)


def person_distance(depth_r, mask):
    if depth_r is None or mask is None:
        return None
    region = depth_r[mask > 100]
    valid = region[(region > 200) & (region < 8000)]
    if len(valid) < 100:
        return None
    return float(np.median(valid))


# ---------------------------------------------------------------- scene
def moody_background(color, dark=0.38, blue=1.30):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bg *= dark
    bg[:, :, 0] *= blue
    return np.clip(bg, 0, 255).astype(np.uint8)


def make_fog_field(h, w):
    rng = np.random.default_rng(1234)
    small = rng.random((h // 6, w // 6)).astype(np.float32)
    f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (0, 0), 40)
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    return f


def apply_fog(bg, fog, t, strength=55):
    sx = int(30 * math.sin(t * 0.25))
    sy = int(15 * math.cos(t * 0.15))
    f = np.roll(np.roll(fog, sx, axis=1), sy, axis=0)
    add = (f * strength)[..., None].astype(np.float32)
    return np.clip(bg.astype(np.float32) + add, 0, 255).astype(np.uint8)


def make_cobweb_layer(h, w):
    layer = np.zeros((h, w, 3), np.uint8)
    color = (175, 185, 200)
    R = int(min(h, w) * 0.32)
    for cx, cy, a0, a1 in [(0, 0, 0.0, math.pi / 2),
                           (w - 1, 0, math.pi / 2, math.pi)]:
        for i in range(7):
            a = a0 + (a1 - a0) * i / 6.0
            ex = int(cx + math.cos(a) * R)
            ey = int(cy + math.sin(a) * R)
            cv2.line(layer, (cx, cy), (ex, ey), color, 1)
        for r in range(30, R, 30):
            pts = []
            for i in range(11):
                a = a0 + (a1 - a0) * i / 10.0
                pts.append((int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)))
            cv2.polylines(layer, [np.array(pts, np.int32)], False, color, 1)
    return layer


def vignette(img, strength=0.45):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
    m = np.clip(1.0 - strength * (d ** 2), 0.0, 1.0)[..., None]
    return (img.astype(np.float32) * m).astype(np.uint8)


def spooky_flicker(img, now):
    """Global unstable-light flicker: candle jitter, dark dips, tint pulses."""
    out = img.astype(np.float32)
    flick = 1.0 + 0.09 * math.sin(now * 9.3) + 0.07 * math.sin(now * 17.7 + 1.1)
    out *= flick
    if random.random() < 0.03:
        out *= random.uniform(0.25, 0.6)          # light sputters
    if random.random() < 0.025:
        out[..., 0] *= 0.55                       # red pulse
        out[..., 2] *= 1.4
    if random.random() < 0.012:
        out[..., 0] *= 0.75                       # sickly green pulse
        out[..., 1] *= 1.3
    return np.clip(out, 0, 255).astype(np.uint8)


def glitch(img):
    """Occasional chromatic aberration, horizontal tear, static burst."""
    if random.random() < 0.10:
        sh = random.randint(3, 9)
        out = img.copy()
        out[..., 2] = np.roll(img[..., 2], -sh, axis=1)
        out[..., 0] = np.roll(img[..., 0], sh, axis=1)
        img = out
    if random.random() < 0.07:
        h, w = img.shape[:2]
        y0 = random.randint(0, max(1, h - 40))
        bh = random.randint(8, 30)
        dx = random.randint(-50, 50)
        band = img[y0:y0 + bh].copy()
        img[y0:y0 + bh] = np.roll(band, dx, axis=1)
    if random.random() < 0.03:
        h, w = img.shape[:2]
        y0 = random.randint(0, max(1, h - 50))
        bh = random.randint(10, 50)
        img[y0:y0 + bh] = np.random.randint(0, 256, (bh, w, 3), np.uint8)
    return img


def shake(img, now, amp=12):
    h, w = img.shape[:2]
    dx = int(math.sin(now * 80.0) * amp)
    dy = int(math.cos(now * 65.0) * amp * 0.6)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h))


def glow_outline(mask, color=(180, 230, 255)):
    m = (mask > 100).astype(np.uint8) * 255
    edge = cv2.dilate(m, np.ones((5, 5), np.uint8)) - \
           cv2.erode(m, np.ones((5, 5), np.uint8))
    edge = cv2.GaussianBlur(edge, (0, 0), 6)
    layer = np.zeros((*edge.shape, 3), np.uint8)
    for c in range(3):
        layer[..., c] = (edge.astype(np.float32) / 255.0 * color[c]).astype(np.uint8)
    return layer


def composite_person(bg, color, mask):
    p = color.astype(np.float32) * 0.9
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = bg.astype(np.float32) * (1.0 - a) + p * a
    return np.clip(out, 0, 255).astype(np.uint8)


def sample_eye_points(n, h=PROC_H, w=PROC_W):
    return [(random.randint(50, w - 50), random.randint(50, h - 50)) for _ in range(n)]


def detect_faces(color_bgr):
    h, w = color_bgr.shape[:2]
    _face_det.setInputSize((w, h))
    _, faces = _face_det.detect(color_bgr)
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


def paste_with_alpha(base, overlay_bgr, alpha, x0, y0):
    H, W = base.shape[:2]
    h, w = overlay_bgr.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return base
    sx, sy = x0c - x0, y0c - y0
    ow, oh = x1c - x0c, y1c - y0c
    ov = overlay_bgr[sy:sy + oh, sx:sx + ow].astype(np.float32)
    al = alpha[sy:sy + oh, sx:sx + ow][..., None]
    base[y0c:y1c, x0c:x1c] = np.clip(
        base[y0c:y1c, x0c:x1c].astype(np.float32) * (1.0 - al) + ov * al,
        0, 255).astype(np.uint8)
    return base


def face_distance(depth_r, faces):
    if depth_r is None or not faces:
        return None
    face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
    x, y, w, h = face["box"]
    cx, cy = x + w // 2, y + h // 2
    r = max(4, w // 8)
    y0, y1 = max(0, cy - r), min(depth_r.shape[0], cy + r)
    x0, x1 = max(0, cx - r), min(depth_r.shape[1], cx + r)
    region = depth_r[y0:y1, x0:x1]
    valid = region[(region > 200) & (region < 8000)]
    if len(valid) < 10:
        return None
    return face, float(np.median(valid))


def render_mask_scare(state, now):
    H, W = PROC_H, PROC_W
    dt = now - state.mask_scare_t0
    p = dt / MASK_SCARE_DUR
    if dt < 0.10:
        return np.full((H, W, 3), 255, np.uint8)   # lightning flash

    # pulsing dark-red ember background
    pulse = 0.5 + 0.5 * math.sin(now * 8.0)
    out = np.zeros((H, W, 3), np.uint8)
    out[..., 0] = int(12 + 18 * pulse)
    out[..., 1] = 2
    out[..., 2] = int(22 + 45 * pulse)

    # ring of flickering lights surrounding the mask
    n_orb = 12
    ring_r = int(min(H, W) * 0.47)
    glow = np.zeros((H, W, 3), np.uint8)
    for i in range(n_orb):
        a = i / n_orb * 2 * math.pi
        ox = int(W / 2 + math.cos(a) * ring_r)
        oy = int(H / 2 + math.sin(a) * ring_r)
        b = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(now * (10 + (i % 4) * 4.5) + i * 2.3))
        cv2.circle(glow, (ox, oy), 26, (int(30 * b), int(110 * b), int(255 * b)), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 24)
    out = np.clip(out.astype(np.float32) + glow.astype(np.float32) * 0.9,
                  0, 255).astype(np.uint8)

    # mask: emerges, flickers, shakes
    if _scary_mask is not None:
        scale = 0.25 + 0.85 * min(1.0, p * 3.0)
        if math.sin(now * 38.0) > -0.35:
            mh, mw = _scary_mask.shape[:2]
            th = int(H * scale)
            tw = int(th * mw / mh)
            ov = cv2.resize(_scary_mask, (tw, th))
            al = cv2.resize(_scary_alpha, (tw, th))
            fade = 1.0 if p < 0.85 else max(0.0, (1.0 - p) / 0.15)
            shx = int(math.sin(now * 55.0) * 7)
            shy = int(math.cos(now * 43.0) * 5)
            out = paste_with_alpha(out, ov, al * fade,
                                   (W - tw) // 2 + shx, (H - th) // 2 + shy)

    return glitch(out)


# ---------------------------------------------------------------- creatures
def _fade(p, in_p, out_p):
    return max(0.0, min(1.0, p / max(in_p, 1e-6))) * \
           max(0.0, min(1.0, (1.0 - p) / max(out_p, 1e-6)))


def draw_wraith(bg, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return bg
    a = _fade(p, 0.12, 0.18)
    s = 0.35 + 1.0 * (p ** 1.3)
    cx, cy = ev["pos"]
    cy = max(60, cy - int(170 * s))
    H, W = bg.shape[:2]

    bw, bh = int(110 * s), int(260 * s)
    shape = np.zeros((H, W), np.float32)
    cv2.ellipse(shape, (cx, cy), (int(bw * 0.5), int(bh * 0.42)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(shape, (cx, cy + int(bh * 0.55)), (int(bw * 0.9), int(bh * 0.55)),
                0, 0, 360, 1.0, -1)
    shape = cv2.GaussianBlur(shape, (0, 0), 18)
    out = bg.astype(np.float32) * (1.0 - shape[..., None] * 0.9 * a)

    # big bright glowing eyes
    er = max(5, int(9 * s))
    ex, ey = int(bw * 0.24), cy - int(bh * 0.04)
    glow = np.zeros((H, W, 3), np.uint8)
    cv2.circle(glow, (cx - ex, ey), er * 3, (0, 0, 255), -1)
    cv2.circle(glow, (cx + ex, ey), er * 3, (0, 0, 255), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 25)
    out += glow.astype(np.float32) * 2.0 * a
    core = np.zeros((H, W, 3), np.uint8)
    cv2.circle(core, (cx - ex, ey), er, (120, 120, 255), -1)
    cv2.circle(core, (cx + ex, ey), er, (120, 120, 255), -1)
    out += core.astype(np.float32) * 1.2 * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_spirit(out, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return out
    a = math.sin(math.pi * p)
    dy, dx = int(-220 * p), int(ev["drift"] * p)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    m = cv2.warpAffine(ev["mask"], M, (ev["mask"].shape[1], ev["mask"].shape[0]))
    layer = np.zeros((*m.shape, 3), np.uint8)
    layer[..., 0] = (m * 0.50).astype(np.uint8)
    layer[..., 1] = (m * 0.95).astype(np.uint8)
    layer[..., 2] = (m * 0.90).astype(np.uint8)
    layer = cv2.GaussianBlur(layer, (0, 0), 12)
    return np.clip(out.astype(np.float32) + layer.astype(np.float32) * a,
                   0, 255).astype(np.uint8)


def draw_eyes(out, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return out
    a = _fade(p, 0.25, 0.25)
    H, W = out.shape[:2]
    er = ev["er"]
    glow = np.zeros((H, W, 3), np.uint8)
    for (x, y) in ev["eyes"]:
        cv2.circle(glow, (x - er, y), er, (0, 160, 255), -1)
        cv2.circle(glow, (x + er, y), er, (0, 160, 255), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    out = np.clip(out.astype(np.float32) + glow.astype(np.float32) * 1.6 * a, 0, 255)
    core = np.zeros((H, W, 3), np.uint8)
    for (x, y) in ev["eyes"]:
        cv2.circle(core, (x - er, y), max(2, er // 2), (120, 220, 255), -1)
        cv2.circle(core, (x + er, y), max(2, er // 2), (120, 220, 255), -1)
    out += core.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_shoulder_eyes(out, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return out
    a = _fade(p, 0.2, 0.25)
    x, y = ev["pos"]
    H, W = out.shape[:2]
    er = 8
    glow = np.zeros((H, W, 3), np.uint8)
    cv2.circle(glow, (x - er, y), er * 2, (0, 0, 255), -1)
    cv2.circle(glow, (x + er, y), er * 2, (0, 0, 255), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 20)
    out = np.clip(out.astype(np.float32) + glow.astype(np.float32) * 2.0 * a, 0, 255)
    core = np.zeros((H, W, 3), np.uint8)
    cv2.circle(core, (x - er, y), er, (200, 200, 255), -1)
    cv2.circle(core, (x + er, y), er, (200, 200, 255), -1)
    out += core.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_ghost(out, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return out
    a = math.sin(math.pi * p) * ev["alpha"]
    x0, y0 = ev["start"]
    dx, dy = ev["vel"]
    x = int(x0 + dx * p)
    y = int(y0 + dy * p + ev["wob"] * math.sin(p * math.pi * ev["wobf"]))
    w, h = ev["size"]
    H, W = out.shape[:2]

    ghost = np.zeros((H, W, 3), np.uint8)
    hr = w // 2
    head_cy = y - h // 2 + hr
    cv2.rectangle(ghost, (x - hr, head_cy), (x + hr, y + h // 2), (225, 230, 250), -1)
    cv2.circle(ghost, (x, head_cy), hr, (235, 240, 255), -1)
    nb = 5
    for i in range(nb):
        sx = x - hr + int((i + 0.5) * (2 * hr) / nb)
        cv2.circle(ghost, (sx, y + h // 2), hr // nb + 3, (225, 230, 250), -1)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 14)

    er = max(3, w // 9)
    eyes = np.zeros((H, W, 3), np.uint8)
    ey = head_cy - hr // 3
    cv2.circle(eyes, (x - w // 6, ey), er, (15, 15, 25), -1)
    cv2.circle(eyes, (x + w // 6, ey), er, (15, 15, 25), -1)

    out = out.astype(np.float32) + ghost.astype(np.float32) * a + eyes.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bat(out, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return out
    x = int(ev["x0"] + ev["dx"] * p)
    y = int(ev["y0"] + ev["dy"] * p + ev["wob"] * math.sin(p * math.pi * ev["wobf"]))
    s = ev["size"]
    dark = (30, 30, 36)
    flap = math.sin(now * ev["flap"]) * 0.9
    cv2.ellipse(out, (x, y), (max(2, s // 6), max(2, s // 9)), 0, 0, 360, dark, -1)
    for side in (1, -1):
        topx = x + side * int(s * 0.15)
        topy = y - int(s * 0.5)
        tipx = x + side * int(s * 0.85)
        tipy = y + int(s * 0.5 * flap)
        pts = np.array([[x, y], [topx, topy], [tipx, tipy]], np.int32)
        cv2.fillPoly(out, pts, dark)
    er = max(1, s // 10)
    cv2.circle(out, (x - max(2, s // 8), y), er, (0, 0, 255), -1)
    cv2.circle(out, (x + max(2, s // 8), y), er, (0, 0, 255), -1)
    return out


def _spider_pos(ev, p):
    drop_end = 0.30
    if p < drop_end:
        k = p / drop_end
        x = ev["x0"] + math.sin(p * 22.0) * 3.0
        y = ev["drop_y"] * k
        thread = True
    else:
        k = (p - drop_end) / (1.0 - drop_end)
        x = ev["x0"] + (ev["x1"] - ev["x0"]) * k
        y = ev["drop_y"] + (ev["y1"] - ev["drop_y"]) * k
        thread = False
    return x, y, thread


def draw_spider(out, ev, now):
    p = (now - ev["t0"]) / ev["dur"]
    if p < 0 or p > 1:
        return out
    x, y, thread = _spider_pos(ev, p)
    ix, iy = int(x), int(y)
    s = ev["size"]
    dark = (45, 40, 38)
    if thread:
        cv2.line(out, (int(ev["x0"]), 0), (ix, iy), (200, 205, 215), 1, cv2.LINE_AA)
    t = now
    lw = max(1, s // 5)
    for j in range(4):
        a0 = math.radians(-35 + j * 22) + math.sin(t * 16.0 + j * 1.4) * 0.16
        for side in (1.0, -1.0):
            a = a0 if side > 0 else math.pi - a0
            dx, dy = math.cos(a), math.sin(a)
            L1 = s * 1.7
            knee = (int(ix + dx * L1), int(iy + dy * L1 + s * 0.4))
            foot = (int(ix + dx * L1 * 1.5), int(iy + dy * L1 * 1.5 + s * 0.9))
            cv2.line(out, (ix, iy), knee, dark, lw)
            cv2.line(out, knee, foot, dark, lw)
    cv2.ellipse(out, (ix, iy), (s, int(s * 1.4)), 0, 0, 360, dark, -1)
    cv2.circle(out, (ix, iy - int(s * 1.3)), int(s * 0.7), dark, -1)
    er = max(1, s // 5)
    cv2.circle(out, (ix - int(s * 0.35), iy - int(s * 1.5)), er, (0, 0, 255), -1)
    cv2.circle(out, (ix + int(s * 0.35), iy - int(s * 1.5)), er, (0, 0, 255), -1)
    return out


def draw_creature(img, ev, now):
    k = ev["kind"]
    if k == "wraith":
        return draw_wraith(img, ev, now)
    if k == "spirit":
        return draw_spirit(img, ev, now)
    if k == "eyes":
        return draw_eyes(img, ev, now)
    if k == "shoulder":
        return draw_shoulder_eyes(img, ev, now)
    if k == "ghost":
        return draw_ghost(img, ev, now)
    if k == "spider":
        return draw_spider(img, ev, now)
    if k == "bat":
        return draw_bat(img, ev, now)
    return img


# ---------------------------------------------------------------- state
def spawn(state, kind, now, mask, head):
    if kind == "wraith" and head is not None:
        state.creatures.append({"kind": "wraith", "t0": now, "dur": WRAITH_DUR,
                                "pos": head, "layer": "behind"})
    elif kind == "spirit" and mask is not None:
        m = (mask > 100).astype(np.float32) * 255.0
        m = cv2.GaussianBlur(m, (0, 0), 6)
        state.creatures.append({"kind": "spirit", "t0": now, "dur": SPIRIT_DUR,
                                "mask": m, "drift": random.randint(-140, 140),
                                "layer": "front"})
    elif kind == "eyes":
        state.creatures.append({"kind": "eyes", "t0": now, "dur": EYES_DUR,
                                "eyes": sample_eye_points(random.randint(5, 9)),
                                "er": 7, "layer": "front"})
    elif kind == "shoulder" and head is not None:
        state.creatures.append({"kind": "shoulder", "t0": now, "dur": SHOULDER_DUR,
                                "pos": (head[0], head[1] - 14), "layer": "front"})
    elif kind == "ghost":
        side = random.choice([-1, 1])
        x0 = PROC_W + 70 if side > 0 else -70
        y0 = random.randint(50, int(PROC_H * 0.6))
        dx = -side * (PROC_W + 140)
        state.creatures.append({
            "kind": "ghost", "t0": now, "dur": GHOST_DUR,
            "start": (x0, y0), "vel": (dx, random.randint(-40, 40)),
            "wob": random.randint(15, 35), "wobf": random.randint(1, 3),
            "size": (random.randint(90, 150), random.randint(130, 220)),
            "alpha": random.uniform(0.45, 0.7), "layer": "front",
        })
    elif kind == "spider":
        x0 = random.randint(60, PROC_W - 60)
        drop_y = random.randint(60, int(PROC_H * 0.55))
        x1 = x0 + random.choice([-1, 1]) * random.randint(80, 320)
        y1 = drop_y + random.randint(-40, 140)
        state.creatures.append({
            "kind": "spider", "t0": now, "dur": SPIDER_DUR,
            "x0": x0, "drop_y": drop_y, "x1": x1, "y1": y1,
            "size": random.randint(10, 16), "layer": "front",
        })
    elif kind == "bat":
        side = random.choice([-1, 1])
        x0 = PROC_W + 50 if side > 0 else -50
        y0 = random.randint(45, 170)
        state.creatures.append({
            "kind": "bat", "t0": now, "dur": BAT_DUR,
            "x0": x0, "y0": y0, "dx": -side * (PROC_W + 100),
            "dy": random.randint(-25, 25), "wob": random.randint(6, 16),
            "wobf": random.randint(1, 3), "size": random.randint(28, 46),
            "flap": random.uniform(18, 28), "layer": "front",
        })


def maybe_spawn_creature(state, now, mask, head):
    if now < state.next_creature:
        return
    state.next_creature = now + random.uniform(CREATURE_MIN, CREATURE_MAX)
    kind = random.choice(["wraith", "spirit", "eyes", "shoulder"])
    spawn(state, kind, now, mask, head)


def maybe_spawn_ghost(state, now):
    if now < state.next_ghost:
        return
    state.next_ghost = now + random.uniform(GHOST_MIN, GHOST_MAX)
    spawn(state, "ghost", now, None, None)


def maybe_spawn_spider(state, now):
    if now < state.next_spider:
        return
    state.next_spider = now + random.uniform(SPIDER_MIN, SPIDER_MAX)
    n = sum(1 for c in state.creatures if c["kind"] == "spider")
    if n < 3:
        spawn(state, "spider", now, None, None)


def maybe_spawn_bat(state, now):
    if now < state.next_bat:
        return
    state.next_bat = now + random.uniform(BAT_MIN, BAT_MAX)
    spawn(state, "bat", now, None, None)


def maybe_lightning(state, now):
    if now < state.next_lightning:
        return
    state.next_lightning = now + random.uniform(LIGHTNING_MIN, LIGHTNING_MAX)
    state.lightning_t0 = now
    spawn(state, "eyes", now + 0.15, None, None)   # eyes open in the dark beat
    play_thunder()


def lightning_gain(now, t0):
    if t0 is None or now < t0:
        return 1.0
    dt = now - t0
    if dt < 0.12:
        return 2.4
    if dt < 0.55:
        return 0.12
    return 1.0


# ---------------------------------------------------------------- pipeline
def process_frame(color, mask, head, depth_r, faces, state, now):
    present = person_present(mask)

    if present and not state.awake:
        state.awake = True
        state.next_creature = now + 2.5
    elif not present:
        state.awake = False

    # close-up face scare
    if faces and depth_r is not None and now >= state.mask_cd:
        fd = face_distance(depth_r, faces)
        if fd is not None and fd[1] < FACE_CLOSE_MM:
            state.mask_scare_t0 = now
            state.mask_cd = now + random.uniform(MASK_CD_MIN, MASK_CD_MAX)
            play_scare()

    # during a scare, take over the whole screen (suspend all effects)
    if state.mask_scare_t0 is not None:
        if now - state.mask_scare_t0 < MASK_SCARE_DUR:
            return render_mask_scare(state, now)
        state.mask_scare_t0 = None

    maybe_lightning(state, now)
    maybe_spawn_spider(state, now)
    maybe_spawn_bat(state, now)
    if state.awake:
        maybe_spawn_ghost(state, now)
        maybe_spawn_creature(state, now, mask, head)

    state.creatures = [c for c in state.creatures if (now - c["t0"]) < c["dur"]]

    bg = moody_background(color)
    bg = apply_fog(bg, state.fog, now)
    g = lightning_gain(now, state.lightning_t0)
    if g != 1.0:
        bg = np.clip(bg.astype(np.float32) * g, 0, 255).astype(np.uint8)

    # behind-layer creatures (occluded by the body)
    for c in state.creatures:
        if c.get("layer") == "behind":
            bg = draw_creature(bg, c, now)

    out = bg
    if present:
        out = composite_person(bg, color, mask)

    # front-layer creatures (always visible)
    for c in state.creatures:
        if c.get("layer") != "behind":
            out = draw_creature(out, c, now)

    if present:
        out = cv2.addWeighted(out, 1.0, glow_outline(mask), 0.7, 0)

    out = vignette(out, 0.45)
    out = cv2.addWeighted(out, 1.0, state.cobweb, 0.18, 0)
    out = spooky_flicker(out, now)
    out = glitch(out)
    if lightning_gain(now, state.lightning_t0) != 1.0:
        out = shake(out, now, 12)
    return out


# ---------------------------------------------------------------- main
def main():
    print("Opening Kinect sensor...")
    kinect = PyKinectRuntime.PyKinectRuntime(
        FrameSourceTypes_Color | FrameSourceTypes_Depth)

    state = SimpleNamespace(
        fog=make_fog_field(PROC_H, PROC_W),
        cobweb=make_cobweb_layer(PROC_H, PROC_W),
        creatures=[],
        awake=False,
        next_creature=0.0,
        next_ghost=0.0,
        next_spider=0.0,
        next_bat=0.0,
        mask_scare_t0=None,
        mask_cd=0.0,
        lightning_t0=None,
        next_lightning=time.time() + random.uniform(3, 6),
    )

    print("The Haunting running. Keys: 1 wraith, 2 spirit, 3 eyes, 4 shoulder,"
          " g ghost, s spider, b bat, f fullscreen, SPACE random, l lightning, q quit")
    cv2.namedWindow("Hjemsøkelsen", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hjemsøkelsen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True
    bg_proc = start_background()
    mask = None
    head = None
    try:
        while True:
            color, depth = grab(kinect)
            now = time.time()
            if color is not None:
                color = resize_color(color)
                depth_r = depth_resized(depth, PROC_H, PROC_W) if depth is not None else None
                mask = person_mask(depth) if depth is not None else \
                    np.zeros((PROC_H, PROC_W), np.uint8)
                head = head_position(mask)
                faces = detect_faces(color) if (mask > 100).sum() > 4000 else []
                out = process_frame(color, mask, head, depth_r, faces, state, now)

                scaring = (state.mask_scare_t0 is not None and
                           now - state.mask_scare_t0 < MASK_SCARE_DUR)
                if not scaring:
                    items = [("Hjemsøkelsen", (PROC_W // 2, 46), 72)]
                    dist = person_distance(depth_r, mask)
                    if dist is None or dist > 1200:
                        items += [("KOM NÆRMERE", (PROC_W // 2, PROC_H // 2 - 30), 118),
                                  ("HVIS DU TØR", (PROC_W // 2, PROC_H // 2 + 70), 78)]
                    out = draw_texts_scary(out, items)
                cv2.imshow("Hjemsøkelsen", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                state.lightning_t0 = now
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty("Hjemsøkelsen", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("g"):
                spawn(state, "ghost", now, None, None)
            elif key == ord("s"):
                spawn(state, "spider", now, None, None)
            elif key == ord("b"):
                spawn(state, "bat", now, None, None)
            elif key == ord(" "):
                spawn(state, random.choice(["wraith", "spirit", "eyes", "shoulder"]),
                      now, mask, head)
            elif chr(key) in "1234":
                kinds = {"1": "wraith", "2": "spirit", "3": "eyes", "4": "shoulder"}
                spawn(state, kinds[chr(key)], now, mask, head)
    finally:
        if bg_proc is not None:
            bg_proc.terminate()
        cv2.destroyAllWindows()
        kinect.close()


if __name__ == "__main__":
    main()
