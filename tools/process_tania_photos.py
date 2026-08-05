#!/usr/bin/env python3
"""Sharpen and de-glare Tania engineering gallery stills from hi-res video frames."""

import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "photos")
ENG = "/Users/iannewman/Desktop/Hollister/Engineering Department"
TMP = "/tmp/holl_admin_hires"

# Match quality bump used elsewhere on the site (upgrade_image_quality.py),
# staying within the ~76–88 gallery range for thumbs / mid-high for fulls.
FULL_MAX = 1820
THUMB_MAX = 1040
FULL_Q = 88
THUMB_Q = 82


def grab(mov, t, out):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{t:.3f}", "-i", mov,
            "-frames:v", "1", "-q:v", "2", out,
        ],
        check=True,
    )


def pil_from(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def to_cv(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def from_cv(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def fit(im, box):
    w, h = im.size
    if max(w, h) <= box:
        return im
    if w >= h:
        return im.resize((box, round(h * box / w)), Image.LANCZOS)
    return im.resize((round(w * box / h), box), Image.LANCZOS)


def levels(im, lo=8, hi=248, exponent=1.0):
    scale = 255.0 / (hi - lo)
    table = []
    for i in range(256):
        v = (i - lo) * scale
        v = 0.0 if v < 0 else (255.0 if v > 255 else v)
        v = 255.0 * pow(v / 255.0, exponent)
        table.append(int(round(v)))
    return im.point(table * 3)


def sharpen(im, radius=0.85, percent=72, threshold=2):
    return im.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def save_pair(im, name):
    full = sharpen(fit(im, FULL_MAX), radius=0.9, percent=78, threshold=2)
    full_path = os.path.join(OUT, name + ".jpg")
    full.save(full_path, "JPEG", quality=FULL_Q, optimize=True, progressive=True, subsampling=0)

    thumb = sharpen(fit(im, THUMB_MAX), radius=0.8, percent=85, threshold=2)
    thumb_path = os.path.join(OUT, name + "-thumb.jpg")
    thumb.save(thumb_path, "JPEG", quality=THUMB_Q, optimize=True, progressive=True, subsampling=0)

    print(
        f"  {name:28s} {full.size[0]}x{full.size[1]}  "
        f"{os.path.getsize(full_path) / 1024:.0f}KB  "
        f"thumb {thumb.size[0]}x{thumb.size[1]}"
    )
    return full.size, thumb.size


def face_protect_mask(h, w):
    """Soft ellipse over Tania's face/upper torso so deglare stays off skin."""
    mask = np.zeros((h, w), np.float32)
    # Approx face center for through-glass desk framing
    cx, cy = int(w * 0.48), int(h * 0.34)
    axes = (int(w * 0.11), int(h * 0.18))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8, w // 80))
    return mask


def reduce_glass_haze(im):
    """
    Cut milky window reflections without wiping the face.
    Strategy: build a specular/haze mask from V-channel highs + low local
    contrast, inpaint those regions, then blend back under a face protect mask.
    Also gently compress remaining highlights and lift mid contrast.
    """
    cv = to_cv(im)
    h, w = cv.shape[:2]
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    # Local contrast proxy: bright + desaturated + flatter than neighbors
    blur = cv2.GaussianBlur(v, (0, 0), 12)
    local = v - blur
    haze = ((v > 168) & (s < 55) & (local > -8)).astype(np.uint8) * 255
    # Strong vertical glare band on right (chair / wall)
    right = np.zeros_like(haze)
    right[:, int(w * 0.58) : int(w * 0.92)] = 255
    right_haze = cv2.bitwise_and(haze, right)
    # Desk / lower mid haze strip (avoid bottom window sill)
    desk = np.zeros_like(haze)
    desk[int(h * 0.42) : int(h * 0.78), int(w * 0.15) : int(w * 0.75)] = 255
    desk_haze = cv2.bitwise_and(haze, desk)
    # Monitor back specular streaks
    mon = np.zeros_like(haze)
    mon[int(h * 0.22) : int(h * 0.52), int(w * 0.18) : int(w * 0.48)] = 255
    mon_haze = cv2.bitwise_and(((v > 200) & (s < 40)).astype(np.uint8) * 255, mon)

    mask = cv2.bitwise_or(right_haze, desk_haze)
    mask = cv2.bitwise_or(mask, mon_haze)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, k, iterations=1)

    protect = face_protect_mask(h, w)
    mask_f = mask.astype(np.float32) / 255.0
    mask_f *= 1.0 - protect
    mask = np.clip(mask_f * 255, 0, 255).astype(np.uint8)

    inpainted = cv2.inpaint(cv, mask, 5, cv2.INPAINT_TELEA)

    # Soft clone: mix inpainted into original where haze was strongest
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    alpha = cv2.GaussianBlur(alpha, (0, 0), 6)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    blended = cv * (1.0 - alpha * 0.85) + inpainted.astype(np.float32) * (alpha * 0.85)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Highlight recovery on remaining milky veil (outside face)
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    # Compress top end slightly
    L2 = L.copy()
    hi = L > 175
    L2[hi] = 175 + (L[hi] - 175) * 0.55
    # Restore a bit of local contrast
    L2 = L2 + (L2 - cv2.GaussianBlur(L2, (0, 0), 3)) * 0.18
    # Keep face L closer to original
    L_out = L * protect + L2 * (1.0 - protect)
    lab[:, :, 0] = np.clip(L_out, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return from_cv(out)


def reduce_monitor_glare(im):
    """Recover washed screen content; leave subject (hair/blouse) alone."""
    cv = to_cv(im)
    h, w = cv.shape[:2]
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    # Screen regions: dual monitors roughly center-left to center
    screen = np.zeros((h, w), np.uint8)
    # left monitor
    cv2.rectangle(screen, (int(w * 0.18), int(h * 0.08)), (int(w * 0.48), int(h * 0.55)), 255, -1)
    # right monitor
    cv2.rectangle(screen, (int(w * 0.48), int(h * 0.08)), (int(w * 0.78), int(h * 0.55)), 255, -1)

    glare = ((v > 200) & (s < 45)).astype(np.uint8) * 255
    glare = cv2.bitwise_and(glare, screen)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare = cv2.dilate(glare, k, iterations=1)

    # Prefer Navier-Stokes for smoother screen fills
    inpainted = cv2.inpaint(cv, glare, 4, cv2.INPAINT_NS)
    alpha = (glare.astype(np.float32) / 255.0)[..., None]
    alpha = cv2.GaussianBlur(alpha, (0, 0), 4)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    blended = cv * (1.0 - alpha * 0.75) + inpainted.astype(np.float32) * (alpha * 0.75)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Local contrast boost only on screen area to bring spreadsheet rows back
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    L_eq = clahe.apply(L_u8).astype(np.float32)
    screen_f = (screen.astype(np.float32) / 255.0)
    # Feather screen mask
    screen_f = cv2.GaussianBlur(screen_f, (0, 0), 10)
    L_out = L * (1.0 - screen_f * 0.55) + L_eq * (screen_f * 0.55)
    # Compress residual blown whites on screens
    hi = (L_out > 210) & (screen_f > 0.3)
    L_out = L_out.copy()
    L_out[hi] = 210 + (L_out[hi] - 210) * 0.4
    lab[:, :, 0] = np.clip(L_out, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return from_cv(out)


def polish(im, color=1.04, contrast=1.08):
    im = levels(im, 6, 250, 0.96)
    im = ImageEnhance.Color(im).enhance(color)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Sharpness(im).enhance(1.12)
    return im


def edit_smile():
    src = os.path.join(TMP, "smile_hires.jpg")
    grab(os.path.join(ENG, "MVI_1047.MOV"), 29.3, src)
    im = pil_from(src)
    im = reduce_glass_haze(im)
    im = polish(im, color=1.05, contrast=1.10)
    # Mild extra unsharp before save_pair's pass (through-glass softness)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.1, percent=45, threshold=2))
    return im


def edit_monitors():
    src = os.path.join(TMP, "monitors_hires.jpg")
    grab(os.path.join(ENG, "MVI_1045.MOV"), 10.0, src)
    im = pil_from(src)
    # Source gallery aspect ~1024x590; slight top crop matches published framing
    w, h = im.size
    target_aspect = 1024 / 590
    crop_h = min(h, int(w / target_aspect))
    top = 0
    im = im.crop((0, top, w, top + crop_h))
    im = reduce_monitor_glare(im)
    im = polish(im, color=1.03, contrast=1.07)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=50, threshold=2))
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    print("=== TANIA ENGINEERING STILLS ===")
    smile = edit_smile()
    smile.save(os.path.join(TMP, "smile_processed.png"))
    save_pair(smile, "admin-desk-smile")

    monitors = edit_monitors()
    monitors.save(os.path.join(TMP, "monitors_processed.png"))
    save_pair(monitors, "admin-monitors")
    print("done")


if __name__ == "__main__":
    main()
