#!/usr/bin/env python3
"""One-pass: reprocess Tania admin stills + export Water section photos."""

import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "photos")
ENG = "/Users/iannewman/Desktop/Hollister/Engineering Department"
WATER_MOV = "/Users/iannewman/Desktop/Hollister/Hollister Water Department/Rodrigo and Jason"
TMP = "/tmp/holl_finish"
HOLL_NEW = "/tmp/holl_new"

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


def save_pair(im, name, full_max=FULL_MAX):
    full = sharpen(fit(im, full_max), radius=0.9, percent=78, threshold=2)
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
    mask = np.zeros((h, w), np.float32)
    cx, cy = int(w * 0.48), int(h * 0.34)
    axes = (int(w * 0.11), int(h * 0.18))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8, w // 80))
    return mask


def reduce_glass_haze(im):
    cv = to_cv(im)
    h, w = cv.shape[:2]
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    blur = cv2.GaussianBlur(v, (0, 0), 12)
    local = v - blur
    haze = ((v > 168) & (s < 55) & (local > -8)).astype(np.uint8) * 255
    right = np.zeros_like(haze)
    right[:, int(w * 0.58) : int(w * 0.92)] = 255
    right_haze = cv2.bitwise_and(haze, right)
    desk = np.zeros_like(haze)
    desk[int(h * 0.42) : int(h * 0.78), int(w * 0.15) : int(w * 0.75)] = 255
    desk_haze = cv2.bitwise_and(haze, desk)
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
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    alpha = cv2.GaussianBlur(alpha, (0, 0), 6)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    blended = cv * (1.0 - alpha * 0.85) + inpainted.astype(np.float32) * (alpha * 0.85)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L2 = L.copy()
    hi = L > 175
    L2[hi] = 175 + (L[hi] - 175) * 0.55
    L2 = L2 + (L2 - cv2.GaussianBlur(L2, (0, 0), 3)) * 0.18
    L_out = L * protect + L2 * (1.0 - protect)
    lab[:, :, 0] = np.clip(L_out, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return from_cv(out)


def reduce_monitor_glare(im):
    cv = to_cv(im)
    h, w = cv.shape[:2]
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    screen = np.zeros((h, w), np.uint8)
    cv2.rectangle(screen, (int(w * 0.18), int(h * 0.08)), (int(w * 0.48), int(h * 0.55)), 255, -1)
    cv2.rectangle(screen, (int(w * 0.48), int(h * 0.08)), (int(w * 0.78), int(h * 0.55)), 255, -1)

    glare = ((v > 200) & (s < 45)).astype(np.uint8) * 255
    glare = cv2.bitwise_and(glare, screen)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare = cv2.dilate(glare, k, iterations=1)

    inpainted = cv2.inpaint(cv, glare, 4, cv2.INPAINT_NS)
    alpha = (glare.astype(np.float32) / 255.0)[..., None]
    alpha = cv2.GaussianBlur(alpha, (0, 0), 4)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    blended = cv * (1.0 - alpha * 0.75) + inpainted.astype(np.float32) * (alpha * 0.75)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    L_eq = clahe.apply(L_u8).astype(np.float32)
    screen_f = cv2.GaussianBlur(screen.astype(np.float32) / 255.0, (0, 0), 10)
    L_out = L * (1.0 - screen_f * 0.55) + L_eq * (screen_f * 0.55)
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


def best_admin_src(png_path, mov_name, t, tmp_name):
    """Prefer video frame when it is larger than the approved PNG crop."""
    png = pil_from(png_path)
    mov = os.path.join(ENG, mov_name)
    hires = os.path.join(TMP, tmp_name)
    if os.path.isfile(mov):
        grab(mov, t, hires)
        hi = pil_from(hires)
        if max(hi.size) >= max(png.size):
            # Match PNG aspect (approved framing)
            pw, ph = png.size
            target = pw / ph
            w, h = hi.size
            crop_h = min(h, int(w / target))
            top = 0
            hi = hi.crop((0, top, w, top + crop_h))
            return hi, f"video {mov_name}@{t}s -> {hi.size}"
    return png, f"png {os.path.basename(png_path)} -> {png.size}"


def stronger_glass(im):
    im = reduce_glass_haze(im)
    cv = to_cv(im)
    h, w = cv.shape[:2]
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    haze = ((v > 155) & (s < 70)).astype(np.uint8) * 255
    region = np.zeros_like(haze)
    region[int(h * 0.38) :, :] = 255
    region[:, int(w * 0.55) :] = 255
    region[int(h * 0.15) : int(h * 0.55), int(w * 0.15) : int(w * 0.55)] = 255
    haze = cv2.bitwise_and(haze, region)
    protect = face_protect_mask(h, w)
    mask = ((haze.astype(np.float32) / 255.0) * (1.0 - protect) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), k, iterations=1)
    inpainted = cv2.inpaint(cv, mask, 6, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 8)[..., None]
    blended = np.clip(cv * (1 - alpha * 0.9) + inpainted.astype(np.float32) * (alpha * 0.9), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L2 = L.copy()
    hi = L > 165
    L2[hi] = 165 + (L[hi] - 165) * 0.45
    L2 = L2 + (L2 - cv2.GaussianBlur(L2, (0, 0), 3)) * 0.22
    lab[:, :, 0] = np.clip(L * protect + L2 * (1.0 - protect), 0, 255)
    return from_cv(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))


def stronger_monitors(im):
    im = reduce_monitor_glare(im)
    cv = to_cv(im)
    h, w = cv.shape[:2]
    hsv = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    screen = np.zeros((h, w), np.uint8)
    cv2.rectangle(screen, (int(w * 0.16), int(h * 0.05)), (int(w * 0.80), int(h * 0.58)), 255, -1)
    glare = cv2.bitwise_and(((v > 190) & (s < 55)).astype(np.uint8) * 255, screen)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    glare = cv2.dilate(glare, k, iterations=2)
    inpainted = cv2.inpaint(cv, glare, 5, cv2.INPAINT_NS)
    alpha = cv2.GaussianBlur(glare.astype(np.float32) / 255.0, (0, 0), 5)[..., None]
    blended = np.clip(cv * (1 - alpha * 0.82) + inpainted.astype(np.float32) * (alpha * 0.82), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L_eq = clahe.apply(np.clip(L, 0, 255).astype(np.uint8)).astype(np.float32)
    sf = cv2.GaussianBlur(screen.astype(np.float32) / 255.0, (0, 0), 12)
    L_out = L * (1.0 - sf * 0.6) + L_eq * (sf * 0.6)
    hi = (L_out > 205) & (sf > 0.25)
    L_out = L_out.copy()
    L_out[hi] = 205 + (L_out[hi] - 205) * 0.35
    lab[:, :, 0] = np.clip(L_out, 0, 255)
    return from_cv(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))


def process_tania():
    print("=== TANIA / ENGINEERING ===")
    smile, note = best_admin_src(
        os.path.join(HOLL_NEW, "admin1_smile.png"),
        "MVI_1047.MOV", 29.3, "smile_hires.jpg",
    )
    print("  smile source:", note)
    smile = stronger_glass(smile)
    smile = polish(smile, color=1.05, contrast=1.12)
    smile = smile.filter(ImageFilter.UnsharpMask(radius=1.15, percent=55, threshold=2))
    save_pair(smile, "admin-desk-smile")

    monitors, note = best_admin_src(
        os.path.join(HOLL_NEW, "admin2_monitors.png"),
        "MVI_1045.MOV", 10.0, "monitors_hires.jpg",
    )
    print("  monitors source:", note)
    monitors = stronger_monitors(monitors)
    monitors = polish(monitors, color=1.03, contrast=1.08)
    monitors = monitors.filter(ImageFilter.UnsharpMask(radius=1.05, percent=55, threshold=2))
    save_pair(monitors, "admin-monitors")
    print("  admin-papers left untouched")


def process_water():
    print("=== WATER ===")
    # Truck: prefer frame file; refresh from MOV ~5s if available
    truck_src = "/tmp/water_best/MVI_1062_005_0.jpg"
    mov1062 = os.path.join(WATER_MOV, "MVI_1062.MOV")
    if os.path.isfile(mov1062):
        grabbed = os.path.join(TMP, "water_truck_5s.jpg")
        grab(mov1062, 5.0, grabbed)
        truck_src = grabbed
        print("  truck from MVI_1062.MOV @5.0s")
    else:
        print("  truck from", truck_src)

    truck = polish(pil_from(truck_src), color=1.04, contrast=1.08)
    w, h = truck.size
    truck = truck.crop((int(w * 0.02), int(h * 0.05), int(w * 0.98), int(h * 0.98)))
    truck = truck.filter(ImageFilter.UnsharpMask(radius=1.0, percent=55, threshold=2))
    save_pair(truck, "water-truck")

    # Hydrant
    hyd_src = "/tmp/water_best/MVI_1065_016_5.jpg"
    mov1065 = os.path.join(WATER_MOV, "MVI_1065.MOV")
    if os.path.isfile(mov1065):
        grabbed = os.path.join(TMP, "water_hydrant_16_5.jpg")
        grab(mov1065, 16.5, grabbed)
        hyd_src = grabbed
        print("  hydrant from MVI_1065.MOV @16.5s")
    else:
        print("  hydrant from", hyd_src)

    hyd = polish(pil_from(hyd_src), color=1.05, contrast=1.09)
    w, h = hyd.size
    hyd = hyd.crop((int(w * 0.02), int(h * 0.02), int(w * 0.98), int(h * 0.98)))
    hyd = hyd.filter(ImageFilter.UnsharpMask(radius=1.05, percent=60, threshold=2))
    save_pair(hyd, "water-hydrant")

    # Valve over (hires still)
    valve = polish(pil_from("/tmp/water_hires/P1220328.jpg"), color=1.03, contrast=1.07)
    w, h = valve.size
    valve = valve.crop((int(w * 0.04), int(h * 0.02), int(w * 0.98), int(h * 0.96)))
    valve = valve.filter(ImageFilter.UnsharpMask(radius=0.95, percent=55, threshold=2))
    save_pair(valve, "water-valve-over")

    # Valve smile: prefer hires P1220290 (fits water work; sharpen)
    smile_src = "/tmp/water_hires/P1220290.jpg"
    if not os.path.isfile(smile_src):
        smile_src = "/tmp/water_match/USER_valve_smile.png"
    print("  valve smile from", smile_src)
    smile = polish(pil_from(smile_src), color=1.04, contrast=1.08)
    w, h = smile.size
    smile = smile.crop((int(w * 0.08), int(h * 0.02), int(w * 0.92), int(h * 0.98)))
    smile = smile.filter(ImageFilter.UnsharpMask(radius=1.0, percent=58, threshold=2))
    save_pair(smile, "water-valve-smile")
    print("  DROPPED posed IMG_1084 (not exported)")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    process_tania()
    process_water()
    # Sanity: admin-team untouched
    team = os.path.join(OUT, "admin-team.jpg")
    if os.path.isfile(team):
        im = Image.open(team)
        print(f"=== PRESERVED admin-team {im.size[0]}x{im.size[1]} ===")
    print("done")


if __name__ == "__main__":
    main()
