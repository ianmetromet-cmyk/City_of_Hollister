#!/usr/bin/env python3
"""Process Water and Storm stills for the homepage photo essay."""

import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "photos")

WATER_JJ = "/Users/iannewman/Desktop/Hollister/Hollister Water Department/Johnny and Jorge"
WATER_RJ = "/Users/iannewman/Desktop/Hollister/Hollister Water Department/Rodrigo and Jason"
STORM = "/Users/iannewman/Desktop/Hollister/Storm Department"

FULL_MAX = 1400
THUMB_MAX = 800

# name, source path, crop fractions (L,T,R,B), lo, hi, exponent, sat
PHOTOS = [
    ("water-hydrant-truck",
     os.path.join(WATER_JJ, "IMG_1112.JPG"),
     (0.0, 0.04, 1.0, 1.0), 12, 250, 0.98, 1.04),
    ("water-curb-paint",
     os.path.join(WATER_JJ, "IMG_1093.JPG"),
     (0.05, 0.0, 1.0, 1.0), 14, 250, 1.00, 1.05),
    ("water-valve-crew",
     os.path.join(WATER_RJ, "IMG_1080.JPG"),
     (0.0, 0.06, 1.0, 1.0), 14, 248, 0.98, 1.04),
    ("storm-seiu-vactor",
     os.path.join(STORM, "P1220224.JPG"),
     (0.05, 0.0, 0.95, 1.0), 10, 248, 0.96, 1.04),
    ("water-valve-wrench",
     os.path.join(WATER_RJ, "IMG_1077.JPG"),
     (0.08, 0.0, 0.95, 1.0), 12, 250, 0.98, 1.04),
    ("water-rodrigo-flush",
     os.path.join(WATER_RJ, "Rodrigo.png"),
     (0.0, 0.0, 1.0, 1.0), 10, 248, 0.97, 1.02),
]


def levels(im, lo, hi, exponent):
    scale = 255.0 / (hi - lo)
    table = []
    for i in range(256):
        v = (i - lo) * scale
        v = 0.0 if v < 0 else (255.0 if v > 255 else v)
        v = 255.0 * pow(v / 255.0, exponent)
        table.append(int(round(v)))
    return im.point(table * 3)


def fit(im, box):
    w, h = im.size
    if w >= h:
        if w <= box:
            return im
        return im.resize((box, round(h * box / w)), Image.LANCZOS)
    if h <= box:
        return im
    return im.resize((round(w * box / h), box), Image.LANCZOS)


def save(im, path, quality):
    im.save(path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=1)


os.makedirs(OUT, exist_ok=True)
print(f"{'name':24s} {'full':12s} {'thumb':11s} {'fullKB':>7s} {'thumbKB':>8s}")
total = 0
for name, src, crop, lo, hi, exponent, sat in PHOTOS:
    im = Image.open(src)
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    im = im.crop((round(crop[0] * w), round(crop[1] * h),
                  round(crop[2] * w), round(crop[3] * h)))
    im = levels(im, lo, hi, exponent)
    if sat != 1.0:
        im = ImageEnhance.Color(im).enhance(sat)

    full = fit(im, FULL_MAX)
    full = full.filter(ImageFilter.UnsharpMask(radius=0.7, percent=55, threshold=3))
    full_path = os.path.join(OUT, name + ".jpg")
    save(full, full_path, 76)

    thumb = fit(im, THUMB_MAX)
    thumb = thumb.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=3))
    thumb_path = os.path.join(OUT, name + "-thumb.jpg")
    save(thumb, thumb_path, 68)

    fb = os.path.getsize(full_path)
    tb = os.path.getsize(thumb_path)
    total += fb + tb
    print(f"{name:24s} {full.size[0]}x{full.size[1]:<7} {thumb.size[0]}x{thumb.size[1]:<6} "
          f"{fb/1024:7.0f} {tb/1024:8.0f}")

print(f"\n{len(PHOTOS)} photos, {total/1024/1024:.2f} MB total")
