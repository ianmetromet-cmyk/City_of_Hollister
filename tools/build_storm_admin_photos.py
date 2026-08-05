#!/usr/bin/env python3
"""Edit and export Storm, Admin, and cleaned Streets banner photos."""

import os
import math
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "photos")
SRC = "/tmp/holl_new"
STREETS_HIRES = "/tmp/streets_orig.jpg"

FULL_MAX = 1400
THUMB_MAX = 800
BANNER_MAX = 600


def pil_from(path):
    im = Image.open(path)
    return ImageOps.exif_transpose(im).convert("RGB")


def to_cv(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def from_cv(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def rotate_bound(im, angle_deg):
    """Rotate PIL image, expanding canvas, then crop back to content."""
    cv = to_cv(im)
    h, w = cv.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    rot = cv2.warpAffine(cv, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REPLICATE)
    return from_cv(rot)


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


def save_pair(im, name, full_max=FULL_MAX):
    full = fit(im, full_max)
    full = full.filter(ImageFilter.UnsharpMask(radius=0.7, percent=55, threshold=3))
    full_path = os.path.join(OUT, name + ".jpg")
    full.save(full_path, "JPEG", quality=76, optimize=True, progressive=True)

    thumb = fit(im, THUMB_MAX)
    thumb = thumb.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=3))
    thumb_path = os.path.join(OUT, name + "-thumb.jpg")
    thumb.save(thumb_path, "JPEG", quality=68, optimize=True, progressive=True)
    print(f"  {name:28s} {full.size[0]}x{full.size[1]}  "
          f"{os.path.getsize(full_path)/1024:.0f}KB")
    return full.size, thumb.size


def inpaint_rect(im, box, grow=18):
    """box = (left, top, right, bottom) in pixel coords."""
    cv = to_cv(im)
    mask = np.zeros(cv.shape[:2], np.uint8)
    l, t, r, b = box
    mask[t:b, l:r] = 255
    if grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
        mask = cv2.dilate(mask, k)
    out = cv2.inpaint(cv, mask, 4, cv2.INPAINT_TELEA)
    return from_cv(out)


def edit_streets():
    """Straighten, crop partial person on left, remove plastic bag."""
    im = pil_from(STREETS_HIRES)
    # Slight clockwise rotation to level the hedge line.
    im = rotate_bound(im, -1.6)
    w, h = im.size
    # Crop left partial figure and a little slack sky.
    im = im.crop((int(w * 0.10), int(h * 0.02), int(w * 0.98), int(h * 0.98)))
    w, h = im.size
    # Plastic bag is in the right man's left hand (lower-right of the trio).
    # Approx relative region after crop.
    bag = (int(w * 0.72), int(h * 0.55), int(w * 0.86), int(h * 0.78))
    im = inpaint_rect(im, bag, grow=22)
    im = levels(im, 6, 248, 0.97)
    im = ImageEnhance.Color(im).enhance(1.03)

    # Banner thumbnail at repo root + lightbox full.
    banner = fit(im, BANNER_MAX)
    banner = banner.filter(ImageFilter.UnsharpMask(radius=0.6, percent=65, threshold=3))
    banner_path = os.path.join(REPO, "streets.jpg")
    banner.save(banner_path, "JPEG", quality=72, optimize=True, progressive=True)

    full_size, thumb_size = save_pair(im, "dept-streets", full_max=1400)
    # Also overwrite photos/dept-streets used by lightbox from department buttons.
    print(f"  streets.jpg banner         {banner.size[0]}x{banner.size[1]}")
    return full_size, thumb_size


def edit_storm3():
    """Straighten and remove the crouched worker at bottom-left."""
    im = pil_from(os.path.join(SRC, "storm3.png"))
    im = rotate_bound(im, 7.5)  # counter-clockwise to undo Dutch angle
    w, h = im.size
    # Crop out the crouched worker (bottom-left) and tighten framing on main subject.
    im = im.crop((int(w * 0.18), int(h * 0.02), int(w * 0.92), int(h * 0.78)))
    # Any residual plastic/glare near right hand on red machine.
    w, h = im.size
    im = inpaint_rect(im, (int(w * 0.70), int(h * 0.55), int(w * 0.82), int(h * 0.72)), grow=14)
    im = levels(im, 10, 250, 0.98)
    return im


def edit_storm1():
    im = pil_from(os.path.join(SRC, "storm1.png"))
    im = rotate_bound(im, 0.8)
    w, h = im.size
    im = im.crop((int(w * 0.02), int(h * 0.02), int(w * 0.98), int(h * 0.98)))
    im = levels(im, 10, 250, 0.98)
    return im


def edit_storm2():
    im = pil_from(os.path.join(SRC, "storm2_paul.png"))
    im = levels(im, 8, 250, 0.97)
    im = ImageEnhance.Color(im).enhance(1.04)
    return im


def edit_admin(name, angle=0.0, crop=None):
    im = pil_from(os.path.join(SRC, name))
    if angle:
        im = rotate_bound(im, angle)
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h),
                      int(crop[2] * w), int(crop[3] * h)))
    im = levels(im, 8, 250, 0.98)
    return im


os.makedirs(OUT, exist_ok=True)
print("=== STREETS BANNER ===")
edit_streets()

print("=== STORM ===")
save_pair(edit_storm1(), "storm-hose-reel")
save_pair(edit_storm2(), "storm-paul")
save_pair(edit_storm3(), "storm-pipe-crew")

print("=== ADMIN / ENGINEERING ===")
save_pair(edit_admin("admin1_smile.png"), "admin-desk-smile")
save_pair(edit_admin("admin2_monitors.png"), "admin-monitors")
save_pair(edit_admin("admin3_papers.png"), "admin-papers")
save_pair(edit_admin("admin4_group.png", angle=-1.4,
                     crop=(0.02, 0.04, 0.98, 0.98)), "admin-team")

# Update Admin department banner card with the team photo.
team = pil_from(os.path.join(OUT, "admin-team.jpg"))
banner = fit(team, BANNER_MAX)
banner.save(os.path.join(REPO, "ssas.jpg"), "JPEG", quality=72, optimize=True, progressive=True)
banner.save(os.path.join(OUT, "dept-admin.jpg"), "JPEG", quality=76, optimize=True, progressive=True)
print("  ssas.jpg / dept-admin.jpg updated from admin-team")
print("done")
