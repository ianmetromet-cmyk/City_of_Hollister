#!/usr/bin/env python3
"""Rebuild the department banner photos from the pre-compression originals.

The originals live in git history at 7761d95, before the banner images were
downsized. Each frame is a group shot against a bright hedge, which left the
people underexposed, so these get a gentle midtone lift and a small white
point extension. Black points are left alone because none of the frames clip.

Frames are also reframed so the green hedge tops share a common relative
horizon across the five banner tiles (Planning previously showed more sky).

Admin is different: the published banner comes from the admin-team crop (five
staff, no hi-vis companion from the 7761d95 ssas plate), so that image is
re-centered horizontally rather than rebuilt from the old plate.

Two outputs per department: the banner thumbnail at the repo root, which keeps
its existing filename so the markup does not change, and a larger copy under
photos/ for the lightbox. Both get the same correction so the photo does not
shift in tone when a visitor opens it.
"""

import os
import statistics
import subprocess
from PIL import Image, ImageEnhance, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN_COMMIT = "7761d95"

BANNER_MAX = 780
# These frames are group shots against a dense hedge, and that foliage detail
# is expensive to encode. 1820 with light sharpen keeps faces readable.
FULL_MAX = 1820

# Match the airport / admin tile horizon: a thin sky strip, hedge near the top.
HEDGE_TARGET = 0.048

# source stem, lightbox name, black point, white point, midtone exponent, saturation
DEPARTMENTS = [
    ("planning",   "dept-planning",   20, 248, 0.93, 1.02),
    ("sanitation", "dept-sanitation",  4, 246, 0.90, 1.00),
    ("airport",    "dept-airport",     6, 246, 0.90, 1.00),
    ("streets",    "dept-streets",     8, 245, 0.90, 1.02),
]

# Left/right fractions on photos/dept-admin-source.jpg (empty hedge on the right).
# That file is the pre-center 780px admin-team banner; admin-team.jpg itself is
# only ~500px and is kept for the half-width essay lead.
ADMIN_CROP = (0.00, 0.00, 0.80, 1.00)
ADMIN_SOURCE = os.path.join(REPO, "photos", "dept-admin-source.jpg")


def levels(im, lo, hi, exponent):
    scale = 255.0 / (hi - lo)
    table = []
    for i in range(256):
        v = (i - lo) * scale
        v = 0.0 if v < 0 else (255.0 if v > 255 else v)
        table.append(int(round(255.0 * pow(v / 255.0, exponent))))
    return im.point(table * 3)


def fit(im, box):
    w, h = im.size
    if w >= h:
        return im if w <= box else im.resize((box, round(h * box / w)), Image.LANCZOS)
    return im if h <= box else im.resize((round(w * box / h), box), Image.LANCZOS)


def save(im, path, quality):
    im.save(path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)


def hedge_median(im, n=40):
    """Fraction of frame height occupied by sky above the hedge top."""
    w, h = im.size
    px = im.load()
    vals = []
    for i in range(n):
        x = int(w * (0.15 + 0.7 * i / (n - 1)))
        y = 0
        while y < int(h * 0.45):
            r, g, b = px[x, y]
            if b > 130 and b >= g - 5 and b >= r and (b - min(r, g)) > 12:
                y += 1
            else:
                break
        vals.append(y / h)
    vals = sorted(vals)
    core = vals[5:-5] if len(vals) > 12 else vals
    return statistics.median(core)


def align_hedge(im, target=HEDGE_TARGET):
    """Crop the top so the hedge line sits near target; mild bottom trim."""
    hf = hedge_median(im)
    w, h = im.size
    if hf <= target + 0.008:
        return im, hf
    top = int(round(h * (hf - target) / (1 - target)))
    bottom_trim = int(round(top * 0.08))
    return im.crop((0, top, w, h - bottom_trim)), hf


def export_pair(im, stem, out_name):
    full = fit(im, FULL_MAX)
    full = full.filter(ImageFilter.UnsharpMask(radius=0.6, percent=35, threshold=4))
    full_path = os.path.join(REPO, "photos", out_name + ".jpg")
    save(full, full_path, 90)

    banner = fit(im, BANNER_MAX)
    banner = banner.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=3))
    banner_path = os.path.join(REPO, stem + ".jpg")
    save(banner, banner_path, 84)

    return full.size, os.path.getsize(full_path), banner.size, os.path.getsize(banner_path)


os.makedirs(os.path.join(REPO, "photos"), exist_ok=True)
rows = []

for stem, out_name, lo, hi, exponent, sat in DEPARTMENTS:
    raw = subprocess.run(
        ["git", "show", f"{ORIGIN_COMMIT}:{stem}.jpg"],
        cwd=REPO, capture_output=True, check=True,
    ).stdout

    tmp = f"/tmp/dept_src_{stem}.jpg"
    with open(tmp, "wb") as fh:
        fh.write(raw)

    im = Image.open(tmp).convert("RGB")
    source_size = im.size
    im = levels(im, lo, hi, exponent)
    if sat != 1.0:
        im = ImageEnhance.Color(im).enhance(sat)
    im = ImageEnhance.Contrast(im).enhance(1.05)

    im, hedge_before = align_hedge(im)
    hedge_after = hedge_median(im)

    fs, fb, bs, bb = export_pair(im, stem, out_name)
    rows.append((stem, source_size, fs, fb, bs, bb, hedge_before, hedge_after))
    print(f"  {stem:12s} hedge {hedge_before:.3f} -> {hedge_after:.3f}")

# Admin: re-center the five-person frame (empty hedge was on the right).
admin = Image.open(ADMIN_SOURCE).convert("RGB")
aw, ah = admin.size
l, t, r, b = ADMIN_CROP
admin = admin.crop((int(aw * l), int(ah * t), int(aw * r), int(ah * b)))
fs, fb, bs, bb = export_pair(admin, "ssas", "dept-admin")
rows.append(("ssas", (aw, ah), fs, fb, bs, bb, None, None))
print(f"  {'ssas':12s} centered crop {ADMIN_CROP} from dept-admin-source")

print(f"\n{'source':12s} {'original':12s} {'lightbox':12s} {'KB':>5s}  {'banner':10s} {'KB':>5s}")
for stem, src, fs, fb, bs, bb, *_rest in rows:
    print(f"{stem:12s} {src[0]}x{src[1]:<6} {fs[0]}x{fs[1]:<7} {fb/1024:5.0f}  "
          f"{bs[0]}x{bs[1]:<5} {bb/1024:5.0f}")
print(f"\nlightbox total {sum(r[3] for r in rows)/1024:.0f} KB, "
      f"banner total {sum(r[5] for r in rows)/1024:.0f} KB")
