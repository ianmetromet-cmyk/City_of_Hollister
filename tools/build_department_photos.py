#!/usr/bin/env python3
"""Rebuild the department banner photos from the pre-compression originals.

The originals live in git history at 7761d95, before the banner images were
downsized. Each frame is a group shot against a bright hedge, which left the
people underexposed, so these get a gentle midtone lift and a small white
point extension. Black points are left alone because none of the frames clip.

Two outputs per department: the banner thumbnail at the repo root, which keeps
its existing filename so the markup does not change, and a larger copy under
photos/ for the lightbox. Both get the same correction so the photo does not
shift in tone when a visitor opens it.
"""

import os
import subprocess
from PIL import Image, ImageEnhance, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN_COMMIT = "7761d95"

BANNER_MAX = 600
# These frames are group shots against a dense hedge, and that foliage detail
# is expensive to encode. 1400 with a lighter sharpen keeps faces readable
# without the half megabyte per photo that 1600 costs here.
FULL_MAX = 1400

# source stem, lightbox name, black point, white point, midtone exponent, saturation
DEPARTMENTS = [
    ("planning",   "dept-planning",   20, 248, 0.93, 1.02),
    ("sanitation", "dept-sanitation",  4, 246, 0.90, 1.00),
    ("airport",    "dept-airport",     6, 246, 0.90, 1.00),
    ("streets",    "dept-streets",     8, 245, 0.90, 1.02),
    ("ssas",       "dept-admin",       2, 247, 0.86, 1.00),
]


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
    im.save(path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=1)


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

    full = fit(im, FULL_MAX)
    full = full.filter(ImageFilter.UnsharpMask(radius=0.6, percent=35, threshold=4))
    full_path = os.path.join(REPO, "photos", out_name + ".jpg")
    save(full, full_path, 68)

    banner = fit(im, BANNER_MAX)
    banner = banner.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=3))
    banner_path = os.path.join(REPO, stem + ".jpg")
    save(banner, banner_path, 68)

    rows.append((stem, source_size, full.size, os.path.getsize(full_path),
                 banner.size, os.path.getsize(banner_path)))

print(f"{'source':12s} {'original':12s} {'lightbox':12s} {'KB':>5s}  {'banner':10s} {'KB':>5s}")
for stem, src, fs, fb, bs, bb in rows:
    print(f"{stem:12s} {src[0]}x{src[1]:<6} {fs[0]}x{fs[1]:<7} {fb/1024:5.0f}  "
          f"{bs[0]}x{bs[1]:<5} {bb/1024:5.0f}")
print(f"\nlightbox total {sum(r[3] for r in rows)/1024:.0f} KB, "
      f"banner total {sum(r[5] for r in rows)/1024:.0f} KB")
