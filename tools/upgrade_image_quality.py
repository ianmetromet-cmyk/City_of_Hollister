#!/usr/bin/env python3
"""Re-export site photos ~30% higher quality (larger long edge + higher JPEG Q)."""

import os
import subprocess
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "photos")
STORM_HI = "/tmp/storm_hires"
ADMIN_SRC = "/tmp/holl_new"
STREETS_DIR = "/Users/iannewman/Desktop/Hollister/Streets Department "

# ~30% above prior defaults (1400 / q76 / q68)
FULL_MAX = 1820
THUMB_MAX = 1040
BANNER_MAX = 780
FULL_Q = 90
THUMB_Q = 82
BANNER_Q = 84


def pil_from(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


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


def save_jpeg(im, path, quality):
    im.save(path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)


def save_pair(im, name, full_max=FULL_MAX):
    full = sharpen(fit(im, full_max))
    full_path = os.path.join(OUT, name + ".jpg")
    save_jpeg(full, full_path, FULL_Q)

    thumb = sharpen(fit(im, THUMB_MAX), radius=0.75, percent=80, threshold=2)
    thumb_path = os.path.join(OUT, name + "-thumb.jpg")
    save_jpeg(thumb, thumb_path, THUMB_Q)

    print(f"  {name:28s} {full.size[0]}x{full.size[1]}  "
          f"{os.path.getsize(full_path)/1024:.0f}KB")
    return full.size


def polish(im, lo=8, hi=248, exponent=0.97, color=1.05, contrast=1.06):
    im = levels(im, lo, hi, exponent)
    im = ImageEnhance.Color(im).enhance(color)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    return im


def upgrade_storm():
    print("=== STORM (native P122 frames) ===")
    # Juan portrait
    juan = polish(pil_from(os.path.join(STORM_HI, "P1220248.jpg")), lo=6, hi=250, exponent=0.96)
    w, h = juan.size
    juan = juan.crop((int(w * 0.08), int(h * 0.02), int(w * 0.92), int(h * 0.98)))
    save_pair(juan, "storm-juan")

    # Hose reel / equipment work
    hose = polish(pil_from(os.path.join(STORM_HI, "P1220247.jpg")), lo=8, hi=248, exponent=0.97)
    w, h = hose.size
    hose = hose.crop((int(w * 0.04), int(h * 0.04), int(w * 0.96), int(h * 0.96)))
    save_pair(hose, "storm-hose-reel")

    # Vactor truck
    truck = polish(pil_from(os.path.join(STORM_HI, "P1220243.jpg")), lo=10, hi=250, exponent=0.98)
    w, h = truck.size
    truck = truck.crop((int(w * 0.02), int(h * 0.08), int(w * 0.98), int(h * 0.95)))
    save_pair(truck, "storm-vactor-truck")

    # SEIU hard-hat
    seiu = polish(pil_from(os.path.join(STORM_HI, "P1220224.jpg")), lo=12, hi=245, exponent=0.98, color=1.04)
    w, h = seiu.size
    seiu = seiu.crop((int(w * 0.12), int(h * 0.02), int(w * 0.88), int(h * 0.98)))
    save_pair(seiu, "storm-seiu-vactor")


def upgrade_admin():
    print("=== ADMIN ===")
    team_full = None
    for src, name, crop in [
        ("admin1_smile.png", "admin-desk-smile", None),
        ("admin2_monitors.png", "admin-monitors", None),
        ("admin3_papers.png", "admin-papers", None),
        ("admin4_group.png", "admin-team", (0.02, 0.04, 0.98, 0.98)),
    ]:
        im = polish(pil_from(os.path.join(ADMIN_SRC, src)), lo=6, hi=250, exponent=0.97, color=1.03)
        if crop:
            w, h = im.size
            im = im.crop((int(crop[0] * w), int(crop[1] * h),
                          int(crop[2] * w), int(crop[3] * h)))
        if name == "admin-team":
            team_full = im
            # Page lead is intentionally half-width; banners use full-res below.
            save_pair(im, name, full_max=501)
        else:
            save_pair(im, name)

    banner = sharpen(fit(team_full, BANNER_MAX), radius=0.7, percent=75)
    save_jpeg(banner, os.path.join(REPO, "ssas.jpg"), BANNER_Q)
    save_jpeg(banner, os.path.join(OUT, "dept-admin.jpg"), FULL_Q)
    print("  ssas.jpg / dept-admin.jpg refreshed")


def upgrade_departments():
    print("=== DEPARTMENT BANNERS (git originals) ===")
    specs = [
        ("planning", "dept-planning", 20, 248, 0.93, 1.02),
        ("sanitation", "dept-sanitation", 4, 246, 0.90, 1.00),
        ("airport", "dept-airport", 6, 246, 0.90, 1.00),
        ("streets", "dept-streets", 8, 245, 0.90, 1.02),
    ]
    for stem, out_name, lo, hi, exponent, sat in specs:
        raw = subprocess.run(
            ["git", "show", f"7761d95:{stem}.jpg"],
            cwd=REPO, capture_output=True, check=True,
        ).stdout
        tmp = f"/tmp/dept_up_{stem}.jpg"
        with open(tmp, "wb") as fh:
            fh.write(raw)
        im = Image.open(tmp).convert("RGB")
        im = levels(im, lo, hi, exponent)
        if sat != 1.0:
            im = ImageEnhance.Color(im).enhance(sat)
        im = ImageEnhance.Contrast(im).enhance(1.05)

        full = sharpen(fit(im, FULL_MAX), radius=0.7, percent=50, threshold=3)
        save_jpeg(full, os.path.join(OUT, out_name + ".jpg"), FULL_Q)

        banner = sharpen(fit(im, BANNER_MAX), radius=0.7, percent=75, threshold=3)
        save_jpeg(banner, os.path.join(REPO, stem + ".jpg"), BANNER_Q)
        print(f"  {stem:12s} full {full.size[0]}x{full.size[1]}  "
              f"{os.path.getsize(os.path.join(OUT, out_name + '.jpg'))/1024:.0f}KB")


def find_streets_match(target_path, folder):
    """Best perceptual match among JPGs in folder against an existing export."""
    from PIL import ImageChops, ImageStat

    def prep(path, size=96):
        im = pil_from(path)
        w, h = im.size
        cw, ch = int(w * 0.7), int(h * 0.7)
        left, top = (w - cw) // 2, (h - ch) // 2
        return im.crop((left, top, left + cw, top + ch)).resize((size, size), Image.LANCZOS)

    target = prep(target_path)
    best = None
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.upper().endswith((".JPG", ".JPEG", ".PNG")):
                continue
            path = os.path.join(root, f)
            try:
                score = sum(ImageStat.Stat(ImageChops.difference(target, prep(path))).mean)
            except Exception:
                continue
            if best is None or score < best[0]:
                best = (score, path)
    return best


STREETS_EXPORTS = [
    "crosswalk-traffic.jpg",
    "medina-crossing.jpg",
    "felix-thermolazer.jpg",
    "setting-cones.jpg",
    "felix-applicator.jpg",
    "crew-member-street.jpg",
    "felix-commercial-strip.jpg",
    "applicator-detail.jpg",
    "crew-portrait.jpg",
]


def upgrade_streets_work():
    print("=== STREETS WORK GALLERY ===")
    if not os.path.isdir(STREETS_DIR):
        print("  Streets folder missing; re-encoding existing exports only")
        for name in STREETS_EXPORTS:
            path = os.path.join(OUT, name)
            if not os.path.exists(path):
                continue
            im = polish(pil_from(path), lo=5, hi=250, exponent=0.98, color=1.03, contrast=1.05)
            stem = name.replace(".jpg", "")
            if stem.endswith("-thumb"):
                continue
            # Lead figures may not have thumbs; gallery tiles do
            full = sharpen(fit(im, FULL_MAX))
            save_jpeg(full, path, FULL_Q)
            thumb_path = os.path.join(OUT, stem + "-thumb.jpg")
            if os.path.exists(thumb_path) or stem not in (
                "crosswalk-traffic", "crew-portrait"
            ):
                if stem not in ("crosswalk-traffic", "crew-portrait"):
                    thumb = sharpen(fit(im, THUMB_MAX), radius=0.75, percent=80)
                    save_jpeg(thumb, thumb_path, THUMB_Q)
            print(f"  {stem:28s} {full.size[0]}x{full.size[1]}  "
                  f"{os.path.getsize(path)/1024:.0f}KB")
        return

    for name in STREETS_EXPORTS:
        path = os.path.join(OUT, name)
        if not os.path.exists(path):
            continue
        match = find_streets_match(path, STREETS_DIR)
        if match and match[0] < 120:
            print(f"  {name}: matched {os.path.basename(match[1])} ({match[0]:.1f})")
            im = polish(pil_from(match[1]), lo=6, hi=250, exponent=0.97, color=1.04, contrast=1.06)
        else:
            score = match[0] if match else None
            print(f"  {name}: re-encode existing (best score={score})")
            im = polish(pil_from(path), lo=5, hi=250, exponent=0.98, color=1.03, contrast=1.05)
        stem = name.replace(".jpg", "")
        full = sharpen(fit(im, FULL_MAX))
        save_jpeg(full, path, FULL_Q)
        if stem not in ("crosswalk-traffic", "crew-portrait"):
            thumb = sharpen(fit(im, THUMB_MAX), radius=0.75, percent=80)
            save_jpeg(thumb, os.path.join(OUT, stem + "-thumb.jpg"), THUMB_Q)
        print(f"    -> {full.size[0]}x{full.size[1]}  {os.path.getsize(path)/1024:.0f}KB")


def upgrade_group_hero():
    print("=== HEADER GROUP ===")
    path = os.path.join(REPO, "hollister.jpg")
    if not os.path.exists(path):
        return
    im = polish(pil_from(path), lo=8, hi=250, exponent=0.97, color=1.04, contrast=1.05)
    # Keep header aspect; bump long edge ~30% from prior ~1200
    out = sharpen(fit(im, 1560), radius=0.75, percent=70)
    save_jpeg(out, path, FULL_Q)
    print(f"  hollister.jpg {out.size[0]}x{out.size[1]}  "
          f"{os.path.getsize(path)/1024:.0f}KB")


def main():
    os.makedirs(OUT, exist_ok=True)
    upgrade_storm()
    upgrade_admin()
    upgrade_departments()
    upgrade_streets_work()
    upgrade_group_hero()
    print("done")


if __name__ == "__main__":
    main()
