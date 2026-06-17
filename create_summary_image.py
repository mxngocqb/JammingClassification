# make_label_pdf.py
import os
import glob
import math
import random
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageOps

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def pick_one_image_per_class(split_dir: str, seed: int = 0):
    """
    split_dir/a
      classA/ img1.png ...
      classB/ ...
    Returns: list of (class_name, image_path)
    """
    random.seed(seed)
    classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    classes.sort()

    picked = []
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
            files.extend(glob.glob(os.path.join(cls_dir, f"*{ext.upper()}")))
        files.sort()
        if not files:
            raise FileNotFoundError(f"Không tìm thấy ảnh trong class: {cls_dir}")
        path = random.choice(files)
        picked.append((cls, path))
    return picked


def autocrop_white(img: Image.Image, thr: int = 245) -> Image.Image:
    """
    Cắt viền trắng dư (nếu ảnh có nền trắng).
    thr càng cao càng "nhạy" cắt.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < thr else 0)
    bbox = mask.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def load_font(size: int = 28):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def get_resample():
    # Pillow mới: Image.Resampling.LANCZOS ; Pillow cũ: Image.LANCZOS
    Resampling = getattr(Image, "Resampling", Image)
    return getattr(Resampling, "LANCZOS", Image.LANCZOS)


def make_grid_pdf(
    items,
    out_pdf,
    cols=None,
    tile_w=900,
    tile_h=240,
    label_h=45,
    pad=2,
    gap=2,
    crop_white=True,
    crop_thr=245,
    dpi=300,
    fit_mode="cover",  # "cover" (full) hoặc "contain"
):
    n = len(items)
    if cols is None:
        cols = 2 if n == 6 else max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    W = cols * tile_w + (cols + 1) * pad
    H = rows * (tile_h + label_h + gap) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    font = load_font(size=max(16, int(label_h * 0.55)))
    resample = get_resample()

    for idx, (cls, path) in enumerate(items):
        r = idx // cols
        c = idx % cols

        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + label_h + gap + pad)

        # ---- Label ----
        text = cls
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x0 + max(0, (tile_w - tw) // 2)
        ty = y0 + max(0, (label_h - th) // 2)
        draw.text((tx, ty), text, fill="black", font=font)

        # ---- Image ----
        img = Image.open(path).convert("RGB")
        if crop_white:
            img = autocrop_white(img, thr=crop_thr)

        paste_x = x0
        paste_y = y0 + label_h + gap

        if fit_mode == "cover":
            # FULL: ảnh phủ kín tile, crop nhẹ phần thừa
            img = ImageOps.fit(img, (tile_w, tile_h), method=resample, centering=(0.5, 0.5))
            canvas.paste(img, (paste_x, paste_y))
        else:
            # CONTAIN: giữ toàn ảnh, có thể còn khoảng trắng
            img = ImageOps.contain(img, (tile_w, tile_h), method=resample)
            ix = paste_x + (tile_w - img.size[0]) // 2
            iy = paste_y  # top-align sát label
            canvas.paste(img, (ix, iy))

    canvas.save(out_pdf, "PDF", resolution=float(dpi))
    print(f"✅ Saved grid PDF: {out_pdf} ({cols}x{rows}, fit={fit_mode}, pad={pad}, gap={gap})")


def make_pages_pdf(
    items,
    out_pdf,
    page_w=1654,
    page_h=2339,
    crop_white=True,
    crop_thr=245,
    dpi=300,
    fit_mode="contain",  # pages thường để contain; muốn full thì cover
):
    pages = []
    font = load_font(size=44)
    resample = get_resample()

    for cls, path in items:
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)

        # title
        bbox = draw.textbbox((0, 0), cls, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((page_w - tw) // 2, 60), cls, fill="black", font=font)

        img = Image.open(path).convert("RGB")
        if crop_white:
            img = autocrop_white(img, thr=crop_thr)

        margin = 120
        top = 160
        max_w = page_w - 2 * margin
        max_h = page_h - top - margin

        if fit_mode == "cover":
            img = ImageOps.fit(img, (max_w, max_h), method=resample, centering=(0.5, 0.5))
            ix, iy = margin, top
        else:
            img = ImageOps.contain(img, (max_w, max_h), method=resample)
            ix = (page_w - img.size[0]) // 2
            iy = top + (max_h - img.size[1]) // 2

        page.paste(img, (ix, iy))
        pages.append(page)

    pages[0].save(out_pdf, "PDF", save_all=True, append_images=pages[1:], resolution=float(dpi))
    print(f"✅ Saved pages PDF: {out_pdf} (pages={len(pages)}, fit={fit_mode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="Image_Dataset_Classifier", help="Thư mục root dataset")
    ap.add_argument("--split", default="Image_testing_database", help="Tên thư mục split bên trong data_root")
    ap.add_argument("--out", default="spectrogram.pdf", help="File PDF output")

    ap.add_argument("--mode", choices=["grid", "pages"], default="grid",
                    help="grid: 1 trang lưới | pages: mỗi label 1 trang")
    ap.add_argument("--cols", type=int, default=None, help="Số cột (grid). VD: 2")
    ap.add_argument("--pad", type=int, default=2, help="Padding giữa các ô (px) (grid)")
    ap.add_argument("--gap", type=int, default=2, help="Khoảng cách giữa label và ảnh (px) (grid)")
    ap.add_argument("--tile_w", type=int, default=1000, help="Rộng ô ảnh (grid)")
    ap.add_argument("--tile_h", type=int, default=400, help="Cao ô ảnh (grid)")
    ap.add_argument("--label_h", type=int, default=45, help="Chiều cao label (grid)")

    ap.add_argument("--fit", choices=["contain", "cover"], default="cover",
                    help="contain: giữ toàn ảnh | cover: FULL ô (có crop nhẹ)")
    ap.add_argument("--no_crop_white", action="store_true", help="Tắt autocrop viền trắng")
    ap.add_argument("--crop_thr", type=int, default=245, help="Ngưỡng cắt trắng (0..255), cao=cắt mạnh")
    ap.add_argument("--seed", type=int, default=0, help="Seed để chọn ảnh ngẫu nhiên")

    args = ap.parse_args()

    split_dir = os.path.join(args.data_root, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Không thấy split_dir: {split_dir}")

    items = pick_one_image_per_class(split_dir, seed=args.seed)
    crop_white = (not args.no_crop_white)

    if args.mode == "grid":
        make_grid_pdf(
            items,
            args.out,
            cols=args.cols,
            tile_w=args.tile_w,
            tile_h=args.tile_h,
            label_h=args.label_h,
            pad=args.pad,
            gap=args.gap,
            crop_white=crop_white,
            crop_thr=args.crop_thr,
            fit_mode=args.fit,
        )
    else:
        make_pages_pdf(
            items,
            args.out,
            crop_white=crop_white,
            crop_thr=args.crop_thr,
            fit_mode=args.fit,
        )


if __name__ == "__main__":
    main()
