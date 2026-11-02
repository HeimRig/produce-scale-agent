# -*- coding: utf-8 -*-
"""采集蔬果图片到 data/raw/<类别>/，自动去重与剔除模糊帧。

摄像头采集：python scripts/collect.py --source webcam --classes apple banana
从文件夹导入：python scripts/collect.py --source dir --dir D:\\pics --class apple
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from app.vision.imageops import Deduper, blur_score

BLUR_MIN = 60.0
MAX_PER_CLASS = 400
JPEG_QUALITY = 92


def _save(image, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}_{len(list(out_dir.glob('*.jpg'))):04d}.jpg"
    image.convert("RGB").save(out_dir / name, quality=JPEG_QUALITY, optimize=True)
    return out_dir / name


def run_webcam(classes):
    import cv2

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        print("摄像头打不开")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_SIZE[1])

    dedupers = {c: Deduper(5) for c in classes}
    counts = {c: len(list((config.RAW_DATA_DIR / c).glob("*.jpg"))) for c in classes}
    for c, d in dedupers.items():
        for p in sorted((config.RAW_DATA_DIR / c).glob("*.jpg"))[-80:]:
            from PIL import Image

            d.is_dup(Image.open(p))

    cur = 0
    last_status = ""
    print("数字键切类别，空格拍照，q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"[{classes[cur]}] {counts[classes[cur]]}/{MAX_PER_CLASS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, last_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key < ord("1") + len(classes):
            cur = key - ord("1")
            last_status = f"switch -> {classes[cur]}"
        if key == 32:
            from PIL import Image

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cls = classes[cur]
            if counts[cls] >= MAX_PER_CLASS:
                last_status = "该类已满"
                continue
            if blur_score(img) < BLUR_MIN:
                last_status = "太糊，重拍"
                continue
            if dedupers[cls].is_dup(img):
                last_status = "重复帧，跳过"
                continue
            p = _save(img, config.RAW_DATA_DIR / cls, cls)
            counts[cls] += 1
            last_status = f"saved {p.name}"
    cap.release()
    cv2.destroyAllWindows()
    print(counts)


def run_dir(src, cls, out_root=None):
    from PIL import Image

    out_root = Path(out_root) if out_root else config.RAW_DATA_DIR
    src = Path(src)
    files = sorted([p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    print(f"发现 {len(files)} 张")
    ded = Deduper(5)
    kept = skipped_blur = skipped_dup = 0
    for p in files:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        if blur_score(img) < BLUR_MIN:
            skipped_blur += 1
            continue
        if ded.is_dup(img):
            skipped_dup += 1
            continue
        _save(img, out_root / cls, cls)
        kept += 1
    print(f"保留 {kept}，模糊剔除 {skipped_blur}，重复剔除 {skipped_dup} → {out_root / cls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "dir"], default="webcam")
    ap.add_argument("--classes", nargs="*", default=["apple", "avocado", "banana", "carrot",
                                                     "cucumber", "grape", "onion", "orange",
                                                     "pear", "pepper", "strawberry", "tomato"])
    ap.add_argument("--class", dest="one_class", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None,
                    help="落盘根目录，默认 data/raw；想当验收集就传 data/realshot_holdout")
    args = ap.parse_args()

    if args.source == "dir":
        if not args.one_class or not args.dir:
            print("--source dir 需要同时给 --dir 与 --class")
            return
        run_dir(args.dir, args.one_class, args.out)
    else:
        run_webcam(args.classes[:9])


if __name__ == "__main__":
    main()
