# -*- coding: utf-8 -*-
"""data/raw/<类别> → data/produce/{train,val}/<类别>，顺带清洗与统计。

python scripts/prepare_dataset.py --clean
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from app.vision.imageops import Deduper, blur_score

BLUR_MIN = 60.0
DEDUP_HAMMING = 1
SEED = 42
REPORT = ROOT / "docs" / "DATASET_REPORT.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--clean", action="store_true", help="先做模糊/重复剔除")
    ap.add_argument("--min-per-class", type=int, default=40)
    ap.add_argument("--extra", nargs="*", default=None,
                    help="额外数据目录（如 data/realshot 的实拍图），可给多个")
    args = ap.parse_args()

    from PIL import Image

    random.seed(SEED)
    raw = config.RAW_DATA_DIR
    if not raw.exists():
        print(f"没有 {raw}，先跑 scripts/fetch_dataset.py 或 scripts/collect.py")
        return

    import json

    allowed = set(json.loads(config.PRICE_TABLE.read_text(encoding="utf-8"))["items"].keys())

    for split in ("train", "val"):
        stale = config.DATASET_DIR / split
        if stale.exists():
            trash = config.DATASET_DIR / "_trash" / f"{split}_{time.strftime('%Y%m%d_%H%M%S')}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stale), str(trash))
            print(f"旧 {split} 已移到 {trash.relative_to(config.DATASET_DIR)}（确认无用后可手动删）")

    extras = args.extra if args.extra is not None else [str(ROOT / "data" / "realshot")]
    extras = [Path(e) for e in extras if Path(e).exists()]
    if extras:
        print("额外数据源：" + "，".join(str(e.relative_to(ROOT)) for e in extras))

    pooled = {}
    pooled_extra = {}
    skipped = []
    for src in [raw, *extras]:
        bucket = pooled if src == raw else pooled_extra
        for cls_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
            cls = cls_dir.name
            if cls not in allowed:
                if src == raw and cls not in skipped:
                    skipped.append(cls)
                continue
            bucket.setdefault(cls, []).extend(
                sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            )

    rows = []
    for cls in sorted(set(pooled) | set(pooled_extra)):
        files = pooled.get(cls, []) + pooled_extra.get(cls, [])
        kept_raw, kept_extra = [], []
        dropped_blur, dropped_dup = 0, 0
        raw_set = set(pooled.get(cls, []))
        ded = Deduper(threshold=DEDUP_HAMMING, size=16)
        for p in files:
            target = kept_raw if p in raw_set else kept_extra
            if not args.clean:
                target.append(p)
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                dropped_blur += 1
                continue
            if blur_score(img) < BLUR_MIN:
                dropped_blur += 1
                continue
            if ded.is_dup(img):
                dropped_dup += 1
                continue
            target.append(p)

        random.shuffle(kept_raw)
        kept = kept_raw + kept_extra
        n_val = max(1, int(len(kept_raw) * args.val_ratio)) if kept_raw else 0
        train_subset = kept_raw[n_val:] + kept_extra
        for split, subset in (("val", kept_raw[:n_val]), ("train", train_subset)):
            dst = config.DATASET_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for p in subset:
                shutil.copy2(p, dst / p.name)

        rows.append((cls, len(files), len(kept), dropped_blur, dropped_dup, len(train_subset), n_val))

    total_train = sum(r[5] for r in rows)
    total_val = sum(r[6] for r in rows)
    lines = [
        "# 数据集统计",
        "",
        f"- 生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 清洗：{'开' if args.clean else '关'}（模糊阈值 {BLUR_MIN}，重复汉明距离 ≤{DEDUP_HAMMING}，16×16 哈希）",
        f"- 划分：val {args.val_ratio:.0%}",
        "",
        "| 类别 | 原始 | 保留 | 剔除模糊 | 剔除重复 | train | val |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, n0, n1, dblur, ddup, ntr, nva in rows:
        flag = " ⚠️" if n1 < args.min_per_class else ""
        lines.append(f"| {cls}{flag} | {n0} | {n1} | {dblur} | {ddup} | {ntr} | {nva} |")
    lines += [
        "",
        f"**合计** train {total_train} / val {total_val}",
        "",
        f"低于 {args.min_per_class} 张的类别已标 ⚠️，建议补采或从公开数据集补齐。",
    ]
    if skipped:
        lines.append("")
        lines.append(f"已跳过不在价目表里的类别目录：{', '.join(skipped)}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
