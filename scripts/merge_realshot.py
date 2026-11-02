# -*- coding: utf-8 -*-
"""把新下载的实拍图按比例并进现有实拍集：
30% 进 data/realshot_holdout（验收集，不动原来的），其余进 data/realshot（训练用）。

    python scripts/merge_realshot.py --src data/realshot_extra --holdout-ratio 0.3
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "realshot"
HOLDOUT = ROOT / "data" / "realshot_holdout"
SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "realshot_extra"))
    ap.add_argument("--holdout-ratio", type=float, default=0.3)
    ap.add_argument("--move", action="store_true", help="搬走而不是复制")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"没有 {src}")
        return

    random.seed(SEED)
    rows = []
    for cls_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        files = sorted(cls_dir.glob("*.jpg"))
        if not files:
            continue
        random.shuffle(files)
        n_hold = max(1, int(len(files) * args.holdout_ratio))
        tr_dir = TRAIN / cls_dir.name
        ho_dir = HOLDOUT / cls_dir.name
        tr_dir.mkdir(parents=True, exist_ok=True)
        ho_dir.mkdir(parents=True, exist_ok=True)

        for f in files[n_hold:]:
            dst = tr_dir / f.name
            if dst.exists():
                continue
            shutil.move(str(f), dst) if args.move else shutil.copy2(f, dst)
        for f in files[:n_hold]:
            dst = ho_dir / f.name
            if dst.exists():
                continue
            shutil.move(str(f), dst) if args.move else shutil.copy2(f, dst)
        rows.append((cls_dir.name, len(files), len(files) - n_hold, n_hold))

    print(f"{'类别':<12}{'新增':>6}{'训练':>8}{'验收集':>8}")
    for name, total, tr, ho in rows:
        print(f"{name:<12}{total:>6}{tr:>8}{ho:>8}")
    print(f"\n合计新增 {sum(r[1] for r in rows)} 张")


if __name__ == "__main__":
    main()
