# -*- coding: utf-8 -*-
"""把 data/realshot 里 30% 抽出来做留出集 data/realshot_holdout，其余留作训练用实拍图。

    python scripts/split_realshot.py --holdout-ratio 0.3
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "realshot"
HOLDOUT = ROOT / "data" / "realshot_holdout"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-ratio", type=float, default=0.3)
    args = ap.parse_args()

    if not SRC.exists():
        print(f"没有 {SRC}，先跑 fetch_dataset.py --source realshot")
        return

    random.seed(SEED)
    total = 0
    for cls_dir in sorted([p for p in SRC.iterdir() if p.is_dir()]):
        files = sorted(cls_dir.glob("*.jpg"))
        if not files:
            continue
        random.shuffle(files)
        n = max(1, int(len(files) * args.holdout_ratio))
        dst = HOLDOUT / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in files[:n]:
            shutil.copy2(f, dst / f.name)
        total += n
        print(f"{cls_dir.name:<12} 留出 {n}/{len(files)}")
    print(f"\n共留出 {total} 张 → {HOLDOUT}")
    print("训练用的实拍图仍在 data/realshot，由 prepare_dataset.py --extra 混入")


if __name__ == "__main__":
    main()
