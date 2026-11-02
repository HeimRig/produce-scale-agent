# -*- coding: utf-8 -*-
"""抓「非果蔬」负样本，用来测拒识对未知物体的检出率。

源：Caltech-101（HF 镜像）。取背景杂图与常见随身物品，都是顾客可能放到秤上的东西。
    python scripts/fetch_negative.py --per-class 40
"""

import argparse
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

REPO = "HamZurger/Caltech-101"
ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]
UA = "produce-agent/1.0 (+https://github.com/HeimRig/produce-scale-agent)"
SPLITS = ("train", "test")
CLASSES = [
    "BACKGROUND_Google", "cellphone", "watch", "headphone", "laptop", "camera",
    "cup", "scissors", "stapler", "wrench", "umbrella", "dollar_bill",
]
CN = {
    "BACKGROUND_Google": "杂背景", "cellphone": "手机", "watch": "手表",
    "headphone": "耳机", "laptop": "笔记本", "camera": "相机", "cup": "杯子",
    "scissors": "剪刀", "stapler": "订书机", "wrench": "扳手",
    "umbrella": "雨伞", "dollar_bill": "纸币",
}
OUT = ROOT / "data" / "negative"


def api(path):
    url = f"https://hf-mirror.com/api/datasets/{REPO}/tree/main/{urllib.parse.quote(path)}?recursive=true"
    items = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as r:
            items.extend(json.loads(r.read()))
            link = r.headers.get("Link") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return items


def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.0 + i)


def resolve(path):
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(f"{ep}/datasets/{REPO}/resolve/main/{path}",
                                         method="HEAD", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status < 400:
                    return f"{ep}/datasets/{REPO}/resolve/main/{path}"
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    t0 = time.time()
    saved, missing = {}, []

    for cls in CLASSES:
        dst = OUT / cls
        dst.mkdir(parents=True, exist_ok=True)
        have = len(list(dst.glob("*.jpg")))
        if have >= args.per_class:
            saved[cls] = have
            continue

        files = []
        for split in SPLITS:
            try:
                items = api(f"Caltech-101/{split}/{cls}")
            except Exception:
                continue
            files += [i["path"] for i in items if i["type"] == "file"
                      and i["path"].lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]
        if not files:
            missing.append(cls)
            saved[cls] = have
            continue

        random.shuffle(files)
        need = args.per_class - have
        n = have
        for path in files:
            if n >= args.per_class:
                break
            url = resolve(path)
            if url is None:
                continue
            try:
                img = Image.open(io.BytesIO(fetch(url))).convert("RGB")
            except Exception:
                continue
            img.save(dst / f"neg_{cls}_{n:04d}.jpg", quality=92)
            n += 1
        saved[cls] = n

    print(f"落盘 {OUT}")
    for cls in CLASSES:
        flag = "" if saved.get(cls) else "  ← 没抓到"
        print(f"  {cls:<20}{CN.get(cls, ''):<8}{saved.get(cls, 0):>5}{flag}")
    total = sum(saved.values())
    print(f"\n共 {total} 张，耗时 {time.time()-t0:.0f}s")
    if missing:
        print("这些类在两个 split 里都没有可下载文件：", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
