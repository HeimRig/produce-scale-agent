# -*- coding: utf-8 -*-
"""从网上下载蔬果训练图集，按本项目 12 个类别落入 data/raw/<类别>/。

默认源 Fruits-360（HF 镜像，113 类 9 万张，白底棚拍，适合冷启动）：
    python scripts/fetch_dataset.py --source fruits360 --per-class 300

真实背景小样（19 类每类 30 张，仅 7 类有图，用于混入少量实景）：
    python scripts/fetch_dataset.py --source fruidet --per-class 30

实拍图（36 类真实背景，落到 data/realshot/，只做验证集不进训练）：
    python scripts/fetch_dataset.py --source realshot --split test --per-class 40
"""

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]
CACHE = ROOT / "data" / "_cache"
OUT_OVERRIDE = None

SOURCES = {
    "fruits360": {
        "repo": "PedroSampaio/fruits-360",
        "files": {
            "train": "data/train-00000-of-00001-086dc013443c484c.parquet",
            "test": "data/test-00000-of-00001-0d294abe3826b2e6.parquet",
        },
        "kind": "parquet",
        "prefix": "f360",
        "out": "raw",
    },
    "fruidet": {
        "repo": "sirunchained/fruit-dataset",
        "kind": "imagefolder",
        "prefix": "fdet",
        "out": "raw",
    },
    "realshot": {
        "repo": "Nattakarn/fruit-and-vegetable-image-recognition",
        "files": {"test": "data/test-00000-of-00001.parquet"},
        "kind": "parquet",
        "prefix": "real",
        "out": "realshot",
    },
    "realshot2": {
        "repo": "Nattakarn/fruit-and-vegetable-image-recognition",
        "files": {
            "train": [
                "data/train-00000-of-00003.parquet",
                "data/train-00001-of-00003.parquet",
                "data/train-00002-of-00003.parquet",
            ]
        },
        "kind": "parquet",
        "prefix": "real2",
        "out": "realshot_extra",
    },
}

TARGETS = [
    "apple", "avocado", "banana", "carrot", "cucumber", "grape",
    "onion", "orange", "pear", "pepper", "strawberry", "tomato",
]

RULES_NOTE = "类别映射见 map_class()"

SKIP_WORDS = ["grapefruit", "sweet potato", "sweetpotato"]

PREFIX_MAP = [
    ("apple", "apple"),
    ("avocado", "avocado"),
    ("banana", "banana"),
    ("pear", "pear"),
    ("grape", "grape"),
    ("onion", "onion"),
    ("pepper", "pepper"),
    ("tomato", "tomato"),
    ("cucumber", "cucumber"),
    ("strawberry", "strawberry"),
    ("carrot", "carrot"),
]

EXACT_MAP = {
    "orange": "orange",
    "mandarine": "orange",
    "tangelo": "orange",
    "clementine": "orange",
    "grapes": "grape",
    "bell pepper": "pepper",
    "capsicum": "pepper",
    "paprika": "pepper",
    "chilli pepper": "pepper",
    "chili pepper": "pepper",
}


def map_class(name):
    low = name.strip().lower()
    for word in SKIP_WORDS:
        if word in low:
            return None
    if low in EXACT_MAP:
        return EXACT_MAP[low]
    for prefix, target in PREFIX_MAP:
        if low.startswith(prefix):
            return target
    return None


def label_names(pf):
    import json

    md = (pf.metadata.metadata or {}).get(b"huggingface")
    if not md:
        return None
    try:
        info = json.loads(md.decode("utf-8", "ignore"))
        return info["info"]["features"]["label"]["names"]
    except Exception:
        return None


def download(url, dest, retries=3):
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if total:
                            pct = have / total * 100
                            print(f"\r  {home(dest)} {have/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="")
            print()
            return dest
        except Exception as e:
            print(f"\n  下载失败（第 {attempt+1} 次）：{e}")
            time.sleep(2)
            have = dest.stat().st_size if dest.exists() else 0
    raise RuntimeError(f"下载失败：{url}")


def home(p):
    return p.name


def resolve_url(repo, path):
    for ep in ENDPOINTS:
        url = f"{ep}/datasets/{repo}/resolve/main/{path}"
        try:
            import urllib.request

            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status < 400:
                    return url
        except Exception:
            continue
    raise RuntimeError(f"{repo}/{path} 两个源都不可用，检查网络或代理")


def fetch_parquet(src_key, split, per_class, seed):
    from PIL import Image
    import io
    import pyarrow.parquet as pq

    cfg = SOURCES[src_key]
    rels = cfg["files"][split]
    if isinstance(rels, str):
        rels = [rels]

    dests = []
    for rel in rels:
        dest = CACHE / f"{src_key}_{rel.split('/')[-1]}"
        if not dest.exists():
            url = resolve_url(cfg["repo"], rel)
            print(f"下载 {cfg['repo']} [{rel.split('/')[-1]}]")
            download(url, dest)
        else:
            print(f"缓存命中 {dest.name}")
        dests.append(dest)

    out_root = Path(OUT_OVERRIDE) if OUT_OVERRIDE else (
        config.RAW_DATA_DIR if cfg["out"] == "raw" else ROOT / "data" / cfg["out"]
    )
    prefix = cfg["prefix"]

    buckets = {t: [] for t in TARGETS}
    label_stat = {}
    for dest in dests:
        pf = pq.ParquetFile(dest)
        names = label_names(pf)
        print(f"{dest.name}: {pf.metadata.num_rows} 行，类别数：{len(names) if names else '未知'}")
        cols = ["image", "label"] if "label" in pf.schema_arrow.names else ["image", "labels"]
        label_col = cols[1]

        for batch in pf.iter_batches(batch_size=256, columns=cols):
            imgs = batch.column("image").to_pylist()
            labs = batch.column(label_col).to_pylist()
            for im, lb in zip(imgs, labs):
                if isinstance(lb, int):
                    name = names[lb] if names and lb < len(names) else str(lb)
                else:
                    name = str(lb)
                tgt = map_class(name)
                if tgt is not None:
                    label_stat[name] = label_stat.get(name, 0) + 1
                if tgt is None or len(buckets[tgt]) >= per_class * 3:
                    continue
                buckets[tgt].append((im, name))
        if all(len(v) >= per_class * 3 for v in buckets.values()):
            break

    random.seed(seed)
    saved = {}
    for tgt, items in buckets.items():
        random.shuffle(items)
        out_dir = out_root / tgt
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for im, name in items[:per_class]:
            raw = im.get("bytes") if isinstance(im, dict) else im
            if not raw:
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            img.save(out_dir / f"{prefix}_{tgt}_{n:04d}.jpg", quality=92)
            n += 1
        saved[tgt] = n

    print("\n原始类别样本（前 20）：")
    for k, v in sorted(label_stat.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k}: {v}")
    return saved


def fetch_imagefolder(per_class, seed, only=None):
    import json
    import urllib.request

    cfg = SOURCES["fruidet"]
    repo = cfg["repo"]
    api = None
    for ep in ENDPOINTS:
        try:
            with urllib.request.urlopen(f"{ep}/api/datasets/{repo}/tree/main?recursive=true", timeout=20) as r:
                api = json.loads(r.read())
            break
        except Exception:
            continue
    if api is None:
        if not only:
            raise RuntimeError("拿不到文件列表")
        # API 不可用但知道命名规则：{cls}/{cls}_0000.jpg 递增
        files = [f"data/{c}/{c}_{i:04d}.jpg" for c in sorted(only) for i in range(per_class)]
    else:
        files = [x["path"] for x in api if x["path"].lower().endswith((".jpg", ".jpeg", ".png"))]
    random.seed(seed)
    random.shuffle(files)

    saved = {}
    misses = 0
    for p in files:
        cls_dir = p.split("/")[-2]
        tgt = map_class(cls_dir)
        if tgt is None or (only and tgt not in only):
            continue
        if saved.get(tgt, 0) >= per_class:
            continue
        try:
            url = resolve_url(repo, p)
        except Exception:
            misses += 1
            if api is None and misses > per_class * 2:
                break
            continue
        out_dir = (Path(OUT_OVERRIDE) if OUT_OVERRIDE else config.RAW_DATA_DIR) / tgt
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}), timeout=30) as r:
                (out_dir / f"fdet_{tgt}_{saved.get(tgt,0):04d}.jpg").write_bytes(r.read())
            saved[tgt] = saved.get(tgt, 0) + 1
            print(f"\r  {tgt}: {saved[tgt]}/{per_class}", end="")
        except Exception as e:
            print(f"\n  跳过 {p}: {e}")
    print()
    return saved


def main():
    global OUT_OVERRIDE

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="fruits360")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None, help="覆盖落盘目录，例如 data/realshot_extra")
    ap.add_argument("--only", default=None, help="只抓这些类别，逗号分隔，如 avocado,strawberry")
    args = ap.parse_args()

    only = set(x.strip() for x in args.only.split(",")) if args.only else None

    if args.out_dir:
        OUT_OVERRIDE = Path(args.out_dir)
        if not OUT_OVERRIDE.is_absolute():
            OUT_OVERRIDE = ROOT / OUT_OVERRIDE

    t0 = time.time()
    if args.source == "fruidet":
        saved = fetch_imagefolder(args.per_class, args.seed, only)
        dest_desc = str(OUT_OVERRIDE or config.RAW_DATA_DIR)
    else:
        saved = fetch_parquet(args.source, args.split, args.per_class, args.seed)
        cfg = SOURCES[args.source]
        dest_desc = str(OUT_OVERRIDE or (config.RAW_DATA_DIR if cfg["out"] == "raw" else ROOT / "data" / cfg["out"]))

    print(f"\n落盘到 {dest_desc}")
    for t in TARGETS:
        flag = "" if saved.get(t) else "  ← 该类无图，需自采或换源"
        print(f"  {t:<12} {saved.get(t, 0):>5}{flag}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
