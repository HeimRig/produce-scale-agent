# -*- coding: utf-8 -*-
"""对比同一模型在「白底验证集」与「实拍图」上的准确率，输出 docs/REALSHOT_REPORT.md。

    python scripts/eval_realshot.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

import config
from app.vision.classifier import ProduceClassifier

WHITE_DIR = config.DATASET_DIR / "val"
REAL_DIR = ROOT / "data" / "realshot_holdout"
REPORT = ROOT / "docs" / "REALSHOT_REPORT.md"
SHOW_MISS = 8


def collect(root):
    items = []
    if not root.exists():
        return items
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        for f in sorted(d.glob("*.jpg")):
            items.append((f, d.name))
    return items


def evaluate(clf, items):
    total = correct = 0
    per = {}
    miss = []
    for f, label in items:
        from PIL import Image

        try:
            preds = clf.predict(Image.open(f), topk=1)
        except Exception:
            continue
        pred = preds[0]["key"]
        conf = preds[0]["prob"]
        total += 1
        ok = pred == label
        correct += int(ok)
        st = per.setdefault(label, {"n": 0, "ok": 0})
        st["n"] += 1
        st["ok"] += int(ok)
        if not ok and len(miss) < SHOW_MISS:
            miss.append((f.name, label, pred, conf))
    return correct, total, per, miss


def main():
    clf = ProduceClassifier()
    if not clf.ready:
        print(f"没有模型：{clf.model_path}")
        return
    print(f"模型 {clf.model_path.name}，类别 {len(clf.labels)}")

    white = collect(WHITE_DIR)
    real = collect(REAL_DIR)
    print(f"白底验证集 {len(white)} 张，实拍图 {len(real)} 张")

    wc, wt, wper, _ = evaluate(clf, white)
    rc, rt, rper, rmiss = evaluate(clf, real)

    lines = [
        "# 实拍泛化测试",
        "",
        f"- 模型：`{clf.model_path.name}`（{len(clf.labels)} 类，MobileNetV3-Small）",
        "- 白底集：`data/produce/val`（Fruits-360，与训练同源）",
        "- 实拍集：`data/realshot_holdout`（真实背景，30% 留出，未参与训练）",
        "",
        "| 数据集 | 张数 | Top-1 |",
        "|---|---|---|",
        f"| 白底验证集 | {wt} | {wc/max(1,wt)*100:.1f}% |",
        f"| 实拍图 | {rt} | {rc/max(1,rt)*100:.1f}% |",
        "",
        "## 分类别",
        "",
        "| 类别 | 白底 | 实拍 |",
        "|---|---|---|",
    ]
    for k in sorted(set(wper) | set(rper)):
        w = wper.get(k)
        r = rper.get(k)
        w_txt = f"{w['ok']}/{w['n']}" if w else "—"
        r_txt = f"{r['ok']}/{r['n']}" if r else "—"
        lines.append(f"| {k} | {w_txt} | {r_txt} |")

    if rmiss:
        lines += ["", "## 实拍误判样例", "", "| 文件 | 真实 | 预测 | 置信度 |", "|---|---|---|---|"]
        for f, gt, pred, conf in rmiss:
            lines.append(f"| {f} | {gt} | {pred} | {conf*100:.0f}% |")

    lines += [
        "",
        "> 实拍集从同一数据源留出 30%，与训练侧的实拍图同分布但不同图片；",
        "> 它不等于「自己拿手机在托盘上拍的」，后者才是最终验收口径。",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
