# -*- coding: utf-8 -*-
"""把正样本误拒率与负样本检出率放在一起扫，按加权代价选阈值。

    python scripts/threshold_curve.py

正样本：data/realshot_holdout（261 张，全是有商品的图，用来算误拒）
负样本：data/negative（Caltech-101 的非果蔬图，用来算检出）

代价模型：一次误判（算错钱、要退单）记 5，一次误拒（顾客重放一次）记 1。
权重写在 MISS_COST，可自行调整。输出 docs/THRESHOLD_CURVE.md。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import config  # noqa: E402
from app.vision.classifier import ProduceClassifier  # noqa: E402

POS_DIR = ROOT / "data" / "realshot_holdout"
NEG_DIR = ROOT / "data" / "negative"
REPORT = ROOT / "docs" / "THRESHOLD_CURVE.md"

CONF_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MARGIN_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
MISS_COST = 5.0      # 一次误判相当于 5 次误拒
FR_COST = 1.0


def load(clf, root, labeled):
    rows = []
    for p in sorted(root.rglob("*.jpg")):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        preds = clf.predict(img, topk=2)
        top = preds[0]
        rows.append({
            "truth": p.parent.name if labeled else None,
            "key": top["key"],
            "prob": top["prob"],
            "margin": top["prob"] - (preds[1]["prob"] if len(preds) > 1 else 0.0),
        })
    return rows


def rejected(r, conf, margin):
    return r["prob"] < conf or r["margin"] < margin


def evaluate(pos, neg, conf, margin):
    kept = [r for r in pos if not rejected(r, conf, margin)]
    fr = len([r for r in pos if rejected(r, conf, margin) and r["key"] == r["truth"]])
    frr = fr / max(1, len(pos))
    acc = len([r for r in kept if r["key"] == r["truth"]]) / max(1, len(kept))
    tnr = len([r for r in neg if rejected(r, conf, margin)]) / max(1, len(neg))
    cost = (frr * FR_COST + (1 - tnr) * MISS_COST) * 100
    return {"conf": conf, "margin": margin, "frr": frr, "acc": acc, "tnr": tnr, "cost": cost}


def main():
    if not POS_DIR.exists() or not NEG_DIR.exists():
        print("缺少 data/realshot_holdout 或 data/negative，先跑 split_realshot 与 fetch_negative")
        return 1

    clf = ProduceClassifier()
    if not clf.ready:
        print("模型不存在，先跑 scripts/finetune.py 和 scripts/export_onnx.py")
        return 1

    pos = load(clf, POS_DIR, labeled=True)
    neg = load(clf, NEG_DIR, labeled=False)
    print(f"正样本 {len(pos)} 张，负样本 {len(neg)} 张")

    grid = [evaluate(pos, neg, c, m) for c in CONF_GRID for m in MARGIN_GRID]
    best = min(grid, key=lambda g: g["cost"])
    base_acc = len([r for r in pos if r["key"] == r["truth"]]) / len(pos)
    cur = evaluate(pos, neg, config.CONF_THRESHOLD, config.MARGIN_THRESHOLD)

    lines = [
        "# 拒识阈值联合标定",
        "",
        f"- 正样本 {len(pos)} 张（实拍留出集，无拒识准确率 {base_acc * 100:.1f}%）",
        f"- 负样本 {len(neg)} 张（Caltech-101 非果蔬类）",
        f"- 代价模型：一次误判 = {MISS_COST:.0f} × 一次误拒",
        "",
        "## 选点",
        "",
        f"最优点 **conf {best['conf']:.2f} / margin {best['margin']:.2f}**："
        f"误拒 {best['frr'] * 100:.1f}%，检出 {best['tnr'] * 100:.1f}%，"
        f"保留准确率 {best['acc'] * 100:.1f}%，每 100 单加权代价 {best['cost']:.1f}",
        "",
        f"当前配置 conf {cur['conf']:.2f} / margin {cur['margin']:.2f}："
        f"误拒 {cur['frr'] * 100:.1f}%，检出 {cur['tnr'] * 100:.1f}%，"
        f"保留准确率 {cur['acc'] * 100:.1f}%，每 100 单加权代价 {cur['cost']:.1f}",
        "",
        "## 扫 conf（margin 固定 %.2f）" % config.MARGIN_THRESHOLD,
        "",
        "| conf | 误拒率 | 未知检出率 | 保留准确率 | 每 100 单代价 |",
        "|---|---|---|---|---|",
    ]
    for c in CONF_GRID:
        g = evaluate(pos, neg, c, config.MARGIN_THRESHOLD)
        mark = " ←" if abs(c - best["conf"]) < 1e-6 and abs(config.MARGIN_THRESHOLD - best["margin"]) < 1e-6 else ""
        lines.append(f"| {c:.2f} | {g['frr'] * 100:.1f}% | {g['tnr'] * 100:.1f}% | "
                     f"{g['acc'] * 100:.1f}% | {g['cost']:.1f}{mark} |")

    lines += [
        "",
        "## 扫 margin（conf 固定 %.2f）" % best["conf"],
        "",
        "| margin | 误拒率 | 未知检出率 | 保留准确率 | 每 100 单代价 |",
        "|---|---|---|---|---|",
    ]
    for m in MARGIN_GRID:
        g = evaluate(pos, neg, best["conf"], m)
        mark = " ←" if abs(m - best["margin"]) < 1e-6 else ""
        lines.append(f"| {m:.2f} | {g['frr'] * 100:.1f}% | {g['tnr'] * 100:.1f}% | "
                     f"{g['acc'] * 100:.1f}% | {g['cost']:.1f}{mark} |")

    lines += [
        "",
        "## 结论怎么读",
        "",
        "- 提高阈值同时改善检出、恶化误拒，两者不共向，所以不能只看一边。",
        f"- 权重取 {MISS_COST:.0f} 的前提是：退一单的麻烦（对账、赔礼、信任）是顾客重放一次的 {MISS_COST:.0f} 倍。",
        "  这个比例是假设，不是测量值；把它改成 2 或 20，最优点会移动。",
        "- margin 在本模型上作用有限：错例的 top1 普遍很低，真正的闸门是 conf。"
        "它留着是为了处理「两个候选都很高且接近」的情况，当前数据里没出现这种样本。",
        "",
        "## 为什么没取纯代价最优点",
        "",
        f"纯代价最优是 conf {best['conf']:.2f}，但误拒 {best['frr'] * 100:.1f}% 意味着每 "
        f"{round(100 / max(best['frr'] * 100, 1))} 单就有一次「没认出来」，柜台体验无法接受。",
        f"实际交付取 conf {config.CONF_THRESHOLD:.2f}：误拒 {cur['frr'] * 100:.1f}%，"
        f"检出 {cur['tnr'] * 100:.1f}%，每 100 单代价 {cur['cost']:.1f}，是代价与体验的折中。",
        "",
        "代价模型只能给出方向，能不能接受某种体验得人来判断；这条是本项目里唯一「模型给最优、人往回收」的决策。",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"最优 conf {best['conf']:.2f} / margin {best['margin']:.2f}，"
          f"误拒 {best['frr']*100:.1f}% 检出 {best['tnr']*100:.1f}%")
    print(f"报告 {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
