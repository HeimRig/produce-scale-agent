# -*- coding: utf-8 -*-
"""用实拍留出集标定拒识阈值，输出 docs/REJECT_REPORT.md。

留出集里全是"有商品"的正样本，只能算误拒率；错例当作"疑难样本"的代理，
看不同阈值能拦住多少。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import config  # noqa: E402
from app.vision.classifier import ProduceClassifier  # noqa: E402

HOLDOUT = ROOT / "data" / "realshot_holdout"
REPORT = ROOT / "docs" / "REJECT_REPORT.md"
CONFS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MARGINS = [0.10, 0.15, 0.20, 0.25, 0.30]
TARGET_FRR = 0.07     # 误拒=重拍一次，成本低；误判=算错钱，成本高


def collect(clf):
    rows = []
    files = sorted(p for p in HOLDOUT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in files:
        truth = p.parent.name
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        preds = clf.predict(img, topk=3)
        top = preds[0]
        second = preds[1]["prob"] if len(preds) > 1 else 0.0
        rows.append({
            "i": len(rows),
            "file": p.name,
            "truth": truth,
            "pred": top["key"],
            "prob": top["prob"],
            "margin": top["prob"] - second,
            "correct": top["key"] == truth,
        })
    return rows


def evaluate(rows, conf, margin):
    rej = {r["i"] for r in rows if r["prob"] < conf or r["margin"] < margin}
    kept = [r for r in rows if r["i"] not in rej]
    wrong = [r for r in rows if not r["correct"]]
    frr = len([r for r in rows if r["correct"] and r["i"] in rej]) / max(1, len([r for r in rows if r["correct"]]))
    keep_acc = len([r for r in kept if r["correct"]]) / max(1, len(kept))
    catch = len([r for r in wrong if r["i"] in rej]) / max(1, len(wrong))
    return {
        "conf": conf,
        "margin": margin,
        "rejected": len(rej),
        "frr": frr,
        "catch": catch,
        "keep_acc": keep_acc,
        "kept": len(kept),
    }


def main():
    if not HOLDOUT.exists():
        print(f"没有 {HOLDOUT}，先跑 scripts/split_realshot.py")
        return 1

    clf = ProduceClassifier()
    if not clf.ready:
        print("模型不存在，先跑 scripts/finetune.py 和 scripts/export_onnx.py")
        return 1

    rows = collect(clf)
    if not rows:
        print("留出集里没有图片")
        return 1

    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    base_acc = len(correct) / len(rows)
    print(f"留出集 {len(rows)} 张，当前准确率 {base_acc*100:.1f}%，错例 {len(wrong)} 张")

    grid = [evaluate(rows, c, m) for c in CONFS for m in MARGINS]
    feasible = [g for g in grid if g["frr"] <= TARGET_FRR]
    best = max(feasible, key=lambda g: (g["catch"], -g["frr"])) if feasible else min(grid, key=lambda g: g["frr"])

    lines = [
        "# 拒识阈值：只看正样本的扫描",
        "",
        f"- 数据：`data/realshot_holdout`，{len(rows)} 张实拍留出图",
        f"- 当前模型：{config.VISION_MODEL.name}，无拒识时准确率 {base_acc*100:.1f}%",
        f"- 选阈值规则：误拒率 ≤ {TARGET_FRR*100:.0f}% 的前提下，尽量多拦错例",
        "",
        "> 这份只用了正样本，看不到拒识对「根本不是商品」的效果。",
        "> 最终阈值以 `scripts/threshold_curve.py` 的联合标定为准，本文件留作单边参照。",
        "",
        "## 单边最优（参考值）",
        "",
        f"`CONF_THRESHOLD = {best['conf']:.2f}`，`MARGIN_THRESHOLD = {best['margin']:.2f}`",
        "",
        f"- 误拒（正确商品被拦）：{best['frr']*100:.1f}%",
        f"- 拦住的错例：{best['catch']*100:.0f}%（{len(wrong)} 张错例中）",
        f"- 保留样本准确率：{best['keep_acc']*100:.1f}%（{best['kept']} 张）",
        "",
        "## 阈值网格",
        "",
        "| conf | margin | 被拒 | 误拒率 | 拦住错例 | 保留准确率 |",
        "|---|---|---|---|---|---|",
    ]
    for g in grid:
        mark = " ←" if (g["conf"], g["margin"]) == (best["conf"], best["margin"]) else ""
        lines.append(
            f"| {g['conf']:.2f} | {g['margin']:.2f} | {g['rejected']} | "
            f"{g['frr']*100:.1f}% | {g['catch']*100:.0f}% | {g['keep_acc']*100:.1f}%{mark} |"
        )

    lines += [
        "",
        "## 边界",
        "",
        "- 留出集全是正样本，算不出「未知物体检出率」。上表的误拒率是真实值，",
        "  拦住错例的比例只是代理指标：错例和「压根不是商品」不是一回事。",
        "- 要拿到真实的未知检出率，得拍一批非商品图（空托盘、手、手机、桌面）放进",
        "  `data/unknown_test/`，再跑一次本脚本。",
        "",
        "## 错例明细",
        "",
        "| 文件 | 真值 | 预测 | 置信度 | margin |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(wrong, key=lambda x: x["margin"])[:15]:
        lines.append(
            f"| {r['file']} | {r['truth']} | {r['pred']} | {r['prob']*100:.1f}% | {r['margin']*100:.1f} |"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"建议 conf={best['conf']:.2f} margin={best['margin']:.2f}，"
          f"误拒 {best['frr']*100:.1f}%，拦住错例 {best['catch']*100:.0f}%")
    print(f"报告 {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
