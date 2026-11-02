# -*- coding: utf-8 -*-
"""测拒识对「根本不是商品」的物体能拦住多少，输出 docs/NEGATIVE_REPORT.md。

数据来自 scripts/fetch_negative.py 抓的 Caltech-101（背景杂图 + 随身物品）。
留出集里的错例不能回答这个问题，这里的图才是真正不含目标商品的画面。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import config  # noqa: E402
from app.vision.classifier import ProduceClassifier  # noqa: E402

NEG_DIR = ROOT / "data" / "negative"
REPORT = ROOT / "docs" / "NEGATIVE_REPORT.md"
CN = {
    "BACKGROUND_Google": "杂背景", "cellphone": "手机", "watch": "手表",
    "headphone": "耳机", "laptop": "笔记本", "camera": "相机", "cup": "杯子",
    "scissors": "剪刀", "stapler": "订书机", "wrench": "扳手",
    "umbrella": "雨伞", "dollar_bill": "纸币",
}
CONF_GRID = [0.45, 0.55, 0.65]


def collect(clf):
    rows = []
    for p in sorted(NEG_DIR.rglob("*.jpg")):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        preds = clf.predict(img, topk=3)
        top = preds[0]
        second = preds[1]["prob"] if len(preds) > 1 else 0.0
        rows.append({
            "cls": p.parent.name,
            "pred": top["key"],
            "pred_cn": top["cn"],
            "prob": top["prob"],
            "margin": top["prob"] - second,
        })
    return rows


def rate(rows, conf, margin):
    rej = [r for r in rows if r["prob"] < conf or r["margin"] < margin]
    return len(rej) / max(1, len(rows)), rej


def main():
    if not NEG_DIR.exists() or not any(NEG_DIR.rglob("*.jpg")):
        print(f"没有 {NEG_DIR}，先跑 scripts/fetch_negative.py")
        return 1

    clf = ProduceClassifier()
    if not clf.ready:
        print("模型不存在，先跑 scripts/finetune.py 和 scripts/export_onnx.py")
        return 1

    rows = collect(clf)
    if not rows:
        print("负样本读不出来")
        return 1

    now, rejected = rate(rows, config.CONF_THRESHOLD, config.MARGIN_THRESHOLD)

    by_cls = {}
    for r in rows:
        slot = by_cls.setdefault(r["cls"], {"n": 0, "rej": 0})
        slot["n"] += 1
        if r["prob"] < config.CONF_THRESHOLD or r["margin"] < config.MARGIN_THRESHOLD:
            slot["rej"] += 1

    guesses = {}
    for r in rows:
        if r["prob"] < config.CONF_THRESHOLD or r["margin"] < config.MARGIN_THRESHOLD:
            continue
        guesses[r["pred_cn"]] = guesses.get(r["pred_cn"], 0) + 1
    top_guesses = sorted(guesses.items(), key=lambda x: -x[1])[:6]

    lines = [
        "# 负样本拒识报告",
        "",
        f"- 负样本：{len(rows)} 张，来自 Caltech-101 的 {len(by_cls)} 个非果蔬类（HF 镜像）",
        f"- 模型：{config.VISION_MODEL.name}",
        f"- 当前阈值：conf {config.CONF_THRESHOLD:.2f} / margin {config.MARGIN_THRESHOLD:.2f}",
        "",
        "## 结论",
        "",
        f"未知物体检出率（拒识率）**{now * 100:.1f}%**，也就是说 100 件非商品里有 "
        f"{round((1 - now) * 100)} 件会被当成蔬果报价。",
        "",
        "| 配置 | 检出率 |",
        "|---|---|",
        f"| 不做拒识 | 0.0%（全部放行） |",
    ]
    for c in CONF_GRID:
        r_, _ = rate(rows, c, config.MARGIN_THRESHOLD)
        mark = " ←" if abs(c - config.CONF_THRESHOLD) < 1e-6 else ""
        lines.append(f"| conf {c:.2f} | {r_ * 100:.1f}%{mark} |")

    lines += [
        "",
        "## 按类别",
        "",
        "| 类别 | 张数 | 拒识 | 检出率 |",
        "|---|---|---|---|",
    ]
    for cls, s in sorted(by_cls.items(), key=lambda x: x[1]["rej"] / max(1, x[1]["n"])):
        lines.append(f"| {CN.get(cls, cls)} | {s['n']} | {s['rej']} | {s['rej'] / s['n'] * 100:.0f}% |")

    lines += [
        "",
        f"## 漏网的 {len(rows) - len(rejected)} 张被当成了什么",
        "",
        "| 误判成 | 张数 |",
        "|---|---|",
    ]
    for name, n in top_guesses:
        lines.append(f"| {name} | {n} |")

    lines += [
        "",
        "## 边界",
        "",
        "- Caltech-101 是 2003 年的网图，分辨率约 300×200，和托盘现场的手机/摄像头拍摄不同；",
        "  它能回答「模型面对非果蔬物体会不会硬猜」，但答不了「你这台秤在你的桌面上表现如何」。",
        "- 要后者只能自己拍：空托盘、手入镜、放手机，各 10 张，放进 data/negative/ 再跑一次本脚本。",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"负样本 {len(rows)} 张，检出率 {now * 100:.1f}%")
    print(f"报告 {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
