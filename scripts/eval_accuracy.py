# -*- coding: utf-8 -*-
"""在验证集上对比 torch / onnx-fp32 / onnx-int8 的准确率与延迟。"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "models" / "produce_mobilenetv3s.pt"
FP32 = ROOT / "models" / "produce_mobilenetv3s.onnx"
INT8 = ROOT / "models" / "produce_mobilenetv3s_int8.onnx"
VAL_DIR = ROOT / "data" / "produce" / "val"
REPORT = ROOT / "docs" / "ACCURACY_REPORT.md"
IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(path):
    img = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    x = np.asarray(img, np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None], dtype=np.float32)


def build_model(arch, n_classes):
    if arch == "mobilenet_v2":
        net = models.mobilenet_v2(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
        return net
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, n_classes)
    return net


def main():
    ckpt = torch.load(PT, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "mobilenet_v3_small")
    net = build_model(arch, len(classes))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    sessions = {}
    for name, p in (("onnx-fp32", FP32), ("onnx-int8", INT8)):
        if p.exists():
            sessions[name] = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])

    files = []
    for cls_dir in sorted([d for d in VAL_DIR.iterdir() if d.is_dir()]):
        for f in sorted(cls_dir.glob("*.jpg")):
            files.append((f, cls_dir.name))

    results = {}
    lat = {k: [] for k in ["torch", *sessions]}
    per_class = {}
    with torch.no_grad():
        for f, label in files:
            x = prep(f)
            t0 = time.perf_counter()
            pred_t = classes[int(net(torch.from_numpy(x)).argmax(1))]
            lat["torch"].append((time.perf_counter() - t0) * 1000)
            row = {"torch": pred_t == label}
            for name, s in sessions.items():
                t0 = time.perf_counter()
                out = s.run(None, {s.get_inputs()[0].name: x})[0]
                lat[name].append((time.perf_counter() - t0) * 1000)
                row[name] = classes[int(np.argmax(out))] == label
            for k, v in row.items():
                results.setdefault(k, []).append(v)
            pc = per_class.setdefault(label, {"n": 0})
            pc["n"] += 1
            for k, v in row.items():
                pc[k] = pc.get(k, 0) + int(v)

    lines = [
        "# 准确率与延迟（本地验证集）",
        "",
        f"- 验证集：{len(files)} 张，{len(per_class)} 类",
        f"- 模型：MobileNetV3-Small，{len(classes)} 类输出",
        "",
        "| 后端 | Top-1 准确率 | 单张延迟 |",
        "|---|---|---|",
    ]
    for k in ["torch", *sessions]:
        acc = sum(results[k]) / max(1, len(results[k]))
        lines.append(f"| {k} | {acc*100:.1f}% | {np.mean(lat[k]):.1f} ms |")

    lines += ["", "## 分类别（int8）", "", "| 类别 | 张数 | 正确 |", "|---|---|---|"]
    key = "onnx-int8" if "onnx-int8" in sessions else "torch"
    for cls, pc in sorted(per_class.items()):
        lines.append(f"| {cls} | {pc['n']} | {pc.get(key, 0)} |")

    lines += [
        "",
        "> 验证集与训练集同源（都是 Fruits-360 白底棚拍），准确率会明显偏高，不能当论文里的现场指标。",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
