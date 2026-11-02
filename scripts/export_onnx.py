# -*- coding: utf-8 -*-
"""导出 ONNX：fp32 → 静态 QDQ 量化 int8，并用验证集校验量化前后一致率。

python scripts/export_onnx.py
"""

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
PT_PATH = ROOT / "models" / "produce_mobilenetv3s.pt"
ONNX_FP32 = ROOT / "models" / "produce_mobilenetv3s.onnx"
ONNX_INT8 = ROOT / "models" / "produce_mobilenetv3s_int8.onnx"
VAL_DIR = ROOT / "data" / "produce" / "val"

IMG_SIZE = 224
OPSET = 18
CALIB_IMAGES = 120
MIN_AGREE = 0.95

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(arch, n_classes):
    if arch == "mobilenet_v2":
        net = models.mobilenet_v2(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
        return net
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, n_classes)
    return net


def prep(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None], dtype=np.float32)


class Calib(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.items = files
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        while self.i < len(self.items):
            f = self.items[self.i]
            self.i += 1
            return {self.input_name: prep(f)}
        return None

    def rewind(self):
        self.i = 0


def export_static(model, path):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        torch.onnx.export(
            model, (dummy,), str(path),
            input_names=["input"], output_names=["logits"],
            opset_version=OPSET, dynamo=True, external_data=False,
        )
    except TypeError:
        torch.onnx.export(
            model, (dummy,), str(path),
            input_names=["input"], output_names=["logits"], opset_version=OPSET,
        )


def strip_and_save(path):
    m = onnx.load(str(path))
    del m.graph.value_info[:]
    onnx.save(m, str(path))


def session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def agree(a_path, b_path, files):
    sa, sb = session(a_path), session(b_path)
    same = 0
    for f in files:
        x = prep(f)
        pa = np.argmax(sa.run(None, {sa.get_inputs()[0].name: x})[0])
        pb = np.argmax(sb.run(None, {sb.get_inputs()[0].name: x})[0])
        same += int(pa == pb)
    return same / max(1, len(files))


def main():
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "mobilenet_v3_small")
    model = build_model(arch, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"backbone: {arch}")

    for p in (ONNX_FP32, ONNX_FP32.with_name(ONNX_FP32.name + ".data"), ONNX_INT8):
        if p.exists():
            p.unlink()

    export_static(model, ONNX_FP32)
    strip_and_save(ONNX_FP32)
    onnx.checker.check_model(str(ONNX_FP32))

    val_files = sorted([p for p in VAL_DIR.rglob("*.jpg")])
    calib_files = val_files[:CALIB_IMAGES]
    if not calib_files:
        raise RuntimeError(f"验证集为空：{VAL_DIR}")

    s_fp32 = session(ONNX_FP32)
    trials = [
        ("QOperator", dict(quant_format=QuantFormat.QOperator, per_channel=True)),
        ("QDQ", dict(quant_format=QuantFormat.QDQ, per_channel=True)),
        ("QDQ_conv", dict(quant_format=QuantFormat.QDQ, per_channel=True,
                          op_types_to_quantize=["Conv", "MatMul", "Gemm"])),
    ]
    best = (0.0, None)
    for tag, kw in trials:
        tmp = ONNX_INT8.with_name(f"produce_mobilenetv3s_int8_{tag}.onnx")
        if tmp.exists():
            tmp.unlink()
        try:
            quantize_static(
                str(ONNX_FP32), str(tmp), Calib(calib_files, s_fp32.get_inputs()[0].name),
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
                **kw,
            )
        except Exception as e:
            print(f"{tag} 量化失败：{e}")
            continue
        r = agree(ONNX_FP32, tmp, val_files)
        print(f"{tag}: {tmp.stat().st_size/1e6:.2f} MB，一致率 {r*100:.1f}%")
        if r > best[0]:
            best = (r, tmp)

    rate, chosen = best
    if ONNX_INT8.exists():
        ONNX_INT8.unlink()
    if chosen is not None:
        chosen.rename(ONNX_INT8)
    for tag, _ in trials:
        stale = ONNX_INT8.with_name(f"produce_mobilenetv3s_int8_{tag}.onnx")
        if stale.exists():
            stale.unlink()

    print(f"fp32 {ONNX_FP32.stat().st_size/1e6:.2f} MB")
    if ONNX_INT8.exists():
        print(f"int8 {ONNX_INT8.stat().st_size/1e6:.2f} MB")
    print(f"最佳量化一致率 {rate*100:.1f}%（{len(val_files)} 张验证集，阈值 {MIN_AGREE*100:.0f}%）")

    if rate < MIN_AGREE:
        print("量化掉点过多，把 config.VISION_MODEL 指向 fp32 模型")
    else:
        print("可用：config.VISION_MODEL 指向 int8 模型")
    print("类别顺序:", classes)


if __name__ == "__main__":
    main()
