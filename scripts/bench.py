# -*- coding: utf-8 -*-
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from app.vision.classifier import ProduceClassifier

RUNS = 50
WARMUP = 10
REPORT = ROOT / "docs" / "PERF_REPORT.md"


def rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return out.splitlines()[0] if out else platform.processor()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line.lower() or "Model" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def pct(vals, p):
    return float(np.percentile(vals, p))


def main():
    img_path = config.MOCK_IMAGE
    if not img_path.exists():
        Image.new("RGB", (1280, 720), (230, 230, 230)).save(img_path)
    img = Image.open(img_path).convert("RGB")

    t0 = time.perf_counter()
    clf = ProduceClassifier()
    load_ms = (time.perf_counter() - t0) * 1000
    mem_after_load = rss_mb()

    if clf.ready:
        for _ in range(WARMUP):
            clf.predict(img)
        lat = []
        for _ in range(RUNS):
            t = time.perf_counter()
            clf.predict(img)
            lat.append((time.perf_counter() - t) * 1000)
        size_mb = clf.model_path.stat().st_size / 1e6
    else:
        lat = [0.0]
        size_mb = 0.0

    lines = [
        "# 性能报告 / Performance Report",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 平台：{platform.system()} {platform.machine()}",
        f"- CPU：{cpu_name()}",
        f"- Python：{platform.python_version()}",
        f"- 视觉后端：{'ONNX Runtime (CPU)' if clf.ready else 'MOCK（未找到模型）'}",
        f"- 模型文件：{clf.model_path.name if clf.ready else '-'}",
        "",
        "## 视觉推理",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 模型体积 | {size_mb:.2f} MB |",
        f"| 加载耗时 | {load_ms:.1f} ms |",
        f"| 单帧推理 P50 | {pct(lat,50):.1f} ms |" if clf.ready else "| 单帧推理 P50 | - |",
        f"| 单帧推理 P95 | {pct(lat,95):.1f} ms |" if clf.ready else "| 单帧推理 P95 | - |",
        f"| 平均推理 | {statistics.mean(lat):.1f} ms |" if clf.ready else "| 平均推理 | - |",
        f"| 常驻内存 | {mem_after_load:.0f} MB |" if mem_after_load else "| 常驻内存 | - |",
        f"| 样本数 | {RUNS}（预热 {WARMUP}） |",
        "",
        "## 目标延迟预算（树莓派 5，8G）",
        "",
        "| 阶段 | 预算 | 说明 |",
        "|---|---|---|",
        "| 拍照 + 取帧 | 80 ms | picamera2 RGB888，1280×720 |",
        "| 视觉识别 | 60 ms | MobileNetV3-Small fp32，224×224 |",
        "| 称重稳定采样 | 400 ms | HX711 10Hz × 7 次取中位数 |",
        "| 报价 + 状态机 | < 5 ms | 纯 Python |",
        "| 语音播报 | 1500 ms | edge-tts 合成 1.4s + 播放，实测；异步不阻塞主链路 |",
        "| 打小票 | 600 ms | 58mm 热敏，串口 9600 |",
        "| **合计** | **≈ 1.2 s** | 不含语音播放（异步），不含顾客付款时间 |",
        "",
        "## 备注",
        "",
        "- 闭环耗时以「放货 → 小票吐出」计；语音异步播放，不阻塞交易。",
        "- 树莓派实测数据待补：在 Pi 上执行 `python scripts/bench.py` 会覆盖本文件。",
        "- LLM 语义层为旁路，不进入主链路，不占用上表延迟。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写入 {REPORT}")


if __name__ == "__main__":
    main()
