# -*- coding: utf-8 -*-
"""长稳测试：连跑 N 单，看延迟是否漂移、线程与临时文件是否回收、内存是否增长。

    python scripts/stability.py --orders 50
    python scripts/stability.py --orders 20 --tts mock     # 不联网、不出声

输出 docs/STABILITY_REPORT.md
"""

import argparse
import gc
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from app import log  # noqa: E402
from app.main import build_agent  # noqa: E402

REPORT = ROOT / "docs" / "STABILITY_REPORT.md"


def rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def tts_files():
    return len(list(Path(tempfile.gettempdir()).glob("produce_agent_tts_*.mp3")))


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=50)
    ap.add_argument("--tts", choices=["auto", "edge", "mock"], default="auto")
    ap.add_argument("--interval", type=float, default=0.0, help="两单之间的间隔秒数，模拟真实收银节奏")
    ap.add_argument("--settle", type=float, default=8.0, help="收尾观察秒数")
    args = ap.parse_args()

    log.setup()
    if args.tts != "auto":
        config.TTS_BACKEND = args.tts

    for old in Path(tempfile.gettempdir()).glob("produce_agent_tts_*.mp3"):
        try:
            old.unlink()
        except OSError:
            pass

    tools, prices, backend = build_agent(seed_weight=520)
    print("后端", backend)

    files0, rss0 = tts_files(), rss_mb()
    gc.collect()
    lat, peak_threads, peak_pending = [], 0, 0
    t_start = time.perf_counter()

    for i in range(args.orders):
        r = tools.checkout()
        if not r.get("ok"):
            print(f"第 {i} 单失败：{r.get('error')}")
            continue
        lat.append(r["elapsed_ms"])
        peak_threads = max(peak_threads, threading.active_count())
        peak_pending = max(peak_pending, getattr(tools.tts, "pending", 0))
        if args.interval:
            time.sleep(args.interval)

    wall = time.perf_counter() - t_start
    time.sleep(args.settle)
    tools.tts.wait()

    files1, rss1 = tts_files(), rss_mb()
    dropped = getattr(tools.tts, "dropped", 0)
    tools.close()

    rss_txt = f"{rss1 - rss0:+.1f} MB" if rss0 and rss1 else "未安装 psutil"
    lines = [
        "# 长稳测试",
        "",
        f"- 连续 {len(lat)} 单，间隔 {args.interval:.1f}s，TTS 后端 {type(tools.tts).__name__}，收尾观察 {args.settle:.0f}s",
        f"- 模型 {config.VISION_MODEL.name}，闭环保留语音异步不阻塞",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 单均价 | {statistics.mean(lat):.1f} ms |",
        f"| P50 / P95 / 最大 | {pct(lat, 50):.1f} / {pct(lat, 95):.1f} / {max(lat):.1f} ms |",
        f"| 首 10 单均值 → 末 10 单均值 | {statistics.mean(lat[:10]):.1f} → {statistics.mean(lat[-10:]):.1f} ms |",
        f"| 总墙钟（不含收尾） | {wall:.1f} s |",
        f"| 线程峰值 | {peak_threads} |",
        f"| 播报队列峰值 | {peak_pending} |",
        f"| 因超过并发上限丢弃 | {dropped} 次 |",
        f"| 临时 mp3 残留 | {files0} → {files1} |",
        f"| 进程 RSS 变化 | {rss_txt} |",
        "",
        "## 判读",
        "",
        "- 末 10 单均值不应高于首 10 单，否则说明有资源没释放。",
        "- 临时文件应收敛回起始值，播报线程应回到 0，否则是清理逻辑漏了。",
        "- 线程峰值应持平而不是逐单抬高。",
        "- RSS 在头十几单会有一次跳升（加载 edge-tts / TLS 模块），之后基本持平；只看绝对值判断不了泄漏。",
        "- 间隔小于语音时长（合成 1.4s + 播放约 5s）时会出现主动丢弃，这是设计行为：",
        "  宁可少播一句，也不让播放器排队积压几十个进程。",
        "- 语音异步不阻塞，所以单均延迟反映的是识别+称重+计价+打票四步。",
        "",
    ]

    ok = (
        files1 <= files0 + 1
        and getattr(tools.tts, "pending", 0) == 0
        and statistics.mean(lat[-10:]) <= statistics.mean(lat[:10]) * 1.5
    )
    lines.append(f"**判定：{'通过' if ok else '不通过'}**")
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
