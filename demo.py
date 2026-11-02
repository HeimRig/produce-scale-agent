# -*- coding: utf-8 -*-
"""一键演示：python demo.py

不需要参数。电脑上没有硬件也能跑完整个闭环（mock 摄像头/秤，小票落盘）。
"""

import random
import sys
import time

from app.main import build_agent


def main():
    weight = random.randint(280, 780)
    t0 = time.perf_counter()
    tools, prices, backend = build_agent(seed_weight=weight)
    print("=" * 52)
    print(" 蔬果计价 · 一次完整闭环")
    print("=" * 52)
    print(f" 后端  {backend}")

    res = tools.checkout(paid=round((weight / 1000) * 10 + 5, 2))
    if not res.get("ok"):
        print(f"\n失败：{res.get('error')}")
        tools.close()
        return 1

    for step in res["steps"]:
        r = step.get("result") or {}
        if step["step"].startswith("recognize"):
            top = "  ".join(f"{p['cn']} {p['prob']*100:.0f}%" for p in r.get("topk", []))
            print(f" 识别  {r['cn']}  {r['prob']*100:.1f}%   [{top}]")
        elif step["step"] == "weigh":
            print(f" 称重  {r['grams']:.1f} g   稳定={r['stable']}")
        elif step["step"] == "quote":
            print(f" 计价  {r['unit_price']:.2f} 元/kg × {r['grams']/1000:.3f} kg = {r['amount']:.2f} 元")

    print(f"\n 小票已落盘 out/    单号 {res['receipt'].get('order_id')}    合计 {res['total']:.2f} 元")
    print(f" 闭环耗时 {res['elapsed_ms']:.1f} ms（语音异步播放，不阻塞）")

    if hasattr(tools.tts, "wait"):
        print(" 等待语音播完…")
        tools.tts.wait()
    print(f" 进程总耗时 {(time.perf_counter()-t0)*1000:.0f} ms")
    tools.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
