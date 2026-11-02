# -*- coding: utf-8 -*-
"""一键演示：python demo.py [--cart]

不带参数跑单件闭环；--cart 跑多件购物车结账。
不需要硬件：摄像头和秤走 mock，小票落盘到 out/。
"""

import random
import sys
import time

from app import log
from app.main import build_agent

CART_SAMPLE = [("apple", 320), ("banana", 450), ("tomato", 260)]


def main():
    cart_mode = "--cart" in sys.argv
    weight = random.randint(280, 780)
    t0 = time.perf_counter()
    log.setup()

    tools, prices, backend = build_agent(seed_weight=weight)
    print("=" * 52)
    print(" 蔬果计价 · " + ("一单多件" if cart_mode else "一次完整闭环"))
    print("=" * 52)
    print(f" 后端  {backend}")

    if cart_mode:
        for key, grams in CART_SAMPLE:
            r = tools.add_to_cart(item_key=key, grams=grams)
            if not r.get("ok"):
                print(f" 加入失败 {key}：{r.get('error')}")
        subtotal = tools.list_cart()
        for it in subtotal["items"]:
            print(f" #{it['index']}  {it['cn']:<6} {it['grams']:>6.0f} g × {it['unit_price']:.2f} = {it['amount']:.2f} 元")
        print(f" 小计  {subtotal['subtotal']:.2f} 元 / {subtotal['count']} 件")
        res = tools.checkout_all(paid=round(subtotal["subtotal"] + 10, 2))
    else:
        res = tools.checkout(paid=round((weight / 1000) * 10 + 5, 2))

    if not res.get("ok"):
        print(f"\n失败：{res.get('error')}  {res.get('reason') or ''}")
        tools.close()
        return 1

    for step in res.get("steps", []):
        r = step.get("result") or {}
        if step["step"].startswith("recognize"):
            top = "  ".join(f"{p['cn']} {p['prob']*100:.0f}%" for p in r.get("topk", []))
            print(f" 识别  {r['cn']}  {r['prob']*100:.1f}%   差值 {r['margin']*100:.1f}   [{top}]")
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
