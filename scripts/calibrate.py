# -*- coding: utf-8 -*-
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from app.perception.scale import Hx711LibScale, Hx711Scale

KNOWN_GRAMS = 500.0


def make_scale():
    try:
        s = Hx711LibScale()
        print("使用 hx711 库读秤")
        return s
    except Exception as e:
        print(f"hx711 库不可用（{e}），改用 lgpio 直读")
        return Hx711Scale()


def main():
    if config.SCALE_BACKEND != "hx711" and "--force" not in sys.argv:
        print("config.SCALE_BACKEND 不是 hx711，实机上请先改成 hx711")
        return

    s = make_scale()
    print("请清空秤盘，3 秒后去皮…")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    offset = s.tare()
    print(f"offset = {offset:.1f}")

    print(f"请放上 {KNOWN_GRAMS:.0f} g 标准砝码，回车继续")
    input()
    ref = s.calibrate(KNOWN_GRAMS)
    print(f"scale  = {ref:.3f} counts/g")

    grams, stable = s.stable_grams()
    print(f"校验读数 {grams:.1f} g  stable={stable}")
    s.close()
    print(f"\n已写入 {config.SCALE_JSON}")
    print("如果用的是 lgpio 直读（没有 scale.json），把这两个值填回 config.py 的 HX711_OFFSET / HX711_REF_UNIT")


if __name__ == "__main__":
    main()
