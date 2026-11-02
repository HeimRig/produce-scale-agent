# -*- coding: utf-8 -*-
import time
import random
import statistics

import config


class MockScale:
    def __init__(self, grams=None):
        # 未指定重量时给一个固定的随机值：每次读数都换会让 spread 变大、永远判不稳
        self.grams = grams if grams is not None else random.uniform(180, 900)
        self._reads = 0

    def set_weight(self, grams):
        self.grams = grams

    def read_grams(self):
        self._reads += 1
        base = self.grams if self.grams is not None else random.uniform(180, 900)
        return base + random.uniform(-0.35, 0.35)

    def stable_grams(self, samples=None, tol=None):
        samples = samples or config.SCALE_SAMPLES
        tol = tol if tol is not None else config.SCALE_SETTLE_TOL_G
        vals = [self.read_grams() for _ in range(samples)]
        spread = max(vals) - min(vals)
        return statistics.median(vals), spread <= tol * samples

    def tare(self):
        self.grams = 0.0

    def close(self):
        pass


class Hx711Scale:
    def __init__(self, dout=None, sck=None, chip=None, ref_unit=None, offset=None):
        import lgpio

        self.lgpio = lgpio
        self.dout = dout if dout is not None else config.HX711_DOUT
        self.sck = sck if sck is not None else config.HX711_SCK
        self.chip = chip if chip is not None else config.HX711_CHIP
        self.ref_unit = ref_unit or config.HX711_REF_UNIT
        self.offset = config.HX711_OFFSET if offset is None else offset

        self.h = lgpio.gpiochip_open(self.chip)
        lgpio.gpio_claim_input(self.h, self.dout)
        lgpio.gpio_claim_output(self.h, self.sck, 0)

    def _read_raw(self, timeout=0.5):
        g = self.lgpio
        t0 = time.time()
        while g.gpio_read(self.h, self.dout) == 1:
            if time.time() - t0 > timeout:
                raise TimeoutError("HX711 not ready")
            time.sleep(0.001)
        val = 0
        for _ in range(24):
            g.gpio_write(self.h, self.sck, 1)
            val = (val << 1) | g.gpio_read(self.h, self.dout)
            g.gpio_write(self.h, self.sck, 0)
        g.gpio_write(self.h, self.sck, 1)
        g.gpio_write(self.h, self.sck, 0)
        if val & 0x800000:
            val -= 0x1000000
        return val

    def read_grams(self):
        return (self._read_raw() - self.offset) / self.ref_unit

    def stable_grams(self, samples=None, tol=None):
        samples = samples or config.SCALE_SAMPLES
        tol = tol if tol is not None else config.SCALE_SETTLE_TOL_G
        vals = [self.read_grams() for _ in range(samples)]
        spread = max(vals) - min(vals)
        return statistics.median(vals), spread <= tol * samples

    def tare(self, samples=10):
        self.offset = statistics.median([self._read_raw() for _ in range(samples)])
        return self.offset

    def calibrate(self, known_grams, samples=10):
        raw = statistics.median([self._read_raw() for _ in range(samples)])
        self.ref_unit = (raw - self.offset) / known_grams
        return self.ref_unit

    def close(self):
        try:
            self.lgpio.gpiochip_close(self.h)
        except Exception:
            pass


class Hx711LibScale:
    """hx711 库读数，标定值存在项目内的 scale.json。"""

    def __init__(self, dout=None, sck=None, scale_json=None, offset=None, ref_unit=None):
        import json
        from pathlib import Path
        from hx711 import HX711

        self.hx = None
        for kw in (
            {"dout_pin": dout or config.HX711_DOUT, "pd_sck_pin": sck or config.HX711_SCK,
             "channel": "A", "gain": 128},
            {"dout_pin": dout or config.HX711_DOUT, "pd_sck_pin": sck or config.HX711_SCK},
        ):
            try:
                self.hx = HX711(**kw)
                break
            except TypeError:
                continue
        if self.hx is None:
            raise RuntimeError("HX711 构造失败")

        p = Path(scale_json or config.SCALE_JSON)
        cfg = {}
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
        self.offset = cfg.get("offset", config.HX711_OFFSET) if offset is None else offset
        self.scale = cfg.get("scale", config.HX711_REF_UNIT) if ref_unit is None else ref_unit
        self.path = p

    def _raw(self):
        for name in ("get_raw_data", "read", "get_value"):
            fn = getattr(self.hx, name, None)
            if fn is None:
                continue
            try:
                v = fn(num_measures=1)
            except TypeError:
                try:
                    v = fn()
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    def read_grams(self):
        v = self._raw()
        if v is None:
            raise TimeoutError("HX711 读不到数据")
        return (v - self.offset) / self.scale

    def stable_grams(self, samples=None, tol=None):
        samples = samples or config.SCALE_SAMPLES
        tol = tol if tol is not None else config.SCALE_SETTLE_TOL_G
        vals = []
        for _ in range(samples):
            try:
                vals.append(self.read_grams())
            except TimeoutError:
                pass
            time.sleep(0.02)
        if not vals:
            raise TimeoutError("HX711 读不到数据")
        spread = max(vals) - min(vals)
        return statistics.median(vals), spread <= tol * samples

    def tare(self, samples=8):
        vals = []
        for _ in range(samples):
            v = self._raw()
            if v is not None:
                vals.append(v)
            time.sleep(0.05)
        if not vals:
            raise TimeoutError("HX711 读不到数据")
        self.offset = sum(vals) / len(vals)
        self._save()
        return self.offset

    def calibrate(self, known_grams, samples=10):
        vals = []
        for _ in range(samples):
            v = self._raw()
            if v is not None:
                vals.append(v)
            time.sleep(0.05)
        self.scale = (sum(vals) / len(vals) - self.offset) / known_grams
        self._save()
        return self.scale

    def _save(self):
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"offset": self.offset, "scale": self.scale}, indent=2), encoding="utf-8"
        )

    def close(self):
        try:
            self.hx.power_down()
        except Exception:
            pass


def build_scale():
    if config.SCALE_BACKEND == "hx711":
        try:
            return Hx711LibScale()
        except Exception as e:
            print(f"[scale] hx711 库不可用（{e}），退回 lgpio 直读")
            return Hx711Scale()
    return MockScale()
