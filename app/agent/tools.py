# -*- coding: utf-8 -*-
import time
import uuid

import config


class AgentTools:
    def __init__(self, camera, scale, classifier, tts, printer, prices):
        self.camera = camera
        self.scale = scale
        self.classifier = classifier
        self.tts = tts
        self.printer = printer
        self.prices = prices
        self._frame = None
        self._preds = None
        self._grams = None

    def capture(self):
        self._frame = self.camera.capture()
        self._preds = None
        return {"ok": True, "size": self._frame.size}

    def recognize(self):
        if self._frame is None:
            self.capture()
        self._preds = self.classifier.predict(self._frame)
        top = self._preds[0]
        return {
            "ok": top["prob"] >= config.CONF_THRESHOLD,
            "key": top["key"],
            "cn": top["cn"],
            "prob": round(top["prob"], 4),
            "topk": [{"key": p["key"], "cn": p["cn"], "prob": round(p["prob"], 4)} for p in self._preds],
            "latency_ms": top.get("latency_ms"),
        }

    def get_price(self, item_key):
        item = self.prices.get(item_key)
        if item is None:
            return {"ok": False, "error": f"价目表没有 {item_key}"}
        return {"ok": True, **item, "unit_price": item["price_per_kg"]}

    def weigh(self):
        grams, stable = self.scale.stable_grams()
        self._grams = grams
        return {
            "ok": stable and grams >= config.MIN_WEIGHT_G,
            "grams": round(grams, 1),
            "stable": stable,
            "min_g": config.MIN_WEIGHT_G,
        }

    def make_quote(self, item_key, grams=None):
        g = grams if grams is not None else self._grams
        if g is None:
            return {"ok": False, "error": "尚未称重"}
        return {"ok": True, **self.prices.quote(item_key, g)}

    def speak(self, text):
        try:
            self.tts.speak(text, blocking=False)
            return {"ok": True, "text": text}
        except Exception as e:
            print(f"[TTS] 播报失败（{type(e).__name__}: {e}），仅文字输出：{text}")
            return {"ok": False, "text": text, "error": str(e)}

    def print_receipt(self, item_key, grams=None, paid=None, conf=None):
        q = self.make_quote(item_key, grams)
        if not q["ok"]:
            return q
        data = {
            "order_id": time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": [
                {
                    "cn": q["cn"],
                    "plu": q["plu"],
                    "unit_price": q["unit_price"],
                    "grams": q["grams"],
                    "amount": q["amount"],
                }
            ],
            "total": q["amount"],
            "paid": paid,
            "conf": conf,
        }
        self.printer.print_receipt(data)
        return {"ok": True, **data}

    def checkout(self, item_key=None, paid=None, grams=None):
        steps = []
        t0 = time.perf_counter()
        last = [t0]

        def mark(name, result=None):
            now = time.perf_counter()
            steps.append(
                {"step": name, "result": result, "ms": round((now - last[0]) * 1000, 1)}
            )
            last[0] = now

        self.capture()
        r = self.recognize()
        mark("recognize", r)
        if not r["ok"] and item_key is None:
            self.capture()
            r = self.recognize()
            mark("recognize_retry", r)
        key = item_key or r["key"]

        w = (
            {"ok": True, "grams": round(float(grams), 1), "stable": True,
             "min_g": config.MIN_WEIGHT_G, "source": "explicit"}
            if grams is not None
            else self.weigh()
        )
        for _ in range(2 if grams is None else 0):
            if w["ok"]:
                break
            self.speak("请将商品放稳在托盘上")
            time.sleep(0.4)
            w = self.weigh()
        mark("weigh", w)
        if not w["ok"]:
            return {"ok": False, "error": "称重不稳定或重量过低", "steps": steps, "grams": w["grams"]}

        q = self.make_quote(key, w["grams"])
        mark("quote", q)

        text = f"{q['cn']}，{q['grams']/1000:.2f}公斤，单价{q['unit_price']:.2f}元每公斤，共{q['amount']:.2f}元"
        self.speak(text)
        mark("speak")

        rc = self.print_receipt(key, w["grams"], paid=paid, conf=r["prob"])
        mark("print", {"order_id": rc.get("order_id")})

        return {
            "ok": True,
            "item": {k: v for k, v in q.items() if k != "ok"},
            "total": q["amount"],
            "receipt": rc,
            "steps": steps,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def close(self):
        for obj in (self.camera, self.scale, self.tts, self.printer):
            try:
                obj.close()
            except Exception:
                pass


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "checkout",
            "description": "对托盘上的蔬果执行完整计价流程：识别、称重、报价、语音播报、打印小票",
            "parameters": {
                "type": "object",
                "properties": {
                    "paid": {"type": "number", "description": "顾客实收金额"},
                    "grams": {"type": "number", "description": "指定重量（克），不给则读秤"},
                    "item_key": {"type": "string", "description": "指定商品键，不给则用识别的结果"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize",
            "description": "拍照并识别当前托盘上的蔬果种类",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "查询某个蔬果的单价",
            "parameters": {
                "type": "object",
                "properties": {"item_key": {"type": "string", "description": "商品英文键，如 apple"}},
                "required": ["item_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weigh",
            "description": "读取当前托盘重量（克）",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_quote",
            "description": "按单价与重量计算金额",
            "parameters": {
                "type": "object",
                "properties": {"item_key": {"type": "string"}, "grams": {"type": "number"}},
                "required": ["item_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "语音播报一段文本",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "print_receipt",
            "description": "打印小票",
            "parameters": {
                "type": "object",
                "properties": {"item_key": {"type": "string"}, "paid": {"type": "number"}},
                "required": ["item_key"],
            },
        },
    },
]
