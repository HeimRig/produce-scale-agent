# -*- coding: utf-8 -*-
from PIL import Image

from app.action.printer import MockPrinter
from app.action.tts import MockTTS
from app.agent.pricing import PriceTable
from app.agent.tools import AgentTools
from app.perception.scale import MockScale


class FakeCamera:
    def capture(self):
        return Image.new("RGB", (64, 64), (200, 30, 30))

    def close(self):
        pass


class FakeClassifier:
    def __init__(self, key="apple", cn="苹果", prob=0.92):
        self.key, self.cn, self.prob = key, cn, prob

    def predict(self, image, topk=3):
        return [{"key": self.key, "cn": self.cn, "prob": self.prob, "latency_ms": 1.0}]


def build(key="apple", cn="苹果", prob=0.92, grams=500):
    return AgentTools(
        camera=FakeCamera(),
        scale=MockScale(grams),
        classifier=FakeClassifier(key, cn, prob),
        tts=MockTTS(),
        printer=MockPrinter(),
        prices=PriceTable(),
    )


def test_checkout_happy_path():
    t = build(grams=500)
    r = t.checkout(paid=10)
    assert r["ok"] is True
    assert r["item"]["key"] == "apple"
    assert abs(r["item"]["grams"] - 500) < 1.0
    assert r["total"] == 4.0
    assert r["receipt"]["order_id"]
    assert [s["step"] for s in r["steps"]] == ["recognize", "weigh", "quote", "speak", "print"]


def test_checkout_explicit_grams_overrides_scale():
    t = build(grams=999)
    r = t.checkout(grams=250)
    assert r["ok"] is True
    assert r["item"]["grams"] == 250
    assert r["total"] == 2.0


def test_checkout_explicit_item_skips_vision():
    t = build(key="apple", cn="苹果")
    r = t.checkout(item_key="tomato", grams=1000)
    assert r["item"]["key"] == "tomato"
    assert r["total"] == 6.0


def test_low_confidence_still_returns_best_guess():
    t = build(prob=0.21, grams=300)
    r = t.checkout()
    assert r["ok"] is True
    assert r["receipt"]["conf"] == 0.21


def test_get_price_and_quote_without_weight():
    t = build()
    assert t.get_price("orange")["unit_price"] == 7.0
    assert t.get_price("durian")["ok"] is False
    assert t.make_quote("apple")["ok"] is False
