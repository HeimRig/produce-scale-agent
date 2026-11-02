# -*- coding: utf-8 -*-
from pathlib import Path

from PIL import Image

import config
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
    def __init__(self, key="apple", cn="苹果", prob=0.92, second=None):
        self.key, self.cn, self.prob = key, cn, prob
        self.second = second

    def predict(self, image, topk=3):
        out = [{"key": self.key, "cn": self.cn, "prob": self.prob, "latency_ms": 1.0}]
        if self.second:
            out.append({"key": self.second[0], "cn": self.second[1], "prob": self.second[2], "latency_ms": 1.0})
        return out


def build(key="apple", cn="苹果", prob=0.92, grams=500, second=None):
    return AgentTools(
        camera=FakeCamera(),
        scale=MockScale(grams),
        classifier=FakeClassifier(key, cn, prob, second),
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


def test_low_confidence_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNKNOWN_DIR", tmp_path)
    t = build(prob=0.21, grams=300)
    r = t.checkout()
    assert r["ok"] is False
    assert "置信度" in r["reason"]
    assert Path(r["saved"]).exists()


def test_close_candidates_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNKNOWN_DIR", tmp_path)
    t = build(key="pear", cn="梨", prob=0.62, second=("apple", "苹果", 0.58))
    r = t.checkout()
    assert r["ok"] is False
    assert "只差" in r["reason"]


def test_explicit_item_bypasses_rejection():
    t = build(key="apple", cn="苹果", prob=0.20)
    r = t.checkout(item_key="banana", grams=500)
    assert r["ok"] is True
    assert r["item"]["key"] == "banana"


def test_cart_add_and_checkout_all():
    t = build(grams=500)
    assert t.add_to_cart(item_key="apple", grams=500)["ok"] is True
    assert t.add_to_cart(item_key="tomato", grams=250)["ok"] is True

    cart = t.list_cart()
    assert cart["count"] == 2
    assert cart["subtotal"] == 5.5

    r = t.checkout_all(paid=10)
    assert r["ok"] is True
    assert r["count"] == 2
    assert r["total"] == 5.5
    assert len(r["receipt"]["items"]) == 2
    assert t.list_cart()["count"] == 0


def test_remove_item():
    t = build()
    t.add_to_cart(item_key="apple", grams=500)
    t.add_to_cart(item_key="tomato", grams=250)
    r = t.remove_item()
    assert r["ok"] is True
    assert r["removed"]["key"] == "tomato"
    assert r["count"] == 1
    r = t.remove_item(index=0)
    assert r["count"] == 0
    assert t.remove_item()["ok"] is False


def test_checkout_all_on_empty_cart_falls_back_to_single():
    t = build(grams=500)
    r = t.checkout_all()
    assert r["ok"] is True
    assert "steps" in r


def test_get_price_and_quote_without_weight():
    t = build()
    assert t.get_price("orange")["unit_price"] == 7.0
    assert t.get_price("durian")["ok"] is False
    assert t.make_quote("apple")["ok"] is False


def test_sales_summary_reads_records(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    monkeypatch.setattr(config, "SALES_JSONL", tmp_path / "sales.jsonl")
    t = build()
    r0 = t.checkout(item_key="apple", grams=500)
    assert r0["ok"] is True
    r = t.sales_summary()
    assert r["ok"] is True
    assert r["orders"] >= 1
    assert r["revenue"] > 0
    assert r["by_item"][0]["cn"]
