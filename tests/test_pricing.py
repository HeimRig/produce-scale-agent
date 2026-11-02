# -*- coding: utf-8 -*-
from app.agent.pricing import PriceTable


def test_quote_math():
    t = PriceTable()
    q = t.quote("apple", 500)
    assert q["unit_price"] == 8.0
    assert q["amount"] == 4.0
    assert q["cn"] == "苹果"


def test_quote_rounding():
    t = PriceTable()
    q = t.quote("pepper", 333)
    assert q["amount"] == round(333 / 1000 * 8.0, 2)


def test_quote_zero_weight():
    t = PriceTable()
    assert t.quote("tomato", 0)["amount"] == 0.0


def test_unknown_key_uses_placeholder():
    t = PriceTable()
    q = t.quote("durian", 500)
    assert q["key"] == "unknown"
    assert q["amount"] == 0.0


def test_all_prices_positive():
    t = PriceTable()
    for key, item in t.items.items():
        assert item["price_per_kg"] > 0, key
        assert item["cn"], key
