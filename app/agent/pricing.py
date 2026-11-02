# -*- coding: utf-8 -*-
import json
from pathlib import Path

import config


class PriceTable:
    def __init__(self, path=None):
        self.path = Path(path or config.PRICE_TABLE)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.items = self.data["items"]

    def get(self, key):
        item = self.items.get(key)
        if item is None:
            return None
        return {"key": key, **item}

    def unknown(self):
        return {"key": "unknown", "cn": "未录入商品", "price_per_kg": 0.0, "plu": "0000", "origin": "-"}

    def quote(self, key, grams):
        item = self.get(key) or self.unknown()
        amount = round(grams / 1000.0 * item["price_per_kg"] + 1e-9, 2)
        return {
            "key": item["key"],
            "cn": item["cn"],
            "plu": item.get("plu", "-"),
            "unit_price": item["price_per_kg"],
            "grams": round(grams, 1),
            "amount": amount,
        }
