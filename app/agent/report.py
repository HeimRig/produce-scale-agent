# -*- coding: utf-8 -*-
import json
import time
from pathlib import Path

import config


def summarize(day=None, path=None):
    p = Path(path or config.SALES_JSONL)
    if not p.exists():
        return {"ok": False, "error": "还没有成交记录"}
    day = day or time.strftime("%Y-%m-%d")

    orders, by_item = [], {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not str(rec.get("time", "")).startswith(day):
            continue
        orders.append(rec)
        for it in rec.get("items", []):
            slot = by_item.setdefault(
                it["cn"], {"cn": it["cn"], "grams": 0.0, "amount": 0.0, "count": 0}
            )
            slot["grams"] += it.get("grams", 0)
            slot["amount"] += it.get("amount", 0)
            slot["count"] += 1

    ranking = sorted(by_item.values(), key=lambda x: -x["amount"])
    for r in ranking:
        r["grams"] = round(r["grams"], 1)
        r["amount"] = round(r["amount"], 2)
    return {
        "ok": True,
        "day": day,
        "orders": len(orders),
        "pieces": sum(r["count"] for r in ranking),
        "revenue": round(sum(o.get("total", 0) for o in orders), 2),
        "by_item": ranking,
    }
