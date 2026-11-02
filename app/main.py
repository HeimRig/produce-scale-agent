# -*- coding: utf-8 -*-
import argparse
import json
import time

import config
from app.action.printer import build_printer
from app.action.tts import build_tts
from app.agent.planner import build_planner
from app.agent.pricing import PriceTable
from app.agent.tools import AgentTools
from app.perception.camera import build_camera
from app.perception.scale import MockScale, build_scale
from app.vision.classifier import ProduceClassifier


def build_agent(seed_weight=None):
    camera = build_camera()
    scale = build_scale()
    if isinstance(scale, MockScale) and seed_weight is not None:
        scale.set_weight(seed_weight)
    classifier = ProduceClassifier()
    tts = build_tts()
    printer = build_printer()
    prices = PriceTable()
    tools = AgentTools(camera, scale, classifier, tts, printer, prices)
    backend = {
        "camera": type(camera).__name__,
        "scale": type(scale).__name__,
        "vision": "onnx" if classifier.ready else "mock",
        "tts": type(tts).__name__,
        "printer": type(printer).__name__,
    }
    return tools, prices, backend


def show(res):
    brief = {k: v for k, v in res.items() if k != "steps"}
    print(json.dumps(brief, ensure_ascii=False, indent=2, default=str))
    for s in res.get("steps", []):
        print(f"  · {s['step']:<16} {s.get('ms', 0):>7.1f} ms")


def run_chat(tools, planner, text, prices):
    plan = planner.plan(text, prices.items)
    action, args = plan.get("action"), plan.get("args", {}) or {}
    ms = plan.get("latency_ms")
    print(f"[路由] {action} {args}" + (f"  ({ms:.0f} ms)" if ms else ""))
    fn = getattr(tools, action, None)
    if fn is None:
        return {"ok": False, "error": f"未知工具 {action}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="循环结账")
    ap.add_argument("--chat", type=str, default=None, help="自然语言指令")
    ap.add_argument("--weight", type=float, default=None, help="mock 模式设定重量(克)")
    ap.add_argument("--paid", type=float, default=None, help="实收金额")
    ap.add_argument("--item", type=str, default=None, help="跳过识别，指定商品键")
    args = ap.parse_args()

    tools, prices, backend = build_agent(seed_weight=args.weight)
    print("[后端]", backend)

    try:
        if args.chat:
            planner, name = build_planner(prices)
            res = run_chat(tools, planner, args.chat, prices)
            print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        elif args.loop:
            while True:
                show(tools.checkout(item_key=args.item, paid=args.paid))
                time.sleep(1.0)
        else:
            show(tools.checkout(item_key=args.item, paid=args.paid))
    finally:
        tools.close()


if __name__ == "__main__":
    main()
