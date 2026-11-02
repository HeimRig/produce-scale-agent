# -*- coding: utf-8 -*-
from app.agent.planner import RulePlanner
from app.agent.pricing import PriceTable


def planner():
    return RulePlanner(), PriceTable().items


def test_price_intent():
    p, ctx = planner()
    assert p.plan("苹果多少钱一斤", ctx)["action"] == "get_price"
    assert p.plan("番茄什么价格", ctx)["action"] == "get_price"
    assert p.plan("香蕉怎么卖", ctx)["action"] == "get_price"


def test_checkout_intent():
    p, ctx = planner()
    assert p.plan("把苹果称一下", ctx)["action"] == "checkout"
    assert p.plan("结账", ctx)["action"] == "checkout"
    assert p.plan("算钱", ctx)["action"] == "checkout"


def test_weigh_and_recognize():
    p, ctx = planner()
    assert p.plan("现在多重", ctx)["action"] == "weigh"
    assert p.plan("这是什么", ctx)["action"] == "recognize"


def test_item_extracted():
    p, ctx = planner()
    r = p.plan("苹果多少钱一斤", ctx)
    assert r["args"]["item_key"] == "apple"
    r = p.plan("把 tomato 称一下", ctx)
    assert r["args"]["item_key"] == "tomato"


def test_grams_parsed():
    p, ctx = planner()
    assert p.plan("称一下 500克苹果", ctx)["args"]["grams"] == 500
    assert p.plan("称 1.2公斤苹果", ctx)["args"]["grams"] == 1200
    assert p.plan("来两斤苹果", ctx)["args"]["grams"] == 1000


def test_unknown_command_falls_back_to_checkout():
    p, ctx = planner()
    assert p.plan("你好", ctx)["action"] == "checkout"
