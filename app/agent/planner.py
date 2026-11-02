# -*- coding: utf-8 -*-
import json
import os
import re

import config
from app import log
from app.agent.tools import TOOL_SCHEMA

LOG = log.get("planner")
NO_ARG_ACTIONS = {"list_cart", "clear_cart", "recognize", "weigh"}

SYSTEM_PROMPT = """你是一个蔬果计价秤的调度器，只能从下面选一个工具：
checkout(paid, grams, item_key) 单件完整计价闭环（称重+报价+播报+打票）
add_to_cart(item_key, grams) 识别并称重，加入购物车，不结账
remove_item(index) 从购物车移除一件
list_cart() 查看购物车
checkout_all(paid) 结算购物车全部商品，打汇总小票
clear_cart() 清空购物车
sales_summary(day) 按天汇总营业额与商品排名
recognize() 识别托盘上的商品
get_price(item_key) 查单价
weigh() 读重量
make_quote(item_key, grams) 只算钱不打票
speak(text) 播报一句话
print_receipt(item_key, paid) 打小票

规则：
- 问价格（多少钱一斤、什么价）用 get_price，不要结账
- 问多重用 weigh，问是什么用 recognize
- 说「再来一个/加一份」用 add_to_cart，说「去掉/不要了」用 remove_item
- 说「结账/买单」时购物车有货用 checkout_all，否则用 checkout
- 问营业额、卖了多少用 sales_summary
- 商品键只用用户提到的；没说就留空
- 只输出 JSON：{"action":"工具名","args":{...}}，不要解释。"""

KEYWORDS = {
    "sales_summary": ["营业额", "今天卖", "卖了多少", "多少单", "营业额多少", "汇总"],
    "list_cart": ["购物车", "几件了", "都有什么", "车里有"],
    "remove_item": ["去掉", "不要了", "删掉", "退掉", "拿掉", "移除"],
    "add_to_cart": ["加入购物车", "加一个", "加一份", "再来一个", "再来两", "再来一", "多来", "加上"],
    "get_price": ["单价", "价格", "怎么卖", "什么价", "多少钱一斤", "一斤多少", "一斤多少钱", "几块一斤"],
    "weigh": ["多重", "重量", "多少克", "几斤", "多少斤"],
    "recognize": ["这是什么", "识别", "看看是什么", "什么东西", "是什么菜"],
    "checkout": ["结账", "计价", "买单", "称一下", "算钱", "多少钱", "称重", "结算", "来一份", "称下"],
}

ORDER = ["sales_summary", "list_cart", "remove_item", "add_to_cart",
         "get_price", "weigh", "recognize", "checkout"]
ACTIONS = set(ORDER) | {"make_quote", "speak", "print_receipt", "checkout_all", "clear_cart"}
WEIGHT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?|[一二两三四五六七八九十])\s*(kg|公斤|千克|斤|g|克)")
CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


class RulePlanner:
    """无 LLM 时的兜底：先认意图，再认商品和重量。"""

    def plan(self, command, ctx=None):
        cmd = command.strip()
        item_key = self._item(cmd, ctx)
        grams = self._grams(cmd)

        for action in ORDER:
            if any(k in cmd for k in KEYWORDS[action]):
                return {"action": action, "args": self._args(action, item_key, grams)}
        return {"action": "checkout", "args": self._args("checkout", item_key, grams)}

    @staticmethod
    def _item(cmd, ctx):
        for key, item in (ctx or {}).items():
            if item.get("cn") and item["cn"] in cmd:
                return key
            if key in cmd.lower():
                return key
        return None

    @staticmethod
    def _grams(cmd):
        m = WEIGHT_RE.search(cmd)
        if not m:
            return None
        raw = m.group(1)
        v = float(CN_NUM[raw]) if raw in CN_NUM else float(raw)
        unit = m.group(2)
        if unit in ("kg", "公斤", "千克"):
            v *= 1000
        elif unit == "斤":
            v *= 500
        return v

    @staticmethod
    def _args(action, item_key, grams):
        if action == "get_price":
            return {"item_key": item_key} if item_key else {}
        if action in ("checkout", "print_receipt", "add_to_cart", "make_quote"):
            args = {"item_key": item_key} if item_key else {}
            if grams:
                args["grams"] = grams
            return args
        return {}


class ApiPlanner:
    def __init__(self):
        from openai import OpenAI

        key = os.environ.get(config.API_KEY_ENV)
        if not key:
            raise RuntimeError(f"缺少环境变量 {config.API_KEY_ENV}")
        self.client = OpenAI(base_url=config.API_BASE, api_key=key, timeout=config.API_TIMEOUT_S)

    def plan(self, command, ctx=None):
        resp = self.client.chat.completions.create(
            model=config.API_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": command},
            ],
            tools=TOOL_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return {"action": call.function.name, "args": json.loads(call.function.arguments or "{}")}
        return _parse_json_action(msg.content or "")


class OllamaPlanner:
    """本地 Ollama（HTTP /api/chat，JSON 模式），不依赖任何 SDK。"""

    def __init__(self):
        self.base = config.OLLAMA_BASE.rstrip("/")
        self.model = config.OLLAMA_MODEL
        tags = self._call("/api/tags", None, method="GET", timeout=5)
        names = [m["name"] for m in tags.get("models", [])]
        if not names:
            raise RuntimeError("Ollama 已启动但没有模型，先 ollama pull 一个")
        if self.model not in names:
            self.model = names[0]
        self.last_latency_ms = None

    def _call(self, path, payload, method="POST", timeout=None):
        import urllib.request

        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout or config.OLLAMA_TIMEOUT_S) as r:
            return json.loads(r.read() or b"{}")

    def plan(self, command, ctx=None):
        import time

        items = "、".join(f"{k}({v.get('cn', '')})" for k, v in (ctx or {}).items())
        system = f"{SYSTEM_PROMPT}\n可选商品：{items}"
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 96},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": command},
            ],
        }
        t0 = time.perf_counter()
        resp = self._call("/api/chat", payload)
        self.last_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        text = (resp.get("message") or {}).get("content", "")
        out = _parse_json_action(text)

        # 意图交给模型，槽位（商品/重量）交给规则：小模型算「两斤」会算错，还会凭空编商品
        rule = RulePlanner().plan(command, ctx)
        if out.get("action") not in ACTIONS:
            out["action"] = rule["action"]
        out["args"] = {} if out["action"] in NO_ARG_ACTIONS else rule["args"]
        if out["action"] == "checkout_all":
            out["args"] = {}
        out["latency_ms"] = self.last_latency_ms
        return out


def _parse_json_action(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"action": "checkout", "args": {}}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "checkout", "args": {}}
    action = obj.get("action") or obj.get("name") or "checkout"
    args = obj.get("args") or obj.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if isinstance(args, dict):
        for k in ("grams", "paid"):
            v = args.get(k)
            if isinstance(v, str):
                try:
                    args[k] = float(v)
                except ValueError:
                    args.pop(k, None)
    return {"action": action, "args": args}


def build_planner(prices):
    ctx = prices.items
    want = config.LLM_BACKEND
    order = {
        "auto": [OllamaPlanner, ApiPlanner],
        "ollama": [OllamaPlanner],
        "api": [ApiPlanner],
        "off": [],
    }.get(want, [])
    for cls in order:
        try:
            p = cls()
            LOG.info("语义层使用 %s", cls.__name__)
            return p, cls.__name__
        except Exception as e:
            LOG.warning("%s 不可用：%s", cls.__name__, e)
    LOG.info("语义层回退到规则解析")
    return RulePlanner(), "RulePlanner"
