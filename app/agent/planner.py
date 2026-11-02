# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path

import config
from app.agent.tools import TOOL_SCHEMA

SYSTEM_PROMPT = """你是一个蔬果计价秤的调度器。可用工具：
checkout(paid) 完整计价闭环
recognize() 识别商品
get_price(item_key) 查单价
weigh() 称重
make_quote(item_key, grams) 算钱
speak(text) 语音播报
print_receipt(item_key, paid) 打小票
商品键：apple avocado banana carrot cucumber grape onion orange pear pepper strawberry tomato
用户说话后，只输出一行 JSON：{"action":"工具名","args":{...}}，不要输出其他内容。"""

KEYWORDS = {
    "get_price": ["单价", "价格", "怎么卖", "什么价", "多少钱一斤", "一斤多少", "一斤多少钱", "几块一斤"],
    "weigh": ["多重", "重量", "多少克", "几斤", "多少斤"],
    "recognize": ["这是什么", "识别", "看看是什么", "什么东西", "是什么菜"],
    "checkout": ["结账", "计价", "买单", "称一下", "算钱", "多少钱", "称重", "结算", "来一份", "称下"],
}

ORDER = ["get_price", "weigh", "recognize", "checkout"]
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
        if action in ("checkout", "print_receipt"):
            args = {"item_key": item_key} if item_key else {}
            if grams:
                args["grams"] = grams
            return args
        if action == "make_quote":
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


class LocalPlanner:
    def __init__(self):
        import onnxruntime_genai as og

        d = Path(config.LLM_LOCAL_DIR)
        if not d.exists():
            raise RuntimeError(f"本地模型目录不存在: {d}")
        self.og = og
        self.model = og.Model(str(d))
        self.tokenizer = og.Tokenizer(self.model)

    def plan(self, command, ctx=None):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{command}<|im_end|>\n<|im_start|>assistant\n"
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(max_length=1024, do_sample=False)
        params.input_ids = self.tokenizer.encode(prompt)
        gen = self.og.Generator(self.model, params)
        while not gen.is_done():
            gen.generate_next_token()
        text = self.tokenizer.decode(gen.get_sequence(0))
        return _parse_json_action(text)


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
    return {"action": action, "args": args}


def build_planner(prices):
    ctx = prices.items
    want = config.LLM_BACKEND
    order = {
        "auto": [LocalPlanner, ApiPlanner],
        "local": [LocalPlanner],
        "api": [ApiPlanner],
        "off": [],
    }.get(want, [])
    for cls in order:
        try:
            p = cls()
            print(f"[LLM] 使用 {cls.__name__}")
            return p, cls.__name__
        except Exception as e:
            print(f"[LLM] {cls.__name__} 不可用: {e}")
    print("[LLM] 回退到规则解析")
    return RulePlanner(), "RulePlanner"
