# -*- coding: utf-8 -*-
CN_NAMES = {
    "apple": "苹果", "avocado": "牛油果", "banana": "香蕉", "carrot": "胡萝卜",
    "cucumber": "黄瓜", "grape": "葡萄", "onion": "洋葱", "orange": "橙子",
    "pear": "梨", "pepper": "辣椒", "strawberry": "草莓", "tomato": "番茄",
    "watermelon": "西瓜", "kiwi": "猕猴桃", "potato": "土豆",
}

PRODUCE_CLASSES = ["apple", "avocado", "banana", "carrot", "cucumber", "grape",
                   "onion", "orange", "pear", "pepper", "strawberry", "tomato"]

DEFAULT_LABELS = [{"key": k, "cn": CN_NAMES[k], "en": k} for k in PRODUCE_CLASSES]


def load_labels(path=None):
    import json
    from pathlib import Path
    import config

    p = Path(path) if path else config.VISION_LABELS
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data
        if isinstance(data, list):
            return [{"key": k, "cn": CN_NAMES.get(k, k), "en": k} for k in data]
    return DEFAULT_LABELS
