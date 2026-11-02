# -*- coding: utf-8 -*-
"""MCP 工具冒烟测试：不进真实传输层，直接调注册好的工具函数。

    python scripts/mcp_smoke.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncio


def main():
    try:
        import mcp_server as srv
    except ImportError as e:
        print(f"没装 mcp：pip install mcp（{e}）")
        return 1

    print("1) 列出工具")
    tools = srv.mcp.list_tools()
    if asyncio.iscoroutine(tools):
        tools = asyncio.run(tools)
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "title", "?")
        print(f"   - {name}: {getattr(t, 'description', '')}")
    print(f"   共 {len(tools)} 个")

    print("\n2) list_items")
    r = srv.list_items()
    print(f"   ok={r['ok']} 商品数={len(r['items'])}")

    print("\n3) get_price(apple)")
    print("  ", srv.get_price("apple"))

    print("\n4) checkout()")
    r = srv.checkout()
    print(f"   ok={r.get('ok')} 合计={r.get('total')} 耗时={r.get('elapsed_ms')} ms")
    for s in r.get("steps", []):
        print(f"     · {s['step']:<16}{s.get('ms', 0):>7.1f} ms")
    if not r.get("ok"):
        print(f"   失败原因：{r.get('error')}")
        return 1

    print("\n5) make_quote(tomato, 350)")
    print("  ", srv.make_quote("tomato", 350))

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
