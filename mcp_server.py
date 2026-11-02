# -*- coding: utf-8 -*-
"""把计价秤能力暴露成 MCP 工具，供外部 Agent / 语音助手调用。

启动：python mcp_server.py
"""

from app.main import build_agent

try:                                    # mcp >= 2：FastMCP 改名为 MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("produce-agent")
_state = {}


def _t():
    if "tools" not in _state:
        tools, prices, _ = build_agent()
        _state["tools"] = tools
        _state["prices"] = prices
    return _state["tools"]


@mcp.tool()
def checkout(paid: float | None = None, grams: float | None = None,
             item_key: str | None = None) -> dict:
    """完整计价闭环：识别→称重→报价→播报→打票。grams 可指定重量，item_key 可跳过识别"""
    return _t().checkout(item_key=item_key, paid=paid, grams=grams)


@mcp.tool()
def recognize_produce() -> dict:
    """拍照识别托盘上的蔬果"""
    return _t().recognize()


@mcp.tool()
def get_price(item_key: str) -> dict:
    """查询单价，item_key 如 apple / banana"""
    return _t().get_price(item_key)


@mcp.tool()
def weigh() -> dict:
    """读取托盘重量（克）"""
    return _t().weigh()


@mcp.tool()
def make_quote(item_key: str, grams: float | None = None) -> dict:
    """按单价与重量算金额"""
    return _t().make_quote(item_key, grams)


@mcp.tool()
def speak(text: str) -> dict:
    """语音播报"""
    return _t().speak(text)


@mcp.tool()
def print_receipt(item_key: str, paid: float | None = None) -> dict:
    """打印小票"""
    return _t().print_receipt(item_key, paid=paid)


@mcp.tool()
def list_items() -> dict:
    """列出价目表全部商品"""
    _t()
    return {"ok": True, "items": _state["prices"].items}


if __name__ == "__main__":
    mcp.run()
