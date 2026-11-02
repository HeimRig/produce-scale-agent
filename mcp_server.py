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
def add_to_cart(item_key: str | None = None, grams: float | None = None) -> dict:
    """识别并称重当前托盘商品，加入购物车，不结账"""
    return _t().add_to_cart(item_key=item_key, grams=grams)


@mcp.tool()
def remove_item(index: int | None = None) -> dict:
    """从购物车移除一件，默认最后一件；index 从 0 开始"""
    return _t().remove_item(index=index)


@mcp.tool()
def list_cart() -> dict:
    """查看购物车内容与小计"""
    return _t().list_cart()


@mcp.tool()
def checkout_all(paid: float | None = None) -> dict:
    """结算购物车全部商品，打一张汇总小票"""
    return _t().checkout_all(paid=paid)


@mcp.tool()
def clear_cart() -> dict:
    """清空购物车"""
    return _t().clear_cart()


@mcp.tool()
def sales_summary(day: str | None = None) -> dict:
    """按天汇总成交单数、件数、金额与商品排名，day 格式 YYYY-MM-DD，默认今天"""
    return _t().sales_summary(day=day)


@mcp.tool()
def recognize_produce() -> dict:
    """拍照识别托盘上的蔬果；两个候选太接近或置信度过低时返回 rejected"""
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
