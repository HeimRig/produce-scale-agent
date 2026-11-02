# -*- coding: utf-8 -*-
import json
import time
import unicodedata
from pathlib import Path

import config


def _w(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _center(text, width):
    pad = max(0, width - _w(text))
    return " " * (pad // 2) + text


def _two_col(left, right, width):
    pad = max(1, width - _w(left) - _w(right))
    return left + " " * pad + right


def format_receipt(data):
    w = config.PRINTER_WIDTH
    line = "-" * w
    rows = ["", _center("蔬果智能计价单", w), _center("Produce Agent", w), line]
    items = data.get("items", [])
    multi = len(items) > 1
    for n, it in enumerate(items, 1):
        if multi and n > 1:
            rows.append("")
        rows.append(f"品名 {it['cn']}" + (f"  #{n}" if multi else ""))
        rows.append(_two_col(f"单价 {it['unit_price']:.2f} 元/kg", f"PLU {it.get('plu','-')}", w))
        rows.append(_two_col(f"净重 {it['grams']/1000:.3f} kg", f"{it['grams']:.0f} g", w))
        rows.append(_two_col("金额", f"{it['amount']:.2f} 元", w))
    rows.append(line)
    rows.append(_two_col("合计", f"{data['total']:.2f} 元", w))
    if data.get("paid") is not None:
        rows.append(_two_col("实收", f"{data['paid']:.2f} 元", w))
        rows.append(_two_col("找零", f"{max(0.0, data['paid']-data['total']):.2f} 元", w))
    rows.append(line)
    rows.append(f"单号 {data.get('order_id','-')}")
    rows.append(f"时间 {data.get('time','-')}")
    if data.get("conf") is not None:
        rows.append(f"识别置信度 {data['conf']*100:.1f}%")
    rows.append(_center("谢谢惠顾 欢迎再来", w))
    rows.append("")
    return "\n".join(rows)


class MockPrinter:
    def __init__(self):
        config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    def print_receipt(self, data):
        text = format_receipt(data)
        ts = time.strftime("%Y%m%d_%H%M%S")
        txt = config.OUT_DIR / f"receipt_{ts}.txt"
        txt.write_text(text, encoding="utf-8")
        (config.OUT_DIR / f"receipt_{ts}.bin").write_bytes(
            b"\x1c\x26" + text.replace("\n", "\r\n").encode(config.PRINTER_CHARSET, "replace")
        )
        print(text)
        _append_record(data)
        return txt

    def close(self):
        pass


class EscposPrinter:
    def __init__(self):
        from escpos.printer import Serial, Usb

        if config.PRINTER_BACKEND == "usb":
            self.p = Usb(
                config.PRINTER_USB_VENDOR,
                config.PRINTER_USB_PRODUCT,
                timeout=0,
                in_ep=0x82,
                out_ep=0x01,
            )
        else:
            self.p = Serial(
                devfile=config.PRINTER_PORT,
                baudrate=config.PRINTER_BAUDRATE,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=1.0,
                dsrdtr=True,
            )
        config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    def print_receipt(self, data):
        text = format_receipt(data)
        self.p._raw(b"\x1c\x26")
        for line in text.split("\n"):
            self.p._raw(line.encode(config.PRINTER_CHARSET, "replace") + b"\r\n")
        self.p._raw(b"\x1b\x64\x03")
        self.p.cut()
        ts = time.strftime("%Y%m%d_%H%M%S")
        (config.OUT_DIR / f"receipt_{ts}.txt").write_text(text, encoding="utf-8")
        _append_record(data)
        return text

    def close(self):
        try:
            self.p.close()
        except Exception:
            pass


def _append_record(data):
    p = config.SALES_JSONL
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def build_printer():
    if config.PRINTER_BACKEND in ("serial", "usb"):
        return EscposPrinter()
    return MockPrinter()
