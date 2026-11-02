# -*- coding: utf-8 -*-
"""统一入口。python manage.py <命令> [参数...]

数据与训练：fetch / prepare / train / export / eval
运行与演示：demo / cart / run / chat / mcp
运维：test / summary / calibrate
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

COMMANDS = {
    "demo": ([PY, "demo.py"], "单件闭环演示"),
    "cart": ([PY, "demo.py", "--cart"], "多件购物车演示"),
    "run": ([PY, "-m", "app.main"], "单次结账，可加 --loop / --weight"),
    "chat": ([PY, "-m", "app.main", "--chat"], "自然语言入口"),
    "mcp": ([PY, "mcp_server.py"], "启动 MCP stdio 服务"),
    "test": ([PY, "-m", "pytest", "tests", "-q"], "单元测试"),
    "fetch": ([PY, "scripts/fetch_dataset.py"], "下载公开图集"),
    "prepare": ([PY, "scripts/prepare_dataset.py"], "清洗并划分数据集"),
    "train": ([PY, "scripts/finetune.py"], "微调 MobileNetV3-Small"),
    "export": ([PY, "scripts/export_onnx.py"], "导出 ONNX"),
    "eval": ([PY, "scripts/eval_realshot.py"], "实拍留出集评估"),
    "threshold": ([PY, "scripts/calibrate_threshold.py"], "标定拒识阈值"),
    "negative": ([PY, "scripts/fetch_negative.py"], "抓非果蔬负样本"),
    "eval-neg": ([PY, "scripts/eval_negative.py"], "测未知物体检出率"),
    "calibrate": ([PY, "scripts/calibrate.py"], "HX711 去皮与标定"),
}


def usage(code=0):
    print("用法  python manage.py <命令> [参数...]\n")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<10}{desc}")
    print("\n  summary   今日成交汇总")
    print("\n示例")
    print("  python manage.py demo")
    print("  python manage.py cart")
    print("  python manage.py run --weight 500")
    print('  python manage.py chat "苹果多少钱一斤"')
    print("  python manage.py fetch --source fruits360 --per-class 300")
    return code


def summary(day=None):
    from app.agent.report import summarize

    r = summarize(day=day)
    if not r.get("ok"):
        print(r.get("error"))
        return 1
    print(f"{r['day']}  {r['orders']} 单  {r['pieces']} 件  合计 {r['revenue']:.2f} 元")
    for it in r["by_item"]:
        print(f"  {it['cn']:<8}{it['grams']:>9.0f} g  {it['amount']:>8.2f} 元  {it['count']} 次")
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        return usage()

    name = sys.argv[1]
    rest = sys.argv[2:]

    if name == "summary":
        return summary(rest[0] if rest else None)
    if name not in COMMANDS:
        print(f"未知命令 {name}\n")
        return usage(2)

    cmd, _ = COMMANDS[name]
    try:
        return subprocess.call(cmd + rest, cwd=ROOT)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
