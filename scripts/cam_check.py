# -*- coding: utf-8 -*-
"""摄像头自检：确认能出图、分辨率正常，顺手存一张当 README 截图。

    python scripts/cam_check.py
    python scripts/cam_check.py --index 1

空格：存图到 assets/samples/    q：退出
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("没装 opencv：pip install opencv-python")
        return 1
    from PIL import Image

    index = config.WEBCAM_INDEX if args.index is None else args.index
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"打不开摄像头 index={index}，换个 --index 1 / 2 再试")
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头就绪：{w}x{h}  index={index}")
    if args.no_window:
        ok, frame = cap.read()
        print("取帧", "成功" if ok else "失败")
        cap.release()
        return 0 if ok else 1

    out_dir = ROOT / "assets" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    t0 = time.time()
    frames = 0
    print("空格存图，q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("取帧失败")
            break
        frames += 1
        fps = frames / max(1e-6, time.time() - t0)
        cv2.putText(frame, f"{fps:.0f} fps  SPACE=save  q=quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("cam_check", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            p = out_dir / f"snap_{time.strftime('%H%M%S')}.jpg"
            img.save(p, quality=92)
            saved += 1
            print(f"已存 {p.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"共存 {saved} 张 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
