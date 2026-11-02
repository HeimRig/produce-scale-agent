# -*- coding: utf-8 -*-
import time
import random
from pathlib import Path

import config


class MockCamera:
    def __init__(self, image_path=None, image_dir=None):
        from PIL import Image

        self.dir = Path(image_dir or config.MOCK_IMAGE_DIR)
        self.fixed = Path(image_path or config.MOCK_IMAGE)
        self._images = []
        if self.dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self._images = sorted([p for p in self.dir.iterdir() if p.suffix.lower() in exts])
        if not self._images and not self.fixed.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", config.CAMERA_SIZE, (235, 235, 235)).save(self.fixed)
        self._idx = 0

    def capture(self):
        from PIL import Image

        time.sleep(0.05)
        if self._images:
            p = self._images[self._idx % len(self._images)]
            self._idx += 1
            return Image.open(p).convert("RGB")
        return Image.open(self.fixed).convert("RGB")

    def close(self):
        pass


class PiCamera:
    def __init__(self, size=None):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(
            main={"size": size or config.CAMERA_SIZE, "format": "RGB888"}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.6)

    def capture(self):
        from PIL import Image

        return Image.fromarray(self.picam2.capture_array())

    def close(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass


class WebcamCamera:
    def __init__(self, index=None, size=None):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(config.WEBCAM_INDEX if index is None else index)
        if not self.cap.isOpened():
            raise RuntimeError("摄像头打不开，检查 WEBCAM_INDEX 或被其他程序占用")
        w, h = size or config.CAMERA_SIZE
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def capture(self):
        from PIL import Image

        for _ in range(3):
            ok, frame = self.cap.read()
            if ok:
                return Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            time.sleep(0.05)
        raise RuntimeError("取帧失败")

    def close(self):
        self.cap.release()


def build_camera():
    if config.CAMERA_BACKEND == "picamera2":
        return PiCamera()
    if config.CAMERA_BACKEND == "webcam":
        return WebcamCamera()
    return MockCamera()
