# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image


def ahash(image, size=8):
    g = image.convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    return (a > a.mean()).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def blur_score(image):
    try:
        import cv2

        g = np.asarray(image.convert("L"), dtype=np.uint8)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        g = np.asarray(image.convert("L"), dtype=np.float32)
        lap = (
            np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4.0 * g
        )
        return float(lap[1:-1, 1:-1].var())


def center_crop(image, ratio=0.8):
    w, h = image.size
    nw, nh = int(w * ratio), int(h * ratio)
    left, top = (w - nw) // 2, (h - nh) // 2
    return image.crop((left, top, left + nw, top + nh))


class Deduper:
    def __init__(self, threshold=5, size=8):
        self.threshold = threshold
        self.size = size
        self.hashes = []

    def is_dup(self, image, remember=True):
        h = ahash(image, self.size)
        for old in self.hashes:
            if hamming(h, old) <= self.threshold:
                return True
        if remember:
            self.hashes.append(h)
        return False
