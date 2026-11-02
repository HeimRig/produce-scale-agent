# -*- coding: utf-8 -*-
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config
from app.vision.labels import load_labels

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class ProduceClassifier:
    def __init__(self, model_path=None, size=None):
        self.labels = load_labels()
        self.size = size or config.VISION_INPUT
        self.session = None
        self.input_name = None
        self.model_path = Path(model_path or config.VISION_MODEL)
        if self.model_path.exists():
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path), so, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name

    @property
    def ready(self):
        return self.session is not None

    def preprocess(self, image):
        img = image.convert("RGB")
        w, h = img.size
        s = config.VISION_RESIZE / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x, dtype=np.float32)

    def predict(self, image, topk=None):
        topk = topk or config.VISION_TOPK
        t0 = time.perf_counter()
        if self.session is None:
            preds = self._mock(image, topk)
        else:
            logits = self.session.run(None, {self.input_name: self.preprocess(image)})[0][0]
            prob = _softmax(np.asarray(logits, dtype=np.float32))
            idx = np.argsort(-prob)[:topk]
            preds = [
                {
                    "key": self.labels[i]["key"],
                    "cn": self.labels[i].get("cn", self.labels[i]["key"]),
                    "prob": float(prob[i]),
                }
                for i in idx
            ]
        for p in preds:
            p["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return preds

    def _mock(self, image, topk):
        digest = hashlib.md5(image.tobytes()).hexdigest()
        base = int(digest[:8], 16) % len(self.labels)
        out = []
        probs = [0.92, 0.05, 0.02]
        for i in range(min(topk, len(self.labels))):
            lb = self.labels[(base + i) % len(self.labels)]
            out.append(
                {"key": lb["key"], "cn": lb.get("cn", lb["key"]), "prob": probs[i] if i < 3 else 0.01}
            )
        return out
