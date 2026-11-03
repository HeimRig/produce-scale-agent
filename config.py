# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODE = "mock"                  # mock=电脑调试，pi=树莓派实机

CAMERA_BACKEND = "mock"        # mock | webcam | picamera2
CAMERA_SIZE = (1280, 720)
WEBCAM_INDEX = 0
MOCK_IMAGE = ROOT / "assets" / "sample.jpg"
MOCK_IMAGE_DIR = ROOT / "assets" / "samples"

RAW_DATA_DIR = ROOT / "data" / "raw"
DATASET_DIR = ROOT / "data" / "produce"

SCALE_BACKEND = "mock"         # mock | hx711
HX711_CHIP = 0
HX711_DOUT = 6                 # BCM6，按自己接线改
HX711_SCK = 5                  # BCM5，按自己接线改
HX711_REF_UNIT = 428.0         # counts/克，标定值
HX711_OFFSET = 8300000.0       # 空载 counts
SCALE_JSON = ROOT / "scale.json"   # 本项目的标定文件，scripts/calibrate.py 生成
SCALE_SAMPLES = 7
SCALE_SETTLE_TOL_G = 0.5

VISION_MODEL = ROOT / "models" / "produce_mobilenetv3s.onnx"
VISION_LABELS = ROOT / "models" / "labels.json"
VISION_INPUT = 224
VISION_RESIZE = 256            # 短边缩放到 256 再中心裁 224
VISION_TOPK = 3
CONF_THRESHOLD = 0.55          # top1 低于此值判为不确定（scripts/threshold_curve.py 联合标定）
MARGIN_THRESHOLD = 0.10        # top1-top2 低于此值判为两个候选分不开
UNKNOWN_DIR = ROOT / "data" / "unknown"
MIN_WEIGHT_G = 5.0

PRICE_TABLE = ROOT / "app" / "data" / "prices.json"

TTS_BACKEND = "auto"           # auto | edge | piper | pyttsx3 | mock
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_MAX_CONCURRENT = 3         # 同时最多几个播报线程，超出的直接丢弃
PIPER_MODEL = ROOT / "models" / "zh_CN-huayan-medium.onnx"
AUDIO_PLAYER = "aplay"
AUDIO_PLAYER_WIN = "ffplay"

# LLM 语义层（旁路，不进主链路）
LLM_BACKEND = "auto"           # auto | ollama | api | off
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen25-3b:latest"
OLLAMA_TIMEOUT_S = 20
API_BASE = "https://api.deepseek.com/v1"
API_MODEL = "deepseek-chat"
API_KEY_ENV = "DEEPSEEK_API_KEY"
API_TIMEOUT_S = 15

PRINTER_BACKEND = "mock"       # mock | serial | usb
PRINTER_PORT = "/dev/ttyUSB0"
PRINTER_BAUDRATE = 9600
PRINTER_USB_VENDOR = 0x0416
PRINTER_USB_PRODUCT = 0x5011
PRINTER_CHARSET = "gb18030"
PRINTER_WIDTH = 32

OUT_DIR = ROOT / "out"
SALES_JSONL = OUT_DIR / "sales.jsonl"
