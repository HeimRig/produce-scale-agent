# Produce Agent

树莓派上的果蔬计价秤。放上商品，它自己完成识别、查价、称重、计价、语音播报和小票打印。

同一套能力有三处出口：命令行状态机、LLM function calling、MCP 服务。

![架构](docs/architecture.svg)

实测（PC，i9-13900H）：

- 闭环 107 ms/单，指放货到小票生成；语音异步播放，不计入
- 视觉推理 2.5 ms/帧（P50），P95 3.1 ms
- 模型 6.39 MB，MobileNetV3-Small fp32
- 准确率：白底验证集 96.2%，实拍留出集 88.5%

## 快速开始

```bash
pip install -r requirements-pc.txt
python demo.py
```

不接硬件也能跑完整闭环，摄像头和秤走 mock，小票写到 `out/`。

模型文件不入库。可以直接从 [Releases](https://github.com/HeimRig/produce-scale-agent/releases/download/v1.0.0/produce_mobilenetv3s.onnx) 下载放进 `models/`，也可以按下面的步骤自己训练。

```bash
python -m pytest tests -q     # 16 个用例：计价、意图路由、闭环编排
```

## 实测数据

### 白底训练集换到实拍场景会失效

| 训练数据 | 白底验证集 | 实拍留出集 |
|---|---|---|
| 只用 Fruits-360（白底棚拍） | 97.5% | 38.0%（208 张） |
| 白底 + 约 700 张实拍混训 | 96.2% | 88.5%（261 张） |

骨干网和类别都没变，只改了训练数据构成。实拍从 38.0% 升到 88.5%，白底指标下降 1.3 个点。明细见 `docs/REALSHOT_REPORT.md`。

### INT8 量化在这个模型上不可用

| 方案 | 体积 | 与 fp32 预测一致率 |
|---|---|---|
| 静态 QDQ（Int8，MinMax 校准） | 2.03 MB | 15.0% |
| QOperator（Int8，per-channel） | 1.76 MB | 15.0% |
| 只量化 Conv / MatMul / Gemm | 2.06 MB | 20.0% |
| fp32（最终采用） | 6.39 MB | 100% |

动态量化更低，早期版本只有 10.3%。hard-swish 和 SE 模块对定点数敏感，三条路径的一致率都掉到 20% 以下，量化后的模型基本相当于随机猜。fp32 在 PC 上 P50 是 2.5 ms，按 6 倍估算在 Pi 5 上也在预算内，所以交付 fp32。要用 int8 需要做 QAT，见 `docs/ARCHITECTURE.md`。

## 设计说明

### 主链路是状态机，LLM 只做意图判断

结账流程是纯 Python 状态机，延迟低于 5 ms。大模型只负责判断"这个多少钱一斤"该调哪个工具。

本地 Ollama（qwen2.5-3B-Q4）实测单次解析 180 ms，冷启动首次请求 6 s，比收银链路慢三个数量级，所以它不参与交易流程。降级顺序是 Ollama、云端 API、关键词规则。

一个实测出来的细节：模型抽参数不可靠。让它处理"来两斤番茄"，会算成 200 克（应为 1000 克）；说"结账"，会凭空编出 `item_key=apple, grams=100`。所以最终是模型只输出 `action`，商品键和重量由正则加价目表匹配决定。

### 工具层是唯一出口

`AgentTools` 同时被状态机、function calling 和 MCP 消费。新增能力只改这一层，换 LLM 或换接入方式都不动业务代码。

### 精度优先

量化验证不过就不上线。见上面的一致性数据。

## 目录

```
produce-agent/
├── demo.py                    一键演示
├── config.py                  全局配置，电脑和 Pi 的差异都在这里
├── app/
│   ├── main.py                命令行入口：单次结账 / 循环 / 自然语言
│   ├── perception/            camera（picamera2 / webcam / mock）、scale（HX711 / mock）
│   ├── vision/                classifier（ONNX Runtime）、labels、imageops
│   ├── agent/                 tools（能力出口）、planner（Ollama / API / 规则）、pricing
│   └── action/                tts（edge / piper / pyttsx3）、printer（ESC/POS）
├── mcp_server.py              MCP 服务，8 个工具
├── scripts/                   数据下载、采集、清洗、训练、导出、压测、标定
├── models/                    模型与 labels.json（不入库）
└── docs/                      架构、数据集、性能、准确率、BOM、演示分镜
```

## 1. 准备数据

```bash
# 白底主力源，Fruits-360 的 113 类自动映射到本项目的 12 类
python scripts/fetch_dataset.py --source fruits360 --split train --per-class 300

# 补胡萝卜，Fruits-360 里没有这个类
python scripts/fetch_dataset.py --source fruidet --per-class 30

# 实拍源，用来消除白底与真实场景的差异
python scripts/fetch_dataset.py --source realshot  --split test  --per-class 40
python scripts/fetch_dataset.py --source realshot2 --split train --per-class 80
python scripts/fetch_dataset.py --source fruidet --only avocado --out-dir data/realshot_extra
python scripts/split_realshot.py            # 首次划分，留出 30%
python scripts/merge_realshot.py --move     # 之后新抓的图按 30/70 分
```

也可以自己拍（`CAMERA_BACKEND` 支持 `mock` / `webcam` / `picamera2`）：

```bash
python scripts/collect.py --source webcam --classes apple banana orange
python scripts/collect.py --source dir --dir D:\photos\apple --class apple
```

采集时会丢弃模糊帧和重复帧。实时采集用 8×8 aHash，汉明距离阈值 5；离线清洗用 16×16，阈值 1。阈值放太松会把整个类判成一张，早期用 8×8 阈值 5 跑黄瓜，60 张只剩 6 张。采集规范见 `docs/DATASET.md`。

## 2. 训练与导出

```bash
python scripts/prepare_dataset.py --clean   # 清洗并划分，实拍只进训练集
python scripts/finetune.py                  # 微调 8 轮，1000 张时 CPU 约 2 分钟
python scripts/export_onnx.py               # 导出 fp32，同时试量化并校验一致率
python scripts/bench.py                     # 压测，写 docs/PERF_REPORT.md
python scripts/eval_realshot.py             # 白底与实拍对比，写 docs/REALSHOT_REPORT.md
```

## 3. 自然语言入口

可选。装了 [Ollama](https://ollama.com) 就能用人话下指令，没装会自动退回关键词规则。

```bash
ollama pull qwen2.5:3b
python -m app.main --chat "苹果多少钱一斤"
python -m app.main --chat "把番茄称一下，500克"
```

`config.py` 里 `LLM_BACKEND` 可选 `auto` / `ollama` / `api` / `off`，`auto` 按 Ollama、云端 API、规则依次降级。

## 4. 树莓派部署

改 `config.py` 四项：

```python
CAMERA_BACKEND = "picamera2"
SCALE_BACKEND  = "hx711"
TTS_BACKEND    = "edge"      # 断网用 piper 或 pyttsx3
PRINTER_BACKEND = "mock"     # 接了打印机再改 serial / usb
```

```bash
sudo apt install -y python3-picamera2 python3-lgpio python3-venv
python3 -m venv --system-site-packages ~/pa-venv && source ~/pa-venv/bin/activate
pip install -r requirements-pi.txt
python scripts/calibrate.py        # 去皮，再放 500g 砝码标定，写入 scale.json
python -m app.main --loop
```

启动时打印的 `[后端]` 里出现 `Hx711LibScale`，说明称重驱动接上了。

U 盘拷贝，电脑端（在项目目录打开 PowerShell）：

```powershell
$src = (Get-Location).Path
$dst = "E:\produce-agent"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
robocopy $src $dst /E /XD out .git __pycache__ data /XF *.pyc
Copy-Item "$src\models\*.onnx","$src\models\labels.json" "$dst\models\" -ErrorAction SilentlyContinue
```

树莓派端：

```bash
lsblk
sudo mkdir -p /mnt/usb && sudo mount /dev/sda1 /mnt/usb
mkdir -p ~/produce-agent && cp -r /mnt/usb/produce-agent/. ~/produce-agent/
sudo umount /mnt/usb
```

## 5. MCP 接入

```bash
pip install mcp
python mcp_server.py
python scripts/mcp_smoke.py     # 冒烟测试，不用接客户端
```

暴露 `checkout`、`recognize_produce`、`get_price`、`weigh`、`make_quote`、`speak`、`print_receipt`、`list_items`。在 MCP 客户端里登记启动命令即可调用。

兼容 mcp 1.x 和 2.x。2.x 把 `FastMCP` 改名成 `MCPServer`，`mcp_server.py` 里做了 import 兜底。

## License

MIT
