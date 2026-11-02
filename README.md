# Produce Agent · 边缘具身果蔬计价秤

一台能自己完成交易的秤：**看见是什么 → 查价 → 称重 → 算钱 → 说出来 → 打票**。
整条 `perception → decision → action` 闭环跑在树莓派上，能力同时暴露为 Python 工具、LLM function calling 和 MCP 服务。

| 闭环延迟 | 视觉推理 | 模型体积 | 实拍准确率 |
|---|---|---|---|
| **107 ms / 单** | 2.5 ms / 帧（P50，P95 3.1 ms） | 6.39 MB | **88.5%**（261 张留出） |

> 闭环 = 放货到小票吐出；语音播报走异步线程，不阻塞交易。

![架构](docs/architecture.svg)

## 30 秒跑起来

```bash
pip install -r requirements-pc.txt
python demo.py
```

不需要接硬件：摄像头与秤走 mock，小票落盘到 `out/`。装上真模型后识别走 ONNX 推理。

自检：`python -m pytest tests -q`（16 个用例，覆盖计价 / 路由 / 闭环编排）。

> 模型文件不入库（6.4 MB）。可以直接从 Releases 下载 `produce_mobilenetv3s.onnx` 放进 `models/`，也可以跑 `scripts/fetch_dataset.py` + `scripts/finetune.py` + `scripts/export_onnx.py` 自己生成，见下文。

## 结果

**域适应：白底训练集在真实场景会崩，这是本项目最有价值的一组数据**

| 训练数据 | 白底验证集 | 实拍留出集 |
|---|---|---|
| 只用 Fruits-360（白底棚拍） | 97.5% | **38.0%**（208 张） |
| 白底 + 约 700 张实拍混训 | 96.2% | **88.5%**（261 张） |

同一骨干网、同一套类别，只改训练数据构成，实拍场景从 38% 提到 88.5%，白底指标基本没退步。详见 `docs/REALSHOT_REPORT.md`。

**量化的结论是反的，但结论本身是结论**

| 方案 | 体积 | 与 fp32 预测一致率 |
|---|---|---|
| 静态 QDQ（Int8，MinMax 校准） | 2.03 MB | 15.0% |
| QOperator（Int8，per-channel） | 1.76 MB | 15.0% |
| 只量化 Conv / MatMul / Gemm | 2.06 MB | 20.0% |
| **fp32（采用）** | **6.39 MB** | **100%** |

动态量化更差（早期版本只有 10.3%）。hard-swish 与 SE 模块对定点数敏感，三条量化路径一致率都掉到 20% 以下，等于把模型打回随机猜。fp32 在 PC 上 P50 只有 2.5 ms，Pi 5 上按 6 倍算仍在预算内。想冲 int8 得上 QAT，见 `docs/ARCHITECTURE.md`。

## 三个设计取舍

**1. 主链路用确定性状态机，大模型只走旁路。**
1.5B 模型在 Pi 5 上 int4 只有 3–5 tok/s，一次工具往返 8–15 秒，实时收银不能用它。所以结账流程是纯 Python 状态机（< 5 ms），大模型只负责把「这个多少钱一斤」解析成工具调用。降级链：本地模型 → 云端 API → 关键词规则。

**2. 工具层是唯一能力出口。**
`AgentTools` 同时被状态机、LLM function calling、MCP 服务三处消费。加能力只改一处，换 LLM 或换接入方式不动业务代码。

**3. 先保证准，再谈省。**
量化掉点验证不过就不上线，宁可交 6.39 MB 的 fp32。

## 目录

```
produce-agent/
├── demo.py                    一键演示
├── config.py                  全局配置（电脑/Pi 切换只改这里）
├── app/
│   ├── main.py                CLI：单次结账 / 循环 / 自然语言
│   ├── perception/            camera（picamera2 / webcam / mock）、scale（HX711 / mock）
│   ├── vision/                classifier（ONNX Runtime）、labels、imageops（去重/模糊判定）
│   ├── agent/                 tools（能力出口）、planner（LLM 路由）、pricing
│   └── action/                tts（edge/piper/pyttsx3）、printer（ESC/POS）
├── mcp_server.py              MCP 工具服务（8 个工具）
├── scripts/                   数据下载/采集/清洗/训练/导出/压测/标定
├── models/                    模型与 labels.json（不入库）
└── docs/                      ARCHITECTURE / POSITIONING / DATASET / PERF / ACCURACY / REALSHOT / BOM / DEMO_RECORD / DEMO_SCRIPT
```

## 一、拿数据

```bash
# 白底主力源：Fruits-360（113 类，按本项目 12 类自动映射）
python scripts/fetch_dataset.py --source fruits360 --split train --per-class 300

# 补胡萝卜（Fruits-360 里没有）
python scripts/fetch_dataset.py --source fruidet --per-class 30

# 实拍源（真实背景，用来解决域差异）
python scripts/fetch_dataset.py --source realshot  --split test  --per-class 40   # 测试分片
python scripts/fetch_dataset.py --source realshot2 --split train --per-class 80   # 训练分片
python scripts/fetch_dataset.py --source fruidet --only avocado --out-dir data/realshot_extra
python scripts/split_realshot.py            # 首次：从 realshot 留出 30%
python scripts/merge_realshot.py --move     # 之后：新抓的图按 30/70 分进留出集与训练集
```

自己拍也行（`CAMERA_BACKEND` 支持 `mock` / `webcam` / `picamera2`）：

```bash
python scripts/collect.py --source webcam --classes apple banana orange
python scripts/collect.py --source dir --dir D:\photos\apple --class apple
```

采集时自动丢弃模糊帧与重复帧：网页实时采集用 8×8 aHash（汉明距离 ≤ 5，容忍连拍），离线清洗用 16×16（≤ 1，只剔近乎复制的）。阈值太松会把整类判成一张——8×8/5 曾在黄瓜 60 张里只留下 6 张。采集规范（光照/角度/背景怎么变、同一颗果子为什么不能跨集合）见 `docs/DATASET.md`。

## 二、训练与导出

```bash
python scripts/prepare_dataset.py --clean   # 清洗 + 划分（实拍只进 train，不进 val）
python scripts/finetune.py                  # 微调 8 轮，约 1000 张时 CPU 2 分钟
python scripts/export_onnx.py               # 导出 fp32 + 试量化并校验一致率
python scripts/bench.py                     # 性能压测 → docs/PERF_REPORT.md
python scripts/eval_realshot.py             # 白底 vs 实拍对比 → docs/REALSHOT_REPORT.md
```

## 三、树莓派部署

`config.py` 改四项：

```python
CAMERA_BACKEND = "picamera2"
SCALE_BACKEND  = "hx711"
TTS_BACKEND    = "edge"      # 有网用 edge，断网用 piper/pyttsx3
PRINTER_BACKEND = "mock"     # 有打印机再改 serial / usb
```

```bash
sudo apt install -y python3-picamera2 python3-lgpio python3-venv
python3 -m venv --system-site-packages ~/pa-venv && source ~/pa-venv/bin/activate
pip install -r requirements-pi.txt
python scripts/calibrate.py        # 去皮 + 500g 砝码标定，写 scale.json
python -m app.main --loop
```

启动时打印的 `[后端]` 里若是 `Hx711LibScale`，说明称重驱动接对了。

### U 盘拷贝（电脑 → Pi）

```powershell
$src = (Get-Location).Path        # 在 produce-agent 目录里打开 PowerShell
$dst = "E:\produce-agent"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
robocopy $src $dst /E /XD out .git __pycache__ data /XF *.pyc
Copy-Item "$src\models\*.onnx","$src\models\labels.json" "$dst\models\" -ErrorAction SilentlyContinue
Get-ChildItem $dst -Recurse -File | Measure-Object -Property Length -Sum
```

```bash
lsblk
sudo mkdir -p /mnt/usb && sudo mount /dev/sda1 /mnt/usb
mkdir -p ~/produce-agent && cp -r /mnt/usb/produce-agent/. ~/produce-agent/
sudo umount /mnt/usb
```

## 四、MCP 接入

```bash
pip install mcp
python mcp_server.py
python scripts/mcp_smoke.py     # 冒烟测试，不用接客户端
```

暴露 `checkout`、`recognize_produce`、`get_price`、`weigh`、`make_quote`、`speak`、`print_receipt`、`list_items` 共 8 个工具。在支持 MCP 的客户端登记命令，任何 Agent 都能直接调这台秤。

> 兼容 mcp 1.x / 2.x：2.x 把 `FastMCP` 改名成 `MCPServer`，`mcp_server.py` 里做了 import 兜底。

## 交付物

| 项 | 状态 |
|---|---|
| 闭环（真实模型端到端） | ✅ 107 ms/单 |
| 视觉模型 | ✅ 12 类，白底 96.2% / 实拍 88.5%，fp32 6.39 MB |
| 数据链路 | ✅ 下载 / 采集 / 清洗 / 划分 / 统计 |
| 单元测试 | ✅ 16 个用例（pytest），覆盖计价、意图路由、闭环编排 |
| 性能报告 | ✅ `docs/PERF_REPORT.md`（PC 实测，Pi 待补） |
| 硬件 BOM | ✅ `docs/BOM.md` |
| 演示视频 | ⬜ 分镜见 `docs/DEMO_SCRIPT.md` |

**没有硬件的推进路径**：打印机 mock 落盘小票、系统 TTS 出声、USB 摄像头取帧，整条闭环在电脑上可完整演示；采购只影响"实物感"，不影响逻辑。
