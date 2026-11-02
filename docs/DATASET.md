# 数据采集与标注规范

## 1. 目标规模

| 项 | 数值 |
|---|---|
| 类别 | 12 类（apple / banana / orange / pear / grape / watermelon / strawberry / kiwi / tomato / cucumber / potato / carrot） |
| 每类训练集 | 200 张起，够跑到 95%+ |
| 每类验证集 | 40 张 |
| 负样本 | 空盘、手入镜、杂物各 30 张 |
| 合计 | 约 3000 张，采集 4–6 小时 |

低于 40 张的类别不要进训练，模型会学成"猜多数类"。

## 2. 采集维度（决定泛化，不是越多越好）

| 维度 | 要覆盖的变化 |
|---|---|
| 光照 | 室内白光、窗边侧光、暖色灯（三种各占 1/3） |
| 角度 | 俯视 0°、斜 30°、斜 60° |
| 背景 | 托盘内（主场）、纸箱、木桌、白纸 |
| 摆放 | 单个、两个相邻、部分被遮挡 |
| 距离 | 近（充满画面）、中、远（占画面 1/4） |
| 品质 | 新鲜、有斑点、轻微磕碰（别只拍完美果） |

**一条硬规则：同一颗果子只能出现在 train 或 val 之一。** 按果子分组采，先分好再拍；否则同一颗果子换个角度进两个集合，准确率虚高 10 个点以上，答辩被问"怎么划分的"就露馅。

## 3. 操作流程

```bash
# 1. 电脑接 USB 摄像头，采一批
python scripts/collect.py --source webcam --classes apple banana orange

# 2. 手机拍的导入
python scripts/collect.py --source dir --dir D:\photos\apple --class apple

# 3. 清洗 + 划分 + 出统计
python scripts/prepare_dataset.py --clean

# 4. 训练
python scripts/finetune.py

# 5. 导出 INT8
python scripts/export_onnx.py
```

`collect.py` 采集时自动做两件事：拉普拉斯方差 < 60 判为模糊直接丢弃；aHash 汉明距离 ≤ 5 判为重复帧丢弃。避免一口气连拍 200 张几乎一样的图。

统计表写到 `docs/DATASET_REPORT.md`，低于阈值的类别会标 ⚠️。

## 4. 网上现成图集

自采之前先跑这一步，能立刻拿到几千张可用图：

```bash
# 主力源：Fruits-360（HF 镜像，113 类 9 万张，白底棚拍）
python scripts/fetch_dataset.py --source fruits360 --split train --per-class 300

# 补充源：真实背景小样（19 类每类 30 张，实测只有 7 类有图）
python scripts/fetch_dataset.py --source fruidet --per-class 30
```

脚本会按 12 个目标类别做关键词映射（`Apple Braeburn → apple`、`Grape Blue → grape`、`Clementine → orange`），落盘到 `data/raw/<类别>/`，并把"哪一类一张都没捞到"打印出来。

| 源 | 实测规模 | 场景 | 覆盖情况 |
|---|---|---|---|
| `PedroSampaio/fruits-360`（HF） | 训练 67k / 测试 22,688 张，113 类，100×100 | 白底棚拍 | 11/12 类，**没有胡萝卜** |
| `sirunchained/fruit-dataset`（HF） | 210 张，19 类目录但只有 7 类有文件 | 真实背景 | 苹果/香蕉/胡萝卜/葡萄等，每类仅 30 张 |
| Fruits-360 官方（Kaggle `moltean/fruits`） | 36 万张，最大 | 白底 + 原图 | 全，但要 Kaggle 账号 |

**已验证的两个坑**：

1. **Fruits-360 里没有 carrot。** 113 个类目从 `Apple Braeburn` 到 `Watermelon`，整份没有胡萝卜。要么用 FruitDet 补 30 张（够跑通，不够稳），要么自己拍 100 张（半小时，效果最好）。
2. **HF 直连慢或被墙。** 脚本默认先试 `hf-mirror.com`，失败再回退 `huggingface.co`；也可以自己设 `HF_ENDPOINT`。

**代价要提前说清**：Fruits-360 是白底棚拍，和你的托盘场景差得远。全用它训练，模型会学"白底 = 有果子"，现场换个木桌就崩。正确用法是**它占每类 70%，自采占 30%**，训练时开强增强（`RandomResizedCrop(0.65~1.0)` + `ColorJitter(0.25)`，`finetune.py` 里已配好）。

License：Fruits-360 是 CC BY-SA 4.0，README 和报告里要注明出处；FruitDet 是 CC BY 4.0。

## 5. 公开数据集补位细节

自采不够时用这两个补齐，只需把类别名对齐：

| 数据集 | 规模 | 可用类别 | 获取 |
|---|---|---|---|
| Fruits-360 | 9 万张，141 类，白底 | apple / banana / orange / pear / tomato / cucumber / potato / carrot 等 | Kaggle `moltean/fruits` |
| Fruit and Vegetable Image Recognition | 2.1 万张，36 类，真实背景 | 覆盖面更广，含 grape / watermelon / kiwi | Kaggle `kritikseth/fruit-and-vegetable-image-recognition` |

注意：Fruits-360 是**白底棚拍**，跟你的托盘场景差得远。直接混进去会让模型过拟合白底——建议它只占每类总量的 30%，且训练时开强数据增强（随机裁剪 0.65–1.0、颜色抖动 0.25）。

## 5. 常见坑

1. **同一批次连续拍摄**：光照、角度几乎不变，等于只采了一张。每 50 张换一次环境。
2. **只拍好果子**：现场有磕碰、有斑点，模型没见过就错。刻意拍一部分"丑"的。
3. **验证集混入训练集同类果子**：见上，最致命。
4. **忘了负样本**：空盘时模型也会自信地报"苹果 92%"。空盘/手入镜的负样本能让拒识逻辑真正生效。
5. **类别名用中文**：ImageFolder 用目录名做类名，目录名必须是英文键（apple…），中文由 `models/labels.json` 映射。

## 6. 实测结果（2026-11-01，每类限 60 张）

| 类别 | 原始 | 保留 | 剔除模糊 | 剔除重复 | train | val |
|---|---|---|---|---|---|---|
| apple | 90 | 71 | 5 | 14 | 61 | 10 |
| banana | 90 | 63 | 14 | 13 | 54 | 9 |
| carrot | 30 | 29 | 1 | 0 | 25 | 4 |
| cucumber | 60 | 40 | 0 | 20 | 34 | 6 |
| grape | 90 | 67 | 4 | 19 | 57 | 10 |
| kiwi | 60 | 33 | 0 | 27 | 29 | 4 |
| orange | 60 | 43 | 0 | 17 | 37 | 6 |
| pear | 60 | 39 | 0 | 21 | 34 | 5 |
| potato | 60 | 39 | 0 | 21 | 34 | 5 |
| strawberry | 60 | 45 | 0 | 15 | 39 | 6 |
| tomato | 60 | 40 | 0 | 20 | 34 | 6 |
| watermelon | 60 | 52 | 0 | 8 | 45 | 7 |

合计 train 483 / val 78。这是"每类 60 张"的最小验证集，够跑通训练和出第一版 ONNX，不够出答辩数字。正式跑把 `--per-class` 调到 300。

**踩过的坑**：一开始用 8×8 aHash + 汉明距离 ≤5 去重，黄瓜 60 张只剩 6 张、猕猴桃只剩 3 张——白底图上同类果子的 8×8 缩略图几乎一模一样，阈值太松等于把整类判成一张。改成 **16×16 哈希 + ≤1** 后，只剔掉真正近乎复制（Fruits-360 的旋转重复帧）的约 25%，保留合理。

## 7. 当前状态

- [x] 12 类网上图集已落盘（`data/raw/`），胡萝卜 30 张偏少
- [x] 清洗 + 划分 + 统计跑通（`docs/DATASET_REPORT.md`）
- [ ] 每类补到 ≥200（`--per-class 300` 重跑 fetch）
- [ ] 负样本（空盘 / 手入镜 / 杂物各 30）
- [ ] 自采 30% 实景混入，压白底过拟合
