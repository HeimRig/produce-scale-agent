# 实拍泛化测试

- 模型：`produce_mobilenetv3s.onnx`（12 类，MobileNetV3-Small）
- 白底集：`data/produce/val`（Fruits-360，与训练同源）
- 实拍集：`data/realshot_holdout`（真实背景，30% 留出，未参与训练）

| 数据集 | 张数 | Top-1 |
|---|---|---|
| 白底验证集 | 80 | 96.2% |
| 实拍图 | 261 | 88.5% |

## 分类别

| 类别 | 白底 | 实拍 |
|---|---|---|
| apple | 10/10 | 13/19 |
| avocado | 5/5 | 4/6 |
| banana | 7/8 | 21/22 |
| carrot | 4/4 | 23/24 |
| cucumber | 6/6 | 27/27 |
| grape | 9/10 | 29/29 |
| onion | 7/7 | 24/27 |
| orange | 6/6 | 15/19 |
| pear | 6/6 | 20/25 |
| pepper | 7/7 | 31/36 |
| strawberry | 6/6 | — |
| tomato | 4/5 | 24/27 |

## 实拍误判样例

| 文件 | 真实 | 预测 | 置信度 |
|---|---|---|---|
| real2_apple_0017.jpg | apple | grape | 41% |
| real2_apple_0031.jpg | apple | onion | 28% |
| real2_apple_0033.jpg | apple | tomato | 72% |
| real2_apple_0038.jpg | apple | onion | 44% |
| real2_apple_0050.jpg | apple | grape | 44% |
| real_apple_0005.jpg | apple | pear | 39% |
| fdet_avocado_0011.jpg | avocado | banana | 37% |
| fdet_avocado_0020.jpg | avocado | cucumber | 15% |

> 实拍集从同一数据源留出 30%，与训练侧的实拍图同分布但不同图片；
> 它不等于「自己拿手机在托盘上拍的」，后者才是最终验收口径。