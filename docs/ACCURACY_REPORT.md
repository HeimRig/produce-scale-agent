# 准确率与延迟（本地验证集）

- 验证集：80 张，12 类
- 模型：MobileNetV3-Small，12 类输出

| 后端 | Top-1 准确率 | 单张延迟 |
|---|---|---|
| torch | 96.2% | 615.8 ms |
| onnx-fp32 | 96.2% | 37.9 ms |
| onnx-int8 | 18.8% | 39.4 ms |

## 分类别（int8）

| 类别 | 张数 | 正确 |
|---|---|---|
| apple | 10 | 0 |
| avocado | 5 | 0 |
| banana | 8 | 4 |
| carrot | 4 | 0 |
| cucumber | 6 | 0 |
| grape | 10 | 10 |
| onion | 7 | 0 |
| orange | 6 | 0 |
| pear | 6 | 0 |
| pepper | 7 | 0 |
| strawberry | 6 | 0 |
| tomato | 5 | 1 |

> 验证集与训练集同源（都是 Fruits-360 白底棚拍），准确率会明显偏高，不能当论文里的现场指标。