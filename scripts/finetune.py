# -*- coding: utf-8 -*-
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "produce"
OUT_PT = ROOT / "models" / "produce_mobilenetv3s.pt"
OUT_LABELS = ROOT / "models" / "labels.json"

EPOCHS = 8
BATCH_SIZE = 32
LR = 3e-4
IMG_SIZE = 224
WORKERS = 0
SEED = 42

ARCH = "mobilenet_v3_small"    # mobilenet_v3_small | mobilenet_v2

CN_MAP = {
    "apple": "苹果", "avocado": "牛油果", "banana": "香蕉", "carrot": "胡萝卜",
    "cucumber": "黄瓜", "grape": "葡萄", "onion": "洋葱", "orange": "橙子",
    "pear": "梨", "pepper": "辣椒", "strawberry": "草莓", "tomato": "番茄",
}

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(arch, n_classes, pretrained=True):
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
        return net
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, n_classes)
    return net


WEIGHT_URLS = {
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


def load_backbone(arch, n_classes, tries=3):
    for i in range(tries):
        try:
            net = build_model(arch, n_classes, pretrained=True)
            print(f"{arch} 预训练权重加载成功（第 {i+1} 次）")
            return net
        except Exception as e:
            print(f"预训练权重第 {i+1} 次失败：{str(e)[:120]}")
            try:
                import urllib.request

                url = WEIGHT_URLS[arch]
                p = OUT_PT.parent / "_weights" / url.split("/")[-1]
                p.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                    f.write(r.read())
                sd = torch.load(p, map_location="cpu")
                net = build_model(arch, n_classes, pretrained=False)
                net.load_state_dict(sd, strict=False)
                print(f"手动下载权重成功：{p.stat().st_size/1e6:.1f} MB")
                return net
            except Exception as e2:
                print(f"手动下载也失败：{str(e2)[:120]}")
    print("全部失败，改为从零训练")
    return build_model(arch, n_classes, pretrained=False)


def main():
    torch.manual_seed(SEED)
    train_ds = datasets.ImageFolder(DATA_DIR / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", val_tf)
    classes = train_ds.classes
    print("类别:", classes)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    model = load_backbone(ARCH, len(classes))
    for p in list(model.parameters())[:14]:
        p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f"epoch {epoch+1}/{EPOCHS} loss={loss_sum/len(train_ds):.4f} val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": ARCH}, OUT_PT)

    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    labels = [{"key": c, "cn": CN_MAP.get(c, c), "en": c} for c in classes]
    OUT_LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best val_acc={best:.4f} -> {OUT_PT}")


if __name__ == "__main__":
    main()
