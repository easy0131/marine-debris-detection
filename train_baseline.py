"""Fine-tune the provided UNet on the unambiguous AIHub drone masks."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
DATA = Path.home() / "Downloads/310.AI기반 국립공원 변화탐지 모니터링 플랫폼 구축/01-1.정식개방데이터/Validation"
IMAGES = DATA / "01.원천데이터/VS_01.Drone.zip"
MASKS = DATA / "02.라벨링데이터/VL_01.LABEL_01. Drone.zip"
LABELS = DATA / "02.라벨링데이터/VL_02.JSON_01. Drone.zip"
BASE = Path.home() / "Downloads/04_coastal debris_submission/04_coastal debris_submission/assets/model/unet_r18_debris_lite.pt"
OUT = HERE / "initial_unet_r18.pt"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def batch(images, masks, ids):
    x = np.stack([(images[i].astype(np.float32) / 255 - MEAN) / STD for i in ids])
    y = np.stack([masks[i] for i in ids]).astype(np.int64)
    return torch.from_numpy(x).permute(0, 3, 1, 2), torch.from_numpy(y)


def load_data():
    with zipfile.ZipFile(LABELS) as z:
        rows = []
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            doc = json.loads(z.read(name))
            debris = {f["properties"]["ANN_CD"] for f in doc["features"]
                      if f["properties"]["ANN_NM"].startswith("해안쓰레기")}
            if debris != {80}:  # 80 also means coniferous forest in this dataset.
                rows.append((doc["name"], debris == {70}))

    positive_groups = {stem.split("_")[0] for stem, positive in rows if positive}
    validation_groups = set(sorted(positive_groups)[::5])
    images, masks, splits, names = [], [], [], []
    with zipfile.ZipFile(IMAGES) as iz, zipfile.ZipFile(MASKS) as mz:
        for stem, positive in rows:
            group = stem.split("_")[0]
            if group not in positive_groups:
                continue  # Choose negatives from the same coastal tiles.
            with Image.open(io.BytesIO(iz.read("/" + stem + ".tif"))) as image, \
                 Image.open(io.BytesIO(mz.read("/" + stem + ".tif"))) as mask:
                a = image.tag_v2[33922]
                b = mask.tag_v2[33922]
                scale = image.tag_v2[33550]
                x0 = round((b[3] - a[3]) / scale[0])
                y0 = round((a[4] - b[4]) / scale[1])
                assert x0 in (0, 1) and y0 in (0, 1) and image.size == (1024, 1024) and min(mask.size) >= 1023
                rgb = np.asarray(image.convert("RGB"))[y0:y0 + 1023, x0:x0 + 1023]
                image_small = np.asarray(Image.fromarray(rgb).resize((341, 341), Image.Resampling.BOX))
                truth = (np.asarray(mask)[:1023, :1023] == 70).reshape(341, 3, 341, 3).mean(axis=(1, 3)) >= 0.5
            if positive and not truth.any():
                continue
            if not positive and truth.any():
                continue  # JSON and TIFF disagree; do not call this a negative.
            if positive:
                yy, xx = np.where(truth)
                x1 = int(np.clip((xx.min() + xx.max()) // 2 - 128, 0, 85))
                y1 = int(np.clip((yy.min() + yy.max()) // 2 - 128, 0, 85))
            else:
                x1 = y1 = 42
            image_patch = image_small[y1:y1 + 256, x1:x1 + 256]
            mask_patch = truth[y1:y1 + 256, x1:x1 + 256]
            if positive and mask_patch.sum() < 50:
                continue
            assert image_patch.shape == (256, 256, 3) and mask_patch.shape == (256, 256)
            images.append(image_patch)
            masks.append(mask_patch)
            splits.append("val" if group in validation_groups else "train")
            names.append(stem)
    assert {names[i].split("_")[0] for i, split in enumerate(splits) if split == "train"}.isdisjoint(validation_groups)
    print("prepared", len(names), "train", sum(s == "train" for s in splits),
          "val", sum(s == "val" for s in splits),
          "positive", sum(bool(m.any()) for m in masks), flush=True)
    return images, masks, splits, names


def boundary(mask):
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def nearby(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
                                 for dy in range(3) for dx in range(3)])


def measure(model, images, masks, ids):
    model.eval()
    pred, truth, shapes = [], [], []
    with torch.inference_mode():
        for start in range(0, len(ids), 8):
            part = ids[start:start + 8]
            x, _ = batch(images, masks, part)
            output = model(x).argmax(1).numpy().astype(bool)
            for idx, mask in zip(part, output):
                if mask.sum() < 50:
                    mask[:] = False
                target = masks[idx]
                pred.append(bool(mask.any()))
                truth.append(bool(target.any()))
                if target.any():
                    p, t = mask, target  # Official score compares mask areas, not just boundaries.
                    precision = (p & nearby(t)).sum() / max(1, p.sum())
                    recall = (t & nearby(p)).sum() / max(1, t.sum())
                    shapes.append(2 * precision * recall / max(1e-9, precision + recall))
    scores = []
    for label in (False, True):
        tp = sum(p == label and t == label for p, t in zip(pred, truth))
        fp = sum(p == label and t != label for p, t in zip(pred, truth))
        fn = sum(p != label and t == label for p, t in zip(pred, truth))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    result = {"macro_f1": sum(scores) / 2, "shape_f1_1px": float(np.mean(shapes)),
              "predicted_positive": sum(pred), "actual_positive": sum(truth)}
    result["combined"] = (result["macro_f1"] + result["shape_f1_1px"]) / 2
    return result


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    images, masks, splits, _ = load_data()
    train = np.array([i for i, s in enumerate(splits) if s == "train"])
    val = [i for i, s in enumerate(splits) if s == "val"]
    assert train.size and val and any(masks[i].any() for i in val)
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    baseline = measure(model, images, masks, val)
    print("baseline", baseline, flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 30.0]))
    history = {"baseline": baseline, "epochs": []}
    best = -1.0
    for epoch in range(2):
        model.train()
        model.encoder.eval()  # Keep BatchNorm statistics fixed with the frozen encoder.
        order = torch.randperm(len(train)).numpy()
        losses = []
        for start in range(0, len(train), 8):
            x, y = batch(images, masks, train[order[start:start + 8]])
            optimizer.zero_grad()
            logits = model(x)
            prob = logits.softmax(1)[:, 1]
            target = y.float()
            dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
            loss = ce(logits, y) + dice
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        score = measure(model, images, masks, val)
        score["epoch"] = epoch + 1
        score["train_loss"] = float(np.mean(losses))
        history["epochs"].append(score)
        print("epoch", epoch + 1, score, flush=True)
        if score["combined"] > best:
            best = score["combined"]
            torch.save({"state_dict": model.state_dict()}, OUT)
    (HERE / "initial_unet_r18_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    assert boundary(np.array([[False, False], [False, True]])).sum() == 1
    assert nearby(np.array([[True, False], [False, False]])).all()
    main()
