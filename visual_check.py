"""Build a self-contained visual comparison for held-out coastal debris patches."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from train_baseline import BASE, HERE, OUT, batch, load_data


def data_uri(image, kind="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=kind, **({"quality": 86} if kind == "JPEG" else {}))
    return f"data:image/{kind.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def layer(mask, color):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, 170)
    return data_uri(Image.fromarray(rgba, "RGBA"))


def predict(path, images, masks, ids):
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    predictions = {}
    with torch.inference_mode():
        for start in range(0, len(ids), 8):
            part = ids[start:start + 8]
            x, _ = batch(images, masks, part)
            masks_out = model(x).argmax(1).numpy().astype(bool)
            for idx, mask in zip(part, masks_out):
                if mask.sum() < 50:
                    mask[:] = False
                predictions[idx] = mask
    return predictions


def category(truth, prediction):
    if truth.any():
        return "TP" if prediction.any() else "FN"
    return "FP" if prediction.any() else "TN"


def summary(cases, key):
    counts = {name: sum(c[key]["status"] == name for c in cases) for name in ("TP", "FP", "FN", "TN")}
    pos_f1 = 2 * counts["TP"] / max(1, 2 * counts["TP"] + counts["FP"] + counts["FN"])
    neg_f1 = 2 * counts["TN"] / max(1, 2 * counts["TN"] + counts["FP"] + counts["FN"])
    return {**counts, "macro_f1": round((pos_f1 + neg_f1) / 2, 3)}


def main():
    torch.set_num_threads(4)
    images, masks, splits, names = load_data()
    ids = [i for i, split in enumerate(splits) if split == "val"]
    old = predict(BASE, images, masks, ids)
    new = predict(OUT, images, masks, ids)
    cases = []
    for idx in ids:
        truth = masks[idx]
        baseline, candidate = old[idx], new[idx]
        false_positive = candidate & ~truth
        missed = truth & ~candidate
        errors = np.zeros((*truth.shape, 4), dtype=np.uint8)
        errors[false_positive] = (255, 72, 82, 185)
        errors[missed] = (123, 102, 255, 185)
        cases.append({
            "name": names[idx],
            "photo": data_uri(Image.fromarray(images[idx]), "JPEG"),
            "truth_layer": layer(truth, (60, 210, 120)),
            "baseline_layer": layer(baseline, (255, 175, 52)),
            "candidate_layer": layer(candidate, (52, 215, 232)),
            "error_layer": data_uri(Image.fromarray(errors, "RGBA")),
            "truth_pixels": int(truth.sum()),
            "baseline": {"status": category(truth, baseline), "pixels": int(baseline.sum())},
            "candidate": {"status": category(truth, candidate), "pixels": int(candidate.sum()),
                          "false_positive_pixels": int(false_positive.sum()), "missed_pixels": int(missed.sum())},
        })
    assert len(cases) == 43 and sum(c["truth_pixels"] > 0 for c in cases) == 15
    stats = {"baseline": summary(cases, "baseline"), "candidate": summary(cases, "candidate")}
    html = (HERE / "visual_check_template.html").read_text(encoding="utf-8")
    html = html.replace("__CASES__", json.dumps(cases, ensure_ascii=False))
    html = html.replace("__STATS__", json.dumps(stats, ensure_ascii=False))
    path = HERE / "visual_check.html"
    path.write_text(html, encoding="utf-8")
    print("saved", path, "cases", len(cases), "bytes", path.stat().st_size)
    print("summary", stats)


if __name__ == "__main__":
    assert category(np.array([[True]]), np.array([[False]])) == "FN"
    main()
