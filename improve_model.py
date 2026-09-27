"""Reproducible CPU fine-tuning; official mask-area score, grouped validation."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

from train_baseline import HERE, BASE, batch, load_data, nearby

OUT = HERE / 'experiment_20260927'


def score_masks(predictions, targets):
    confusion = np.zeros((2, 2), dtype=int)
    shapes = []
    for prediction, target in zip(predictions, targets):
        prediction = prediction.copy()
        if prediction.sum() < 50:
            prediction[:] = False
        confusion[int(target.any()), int(prediction.any())] += 1
        if target.any():
            precision = (prediction & nearby(target)).sum() / max(1, prediction.sum())
            recall = (target & nearby(prediction)).sum() / target.sum()
            shapes.append(2 * precision * recall / max(1e-9, precision + recall))
    f1 = [2 * confusion[i, i] / max(1, confusion[i].sum() + confusion[:, i].sum()) for i in (0, 1)]
    shape = float(np.mean(shapes)) if shapes else 0.0
    return dict(macro_f1=float(np.mean(f1)), shape_f1=shape,
                combined=float((np.mean(f1) + shape) / 2), confusion=confusion.tolist())


@torch.inference_mode()
def probabilities(model, images, masks, ids, tta=False):
    model.eval()
    outputs = []
    for start in range(0, len(ids), 8):
        x, _ = batch(images, masks, ids[start:start + 8])
        p = model(x).softmax(1)[:, 1]
        if tta:
            for dims in ((2,), (3,), (2, 3)):
                p += model(x.flip(dims)).softmax(1)[:, 1].flip(tuple(d - 1 for d in dims))
            p /= 4
        outputs.append(p.numpy())
    return np.concatenate(outputs)


def self_check():
    target = np.zeros((16, 16), dtype=bool)
    target[3:12, 3:12] = True
    empty = np.zeros_like(target)
    assert score_masks([target, empty], [target, empty])['combined'] == 1
    assert score_masks([np.roll(target, 1, axis=0), empty], [target, empty])['shape_f1'] == 1
    assert score_masks([empty], [target])['shape_f1'] == 0
    small = empty.copy()
    small[:7, :7] = True
    assert score_masks([small], [target])['confusion'] == [[0, 0], [1, 0]]


def main():
    self_check()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    rng = np.random.default_rng(20260927)
    cache = OUT / 'data.npz'
    if not cache.exists():
        images, masks, splits, names = load_data()
        np.savez_compressed(cache, images=images, masks=masks, splits=splits, names=names)
    data = np.load(cache)
    images, masks = data['images'], data['masks']
    train = np.flatnonzero(data['splits'] == 'train')
    val = np.flatnonzero(data['splits'] == 'val')
    assert set(n.split('_')[0] for n in data['names'][train]).isdisjoint(n.split('_')[0] for n in data['names'][val])
    model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=3, classes=2)
    results = []
    best = -1

    def evaluate(name, tta=False):
        nonlocal best
        p = probabilities(model, images, masks, val, tta)
        np.save(OUT / (name + '_prob.npy'), p)
        for threshold in (0.25, 0.35, 0.5, 0.65, 0.75):
            row = dict(name=name, threshold=threshold, tta=tta, **score_masks(p > threshold, masks[val]))
            results.append(row)
            if row['combined'] > best:
                best = row['combined']
                torch.save({'state_dict': model.state_dict()}, OUT / 'best.pt')
                (OUT / 'best.json').write_text(json.dumps(row, indent=2))
        winner = max(results[-5:], key=lambda r: r['combined'])
        print(json.dumps(winner), flush=True)
        (OUT / 'metrics.json').write_text(json.dumps(results, indent=2))

    model.load_state_dict(torch.load(BASE, map_location='cpu', weights_only=True)['state_dict'])
    evaluate('original')
    model.load_state_dict(torch.load(HERE / 'initial_unet_r18.pt', map_location='cpu', weights_only=True)['state_dict'])
    evaluate('initial')
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 10.]))
    for epoch in range(1, 7):
        started = time.time()
        model.train()
        model.encoder.eval()
        losses = []
        for ids in np.array_split(rng.permutation(train), int(np.ceil(len(train) / 8))):
            x, y = batch(images, masks, ids)
            k = int(rng.integers(4))
            x, y = x.rot90(k, (2, 3)), y.rot90(k, (1, 2))
            if rng.random() < .5:
                x, y = x.flip((3,)), y.flip((2,))
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            p, target = logits.softmax(1)[:, 1], y.float()
            dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
            loss = ce(logits, y) + dice
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        evaluate(f'epoch{epoch}')
        print(f'epoch {epoch}: loss={np.mean(losses):.4f}, seconds={time.time()-started:.1f}', flush=True)
    model.load_state_dict(torch.load(OUT / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
    evaluate('best_flip_tta', tta=True)
    print('DONE', (OUT / 'best.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
