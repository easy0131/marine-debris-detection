"""Train UNet on JSON masks with fixed geographic validation and GPU support."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from improve_model import score_masks, self_check
from prepare_coastal import close
from train_baseline import BASE, MEAN, STD


def evaluation_data(data):
    images, masks, names = [], [], []
    for i in np.flatnonzero(data['splits'] == 'val'):
        # Fixed positions chosen without looking at masks; keep small positive targets.
        for y, x in ((0, 0), (0, 85), (85, 0), (85, 85)):
            images.append(data['images'][i, y:y + 256, x:x + 256])
            masks.append(data['masks'][i, y:y + 256, x:x + 256])
            names.append(f"{data['names'][i]}_y{y}_x{x}")
    return np.asarray(images), np.asarray(masks), names


def augment(image, mask, rng):
    side = int(rng.integers(224, 342))
    x, y = rng.integers(0, 342 - side, size=2)
    image = cv2.resize(image[y:y + side, x:x + side], (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask[y:y + side, x:x + side].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    k = int(rng.integers(4))
    image, mask = np.rot90(image, k), np.rot90(mask, k)
    if rng.random() < .5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    image = image.astype(np.float32) / 255
    gray = image.mean(2, keepdims=True)
    image = gray + (image - gray) * rng.uniform(.6, 1.4)
    image = (image - .5) * rng.uniform(.75, 1.25) + .5 + rng.uniform(-.12, .12)
    image *= rng.uniform(.9, 1.1, size=(1, 1, 3))
    image = np.clip(image, 0, 1) ** rng.uniform(.8, 1.25)
    if rng.random() < .35:
        image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(.4, 1.1)))
    return (image - MEAN) / STD, mask


@torch.inference_mode()
def predict(model, images, device, batch_size, tta=False):
    model.eval()
    predictions = []
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy((images[start:start + batch_size].astype(np.float32) / 255 - MEAN) / STD)
        x = x.permute(0, 3, 1, 2).to(device)
        p = model(x).softmax(1)[:, 1]
        if tta:
            for dims in ((2,), (3,), (2, 3)):
                p += model(x.flip(dims)).softmax(1)[:, 1].flip(tuple(d - 1 for d in dims))
            p /= 4
        predictions.append(p.cpu().numpy())
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'experiment_v2/data.npz')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'experiment_spatial')
    parser.add_argument('--init', type=Path, default=BASE, help='Use original provided weights to avoid holdout contamination')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--samples', type=int, default=512, help='Balanced samples per epoch')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 2 or args.batch_size < 1:
        parser.error('epochs >= 1, samples >= 2 and batch-size >= 1 required')
    self_check()
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    rng = np.random.default_rng(20260927)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. In Colab choose Runtime > Change runtime type > T4 GPU.')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'metrics.json').exists() and not args.resume:
        raise FileExistsError('Experiment exists; choose a new --out or --resume')
    with np.load(args.data) as archive:
        data = {k: archive[k] for k in archive.files}
    train = np.flatnonzero(data['splits'] == 'train')
    val = np.flatnonzero(data['splits'] == 'val')
    assert len(train) and len(val)
    for i in train:
        assert not any(close(data['bounds'][i], data['bounds'][j], 0) for j in val), 'Spatial leakage'
    positive = np.array([i for i in train if data['masks'][i].any()])
    negative = np.array([i for i in train if not data['masks'][i].any()])
    assert len(positive) and len(negative), 'Both classes required'
    vi, vm, vn = evaluation_data(data)
    # Also expose the exact validation inputs to the existing submission smoke checker.
    np.savez_compressed(args.out / 'data.npz', images=vi, masks=vm, names=vn, splits=['val'] * len(vn))
    model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=3, classes=2).to(device)
    checkpoint = torch.load(args.init, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 2e-5},
        {'params': list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), 'lr': 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 10.], device=device))
    metrics, best, start_epoch = [], -1., 1
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start_epoch = saved['epoch'] + 1
        metrics = saved['metrics']
        best = max(r['combined'] for r in metrics)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=device, train_images=len(train), train_positive=len(positive), validation_images=len(val),
                  validation_crops=len(vn), validation_positive=int(vm.any((1, 2)).sum()),
                  note='Spatial AIHub proxy. Chosen on validation, not an independent test or Public score.')
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config), flush=True)

    def evaluate(name, tta=False):
        nonlocal best
        p = predict(model, vi, device, args.batch_size, tta)
        np.save(args.out / (name + '_prob.npy'), p)
        rows = [dict(name=name, threshold=t, tta=tta, **score_masks(p > t, vm)) for t in (.25, .4, .5, .6, .75)]
        metrics.extend(rows)
        winner = max(rows, key=lambda r: r['combined'])
        if winner['combined'] > best:
            best = winner['combined']
            torch.save({'state_dict': model.state_dict()}, args.out / 'best.pt')
            (args.out / 'best.json').write_text(json.dumps(winner, indent=2), encoding='utf-8')
        (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(winner), flush=True)
        return winner

    if not args.resume:
        evaluate('provided_original')
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        # Small batches: update convolutional weights, preserve all BatchNorm running statistics.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        ids = np.r_[rng.choice(positive, args.samples // 2), rng.choice(negative, args.samples - args.samples // 2)]
        rng.shuffle(ids)
        losses = []
        factor = .5 * (1 + math.cos(math.pi * (epoch - 1) / max(args.epochs, 1)))
        for group, lr in zip(optimizer.param_groups, (2e-5, 3e-4)):
            group['lr'] = lr * (.1 + .9 * factor)
        for offset in range(0, len(ids), args.batch_size):
            examples = [augment(data['images'][i], data['masks'][i], rng) for i in ids[offset:offset + args.batch_size]]
            x = torch.from_numpy(np.stack([e[0] for e in examples])).permute(0, 3, 1, 2).to(device)
            y = torch.from_numpy(np.stack([e[1] for e in examples]).astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                logits = model(x)
                p, target = logits.float().softmax(1)[:, 1], y.float()
                dice = 1 - (2 * (p * target).sum((1, 2)) + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)
                has_target = target.sum((1, 2)) > 0
                loss = ce(logits, y) + (dice[has_target].mean() if has_target.any() else p.sum() * 0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; checkpoint has not been overwritten')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        evaluate(f'epoch{epoch}')
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None, metrics=metrics), args.out / 'last.pt')
        print(f'epoch {epoch}: loss={np.mean(losses):.4f}, seconds={time.time() - started:.1f}', flush=True)
    if args.tta:
        model.load_state_dict(torch.load(args.out / 'best.pt', map_location=device, weights_only=True)['state_dict'])
        evaluate('best_flip_tta', tta=True)
    print('SELECTED', (args.out / 'best.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
