"""Train UNet on JSON masks with fixed geographic validation and GPU support."""
import argparse
import hashlib
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
    if mask.any() and rng.random() < .75:
        # Sample a foreground pixel but randomize its position; validation never uses this sampler.
        yy, xx = np.where(mask)
        j = int(rng.integers(len(xx)))
        x = int(rng.integers(max(0, xx[j] - side + 1), min(xx[j], 341 - side) + 1))
        y = int(rng.integers(max(0, yy[j] - side + 1), min(yy[j], 341 - side) + 1))
    image = cv2.resize(image[y:y + side, x:x + side], (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask[y:y + side, x:x + side].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
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


def refinement_loss(logits, target):
    """Weight positive patches equally, matching the patch-averaged shape metric."""
    p = logits.float().softmax(1)[:, 1]
    truth = target.float()
    area = truth.sum((1, 2))
    dice = 1 - (2 * (p * truth).sum((1, 2)) + 1) / (p.sum((1, 2)) + area + 1)
    positive = area > 0
    return torch.nn.functional.cross_entropy(logits, target, weight=logits.new_tensor([1., 10.])) + (dice * positive).sum() / positive.sum().clamp_min(1)


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
    parser.add_argument('--refine', action='store_true', help='Low-rate refinement with hard negatives and per-patch Dice')
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
    with args.data.open('rb') as file:
        data_sha256 = hashlib.file_digest(file, 'sha256').hexdigest()
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
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 30.], device=device))
    metrics, best, start_epoch = [], -1., 1
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location=device, weights_only=True)
        if saved.get('data_sha256') != data_sha256:
            raise ValueError('Resume dataset changed or lacks a fingerprint; use a new --out directory')
        if saved.get('refine', False) != args.refine:
            raise ValueError('Resume training mode changed; use a new --out directory')
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
    hard_negative = negative
    if args.refine:
        # ponytail: center-crop mining is cheap; mine all corners if edge-only false positives dominate.
        torch_rng = torch.get_rng_state()
        mining_model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=3, classes=2).to(device)
        mining_model.load_state_dict(checkpoint['state_dict'])
        probability = predict(mining_model, data['images'][negative, 42:298, 42:298], device, args.batch_size)
        hardness = np.partition(probability.reshape(len(negative), -1), -50, axis=1)[:, -50:].mean(1)
        hard_negative = negative[np.argsort(hardness)[-max(1, len(negative) // 4):]]
        assert set(hard_negative).issubset(set(train)) and set(hard_negative).isdisjoint(set(val))
        del mining_model, probability
        torch.set_rng_state(torch_rng)
        (args.out / 'hard_negatives.json').write_text(json.dumps(data['names'][hard_negative].tolist()), encoding='utf-8')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=device, data_sha256=data_sha256, train_images=len(train), train_positive=len(positive), validation_images=len(val),
                  validation_crops=len(vn), validation_positive=int(vm.any((1, 2)).sum()),
                  note='Spatial AIHub proxy. Chosen on validation, not an independent test or Public score.')
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config), flush=True)

    def evaluate(name, tta=False):
        nonlocal best
        p = predict(model, vi, device, args.batch_size, tta)
        thresholds = (.05, .1, .15, .2, .3, .5, .7) if args.refine else (.1, .25, .4, .5, .6, .75)
        rows = [dict(name=name, threshold=t, tta=tta, **score_masks(p > t, vm)) for t in thresholds]
        metrics.extend(rows)
        winner = max(rows, key=lambda r: r['combined'])
        if winner['combined'] > best:
            best = winner['combined']
            winner['probability_file'] = 'best_prob.npy'
            np.save(args.out / 'best_prob.npy', p)
            torch.save({'state_dict': model.state_dict()}, args.out / 'best.pt')
            (args.out / 'best.json').write_text(json.dumps(winner, indent=2), encoding='utf-8')
        (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(winner), flush=True)
        return winner

    if not args.resume:
        evaluate('initial_checkpoint', tta=args.refine and args.tta)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        # Warm up the decoder; retain pretrained encoder BN but adapt decoder BN to real imagery.
        model.encoder.eval()
        model.encoder.requires_grad_(args.refine or epoch > 2)
        negative_count = args.samples - args.samples // 2
        mined_count = negative_count // 2 if args.refine else 0
        ids = np.r_[rng.choice(positive, args.samples // 2), rng.choice(negative, negative_count - mined_count),
                    rng.choice(hard_negative, mined_count)]
        rng.shuffle(ids)
        losses = []
        factor = .5 * (1 + math.cos(math.pi * (epoch - 1) / max(args.epochs, 1)))
        for group, lr in zip(optimizer.param_groups, (5e-6, 3e-5) if args.refine else (2e-5, 3e-4)):
            group['lr'] = lr * (.1 + .9 * factor)
        for offset in range(0, len(ids), args.batch_size):
            examples = [augment(data['images'][i], data['masks'][i], rng) for i in ids[offset:offset + args.batch_size]]
            x = torch.from_numpy(np.stack([e[0] for e in examples])).permute(0, 3, 1, 2).to(device)
            y = torch.from_numpy(np.stack([e[1] for e in examples]).astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                logits = model(x)
                p, target = logits.float().softmax(1)[:, 1], y.float()
                dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
                loss = refinement_loss(logits, y) if args.refine else ce(logits, y) + dice
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
                        epoch=epoch, data_sha256=data_sha256, refine=args.refine, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None, metrics=metrics), args.out / 'last.tmp')
        (args.out / 'last.tmp').replace(args.out / 'last.pt')
        print(f'epoch {epoch}: loss={np.mean(losses):.4f}, seconds={time.time() - started:.1f}', flush=True)
    if args.tta:
        model.load_state_dict(torch.load(args.out / 'best.pt', map_location=device, weights_only=True)['state_dict'])
        evaluate('best_flip_tta', tta=True)
    print('SELECTED', (args.out / 'best.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
