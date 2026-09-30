"""Package the selected checkpoint and execute its notebook on validation inputs."""
import csv
import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from improve_model import OUT, self_check
from train_baseline import BASE, HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=OUT)
    parser.add_argument('--ensemble', type=Path, help='Average with one other experiment; requires --threshold')
    parser.add_argument('--threshold', type=float, help='Override the selected probability threshold')
    parser.add_argument('--template', type=Path, default=BASE.parents[2])
    parser.add_argument('--destination', type=Path, default=HERE / 'submission_improved_20260927')
    args = parser.parse_args()
    if args.ensemble and args.threshold is None:
        parser.error('--ensemble requires an explicitly validated --threshold')
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error('--threshold must be between 0 and 1')
    experiment = args.experiment.resolve()
    self_check()
    torch.set_num_threads(4)
    selected = json.loads((experiment / 'best.json').read_text())
    experiments = [experiment] + ([args.ensemble.resolve()] if args.ensemble else [])
    selections = [json.loads((path / 'best.json').read_text()) for path in experiments]
    threshold = selected['threshold'] if args.threshold is None else args.threshold
    data = np.load(experiment / 'data.npz')
    val = np.flatnonzero(data['splits'] == 'val')
    probabilities = []
    for path, choice in zip(experiments, selections):
        with np.load(path / 'data.npz') as other:
            other_val = np.flatnonzero(other['splits'] == 'val')
            for key in ('names', 'images', 'masks'):
                assert np.array_equal(data[key][val], other[key][other_val]), f'Validation mismatch: {path}, {key}'
        probability = np.load(path / choice.get('probability_file', choice['name'] + '_prob.npy'))
        assert probability.shape == data['masks'][val].shape
        assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
        probabilities.append(probability)
    expected_prob = sum(probabilities) / len(probabilities)
    source = args.template.resolve()
    destination = args.destination.resolve()
    (destination / 'assets/model').mkdir(parents=True, exist_ok=True)
    checkpoints = ['assets/model/unet_r18_debris_lite.pt']
    if args.ensemble:
        checkpoints.append('assets/model/ensemble.pt')
    for path, checkpoint in zip(experiments, checkpoints):
        shutil.copy2(path / 'best.pt', destination / checkpoint)
    for name in ('requirements.txt', 'LICENSE', 'NOTICE'):
        shutil.copy2(source / name, destination / name)
    notebook = json.loads((source / 'predict.ipynb').read_text(encoding='utf-8'))
    cells = []
    for cell in notebook['cells']:
        content = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and ('%aifactory' in content or '%pip' in content or '%load_ext' in content):
            continue
        if content.startswith('## 제출'):
            continue
        if 'def predict_batch' in content:
            content = content.replace('(im / 255.0 - MEAN) / STD', '(im.astype(np.float32) / 255.0 - MEAN) / STD')
            start, end = content.index('def load_model('), content.index('def read_image(')
            loader = 'def load_model(device: str):\n    models = torch.nn.ModuleList()\n'
            loader += f'    for checkpoint in {checkpoints!r}:\n'
            loader += '        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)\n'
            loader += '        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])\n'
            loader += '        models.append(model)\n    return models.to(device).eval()\n\n\n'
            content = content[:start] + loader + content[end:]
            old = 'prob = torch.softmax(model(x), dim=1)\n    return (prob[:, 1] > prob[:, 0]).cpu().numpy()'
            new = f'probability = 0\n    for member, tta in zip(model, {[bool(c["tta"]) for c in selections]!r}):\n'
            new += '        p = torch.softmax(member(x), dim=1)[:, 1]\n'
            new += '        if tta:\n            for dims in ((2,), (3,), (2, 3)):\n                p += torch.softmax(member(x.flip(dims)), dim=1)[:, 1].flip(tuple(d - 1 for d in dims))\n            p /= 4\n'
            new += f'        probability += p\n    return (probability / len(model) > {threshold}).cpu().numpy()'
            assert old in content
            content = content.replace(old, new)
        cell['source'] = content.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        cells.append(cell)
    notebook['cells'] = cells
    (destination / 'predict.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    smoke = destination / 'smoke'
    (smoke / 'images').mkdir(parents=True, exist_ok=True)
    positive = data['masks'][val].any(axis=(1, 2))
    positions = np.r_[np.flatnonzero(positive)[:2], np.flatnonzero(~positive)[:2]]
    ids = val[positions]
    assert len(ids), 'No validation inputs available'
    with (smoke / 'patches.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id'])
        for index in ids:
            name = f'check{index}'
            writer.writerow([name])
            Image.fromarray(data['images'][index]).save(smoke / 'images' / (name + '.png'))
    os.environ['AIF_INPUT_DIR'] = str(smoke)
    os.environ['AIF_PREDICTION_PATH'] = str(smoke / 'prediction.csv')
    os.chdir(destination)
    scope = {}
    for cell in cells:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'predict.ipynb', 'exec'), scope)
    rows = list(csv.DictReader((smoke / 'prediction.csv').open()))
    assert [row['id'] for row in rows] == [f'check{i}' for i in ids]
    for row, probability in zip(rows, expected_prob[positions]):
        expected = probability > threshold
        if expected.sum() < 50:
            expected[:] = False
        decoded = np.zeros(65536, dtype=bool)
        values = list(map(int, row['rle'].split()))
        assert len(values) % 2 == 0
        for start, length in zip(values[::2], values[1::2]):
            assert start >= 0 and length > 0 and start + length <= 65536
            decoded[start:start + length] = True
        assert np.array_equal(decoded.reshape(256, 256), expected)
    for mask in (np.zeros((256, 256), bool), np.ones((256, 256), bool), np.eye(256, dtype=bool)):
        encoded = list(map(int, scope['rle_encode'](mask).split()))
        decoded = np.zeros(65536, bool)
        for start, length in zip(encoded[::2], encoded[1::2]):
            decoded[start:start + length] = True
        assert np.array_equal(mask.reshape(-1), decoded)
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in destination.rglob('*'):
            if path.is_file() and 'smoke' not in path.relative_to(destination).parts:
                output.write(path, path.relative_to(destination))
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        assert 'predict.ipynb' in output.namelist()
    print('Verified notebook, predictions, RLE and ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
