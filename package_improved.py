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
    parser.add_argument('--template', type=Path, default=BASE.parents[2])
    parser.add_argument('--destination', type=Path, default=HERE / 'submission_improved_20260927')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    self_check()
    torch.set_num_threads(4)
    selected = json.loads((experiment / 'best.json').read_text())
    source = args.template.resolve()
    destination = args.destination.resolve()
    (destination / 'assets/model').mkdir(parents=True, exist_ok=True)
    shutil.copy2(experiment / 'best.pt', destination / 'assets/model/unet_r18_debris_lite.pt')
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
            old = 'return (prob[:, 1] > prob[:, 0]).cpu().numpy()'
            new = 'p = prob[:, 1]\n'
            if selected['tta']:
                new += '    for dims in ((2,), (3,), (2, 3)):\n        p += torch.softmax(model(x.flip(dims)), dim=1)[:, 1].flip(tuple(d - 1 for d in dims))\n    p /= 4\n'
            new += f"    return (p > {selected['threshold']}).cpu().numpy()"
            assert old in content
            content = content.replace(old, new)
        cell['source'] = content.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        cells.append(cell)
    notebook['cells'] = cells
    (destination / 'predict.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    smoke = experiment / 'smoke'
    (smoke / 'images').mkdir(parents=True, exist_ok=True)
    data = np.load(experiment / 'data.npz')
    val = np.flatnonzero(data['splits'] == 'val')
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
    expected_prob = np.load(experiment / selected.get('probability_file', selected['name'] + '_prob.npy'))[positions]
    rows = list(csv.DictReader((smoke / 'prediction.csv').open()))
    assert [row['id'] for row in rows] == [f'check{i}' for i in ids]
    for row, probability in zip(rows, expected_prob):
        expected = probability > selected['threshold']
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
            if path.is_file():
                output.write(path, path.relative_to(destination))
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        assert 'predict.ipynb' in output.namelist()
    print('Verified notebook, predictions, RLE and ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
