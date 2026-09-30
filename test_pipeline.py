"""Run with python test_pipeline.py; no downloads or training required."""
import argparse
import io
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from improve_model import score_masks, self_check as score_check
from prepare_coastal import close, self_check as geometry_check
from train_spatial import augment, evaluation_data, refinement_loss


geometry_check()  # Class-code collision, georeferencing and polygon holes.
score_check()     # Official 1px tolerance and discarded sub-50px predictions.
mask = np.zeros((16, 16), bool)
mask[:5, :10] = True
empty = np.zeros_like(mask)
assert score_masks([mask, empty], [mask, empty])['combined'] == 1  # Exactly 50 is positive.
assert not close([0, 0, 10, 10], [41, 0, 50, 10], 30)

# Validation crop locations must be independent of where the ground truth sits.
data = dict(images=np.arange(341 * 341 * 3, dtype=np.int64).astype(np.uint8).reshape(1, 341, 341, 3),
            masks=np.zeros((1, 341, 341), bool), splits=np.array(['val']), names=np.array(['tile_date']))
data['masks'][0, 0, 0] = True
images, masks, names = evaluation_data(data)
data['masks'][:] = False
data['masks'][0, -1, -1] = True
other_images, other_masks, other_names = evaluation_data(data)
assert np.array_equal(images, other_images) and names == other_names
assert masks.sum() == other_masks.sum() == 1  # Do not erase small positive targets.

# Foreground sampling can retain a target at an extreme edge without centering every patch.
rng = np.random.default_rng(42)
retained = 0
for _ in range(32):
    image, target = augment(data['images'][0], data['masks'][0], rng)
    assert image.shape == (256, 256, 3) and image.dtype == np.float32
    assert np.isfinite(image).all() and target.shape == (256, 256)
    assert set(np.unique(target)).issubset({0, 1})
    retained += bool(target.any())
assert retained >= 16
print('Geometry, scoring, unbiased validation crops and training augmentation checks passed.')

# Refinement must learn both empty and positive patches without NaNs or favoring inverted masks.
for truth in (torch.zeros((2, 16, 16), dtype=torch.long),
              torch.from_numpy(np.stack([mask, empty]).astype(np.int64))):
    correct = torch.stack([1 - truth.float(), truth.float()], dim=1) * 8 - 4
    wrong = (-correct).requires_grad_()
    loss = refinement_loss(wrong, truth)
    assert torch.isfinite(loss) and loss > refinement_loss(correct, truth)
    loss.backward()
    assert torch.isfinite(wrong.grad).all() and wrong.grad.abs().sum() > 0
print('Refinement loss handles empty masks and produces useful finite gradients.')

parser = argparse.ArgumentParser()
parser.add_argument('--colab-notebook', type=Path)
parser.add_argument('--kaggle-notebook', type=Path)
args = parser.parse_args()
if args.colab_notebook:
    notebook = json.loads(args.colab_notebook.read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells'] if 'uploaded = files.upload()' in ''.join(c['source']))
    code = source[source.index('uploaded = files.upload()'):source.index('os.chdir(work)')]
    code = code.replace("Path('/content/coastal').resolve()", 'test_work')
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='colab-upload-check-', dir=root) as temporary:
        work = Path(temporary).resolve()
        assert work.is_relative_to(root)
        (work / 'coastal_training_bundle.zip').write_bytes(b'stale file must not be read')
        for malicious in (False, True):
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, 'w') as archive:
                for name in ('data/data.npz', 'data/data_report.json', 'train_spatial.py'):
                    archive.writestr(name, b'new upload')
                if malicious:
                    archive.writestr('../outside.txt', b'must not extract')
            files = SimpleNamespace(upload=lambda: {'coastal_training_bundle (1).zip': payload.getvalue()})
            try:
                exec(compile(code, 'colab-upload-cell', 'exec'), {'files': files, 'test_work': work})
            except AssertionError:
                assert malicious
            else:
                assert not malicious
                assert (work / 'data/data.npz').read_bytes() == b'new upload'
    print('Colab uses the new upload despite duplicate filenames and rejects ZIP path traversal.')
if args.kaggle_notebook:
    import shutil
    notebook = json.loads(args.kaggle_notebook.read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells'] if "Path('/kaggle/input')" in ''.join(c['source']))
    code = source[source.index('work = Path'):source.index('\nimport os')]
    code = code.replace("Path('/kaggle/working/coastal').resolve()", 'test_work').replace("Path('/kaggle/input')", 'test_input')
    with tempfile.TemporaryDirectory(prefix='kaggle-input-check-', dir=Path(__file__).parent) as temporary:
        for mode in ('folder', 'zip', 'bin', 'unsafe_zip'):
            incoming = Path(temporary) / mode
            incoming.mkdir()
            work = incoming / 'output'
            files_to_write = {'data/data.npz': b'new data', 'data/data_report.json': b'{}', 'train_spatial.py': b'pass'}
            if mode == 'folder':
                for name, payload in files_to_write.items():
                    file = incoming / 'dataset' / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(payload)
            else:
                name = 'coastal_training_bundle.bin' if mode == 'bin' else 'coastal_training_bundle.zip'
                with zipfile.ZipFile(incoming / name, 'w') as archive:
                    for name, payload in files_to_write.items():
                        archive.writestr(name, payload)
                    if mode == 'unsafe_zip':
                        archive.writestr('../outside.txt', b'forbidden')
            try:
                exec(compile(code, 'kaggle-input-cell', 'exec'), dict(Path=Path, shutil=shutil, zipfile=zipfile, test_work=work, test_input=incoming))
            except AssertionError:
                assert mode == 'unsafe_zip'
            else:
                assert mode != 'unsafe_zip' and (work / 'data/data.npz').read_bytes() == b'new data'
    print('Kaggle handles extracted datasets and ZIPs and rejects unsafe archive paths.')
