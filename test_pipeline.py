"""Run with python test_pipeline.py; no downloads or training required."""
import argparse
import io
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from improve_model import score_masks, self_check as score_check
from prepare_coastal import close, self_check as geometry_check
from train_spatial import augment, evaluation_data


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

parser = argparse.ArgumentParser()
parser.add_argument('--colab-notebook', type=Path)
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
