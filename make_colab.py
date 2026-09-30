"""Create a self-contained Colab training notebook and private data bundle."""
import argparse
import ast
import copy
import json
import shutil
import zipfile
from pathlib import Path

from train_baseline import BASE, HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'experiment_v2')
    parser.add_argument('--output', type=Path, default=Path.home() / 'Desktop/해안쓰레기/GPU학습')
    parser.add_argument('--init', type=Path, default=BASE)
    parser.add_argument('--refine', action='store_true', help='Build 12-epoch refinement notebooks for Colab and Kaggle')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = HERE / 'colab_bundle'
    bundle.mkdir(exist_ok=True)
    for name in ('prepare_coastal.py', 'train_spatial.py', 'train_baseline.py', 'improve_model.py', 'package_improved.py', 'test_pipeline.py'):
        ast.parse((HERE / name).read_text(encoding='utf-8'))
        shutil.copy2(HERE / name, bundle / name)
    (bundle / 'data').mkdir(exist_ok=True)
    for name in ('data.npz', 'data_report.json'):
        shutil.copy2(args.data / name, bundle / 'data' / name)
    source = BASE.parents[2]
    (bundle / 'template').mkdir(exist_ok=True)
    for name in ('requirements.txt', 'LICENSE', 'NOTICE'):
        shutil.copy2(source / name, bundle / 'template' / name)
    template = json.loads((source / 'predict.ipynb').read_text(encoding='utf-8'))
    template['cells'] = [c for c in template['cells'] if not (
        c['cell_type'] == 'code' and any(t in ''.join(c['source']) for t in ('%aifactory', '%load_ext', '%pip')))
        and not ''.join(c['source']).startswith('## 제출')]
    for i, cell in enumerate(template['cells']):
        cell['id'] = f'cell-{i}'
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
            ast.parse(''.join(cell['source']))
    (bundle / 'template/predict.ipynb').write_text(json.dumps(template, ensure_ascii=False, indent=1), encoding='utf-8')
    shutil.copy2(args.init, bundle / 'provided_original.pt')
    archive = args.output / 'coastal_training_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for file in sorted(bundle.rglob('*')):
            if file.is_file():
                z.write(file, file.relative_to(bundle))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    cells = []

    def cell(kind, text):
        result = dict(cell_type=kind, id=f'cell-{len(cells)}', metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
            if not text.startswith('%'):
                ast.parse(text)
        cells.append(result)

    cell('markdown', '''# 해안쓰레기 GPU 학습
1. **런타임 → 런타임 유형 변경 → T4 GPU → 저장**을 선택합니다. 무료 GPU 배정은 가용량에 따라 달라집니다.
2. 아래 셀을 위에서부터 실행합니다. 업로드 창에서 함께 제공한 `coastal_training_bundle.zip` 하나를 선택합니다.
3. 학습이 끝나면 검증된 제출 ZIP을 다운로드합니다. 이 노트북은 대회 서버에 자동 제출하지 않습니다.

데이터 출처·개수는 세 번째 셀에서 표시합니다. 점수는 AIHub 지역 분리 검증값이며 대회 Public 점수가 아닙니다.
학습 중 런타임이 끊기면 로컬 파일이 사라질 수 있습니다. 마지막 셀로 체크포인트를 내려받을 수 있습니다.
''')
    cell('code', '%pip -q install segmentation-models-pytorch==0.5.0 opencv-python-headless\n')
    cell('code', '''import torch
assert torch.cuda.is_available(), '상단 런타임 > 런타임 유형 변경 > T4 GPU를 선택하세요.'
print('사용 GPU:', torch.cuda.get_device_name(0))
from google.colab import files
uploaded = files.upload()  # coastal_training_bundle.zip 선택
assert len(uploaded) == 1, '학습 데이터 ZIP 하나를 선택하세요.'
from pathlib import Path
import zipfile, os, json, io
work = Path('/content/coastal').resolve()
work.mkdir(exist_ok=True)
# Read the selected upload, including a renamed duplicate, rather than a stale file on disk.
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    assert {'data/data.npz', 'data/data_report.json', 'train_spatial.py'} <= set(z.namelist()), '학습용 ZIP을 선택하세요.'
    for name in z.namelist():
        target = (work / name).resolve()
        assert target.is_relative_to(work), 'ZIP 내부 경로 오류'
    z.extractall(work)
del uploaded
os.chdir(work)
report = json.loads(Path('data/data_report.json').read_text())
print('학습 데이터:', report['prepared'])
print('검증 지역 수:', len(report['holdout_groups']))
import subprocess, sys
subprocess.run([sys.executable, 'test_pipeline.py'], check=True)
''')
    cell('code', f'''import subprocess, sys, hashlib
with Path('data/data.npz').open('rb') as file:
    data_id = hashlib.file_digest(file, 'sha256').hexdigest()[:10]
with Path('provided_original.pt').open('rb') as file:
    model_id = hashlib.file_digest(file, 'sha256').hexdigest()[:10]
EXPERIMENT = 'experiment_{'refine' if args.refine else 'gpu'}_' + data_id + '_' + model_id
command = [sys.executable, '-u', 'train_spatial.py', '--data', 'data/data.npz',
           '--init', 'provided_original.pt', '--out', EXPERIMENT,
           '--device', 'cuda', '--epochs', '{12 if args.refine else 30}', '--samples', '{768 if args.refine else 512}', '--batch-size', '16', '--tta'] + {(['--refine'] if args.refine else [])!r}
if (Path(EXPERIMENT) / 'last.pt').exists():
    command.append('--resume')
subprocess.run(command, check=True)
''')
    cell('code', '''print((Path(EXPERIMENT) / 'best.json').read_text())
subprocess.run([sys.executable, 'package_improved.py', '--experiment', EXPERIMENT,
                '--template', 'template', '--destination', 'submission_gpu'], check=True)
files.download('submission_gpu.zip')
''')
    cell('markdown', '선택: 학습 체크포인트·검증 기록 백업. 브라우저에서 다운로드를 허용하세요.\n')
    cell('code', '''import shutil
shutil.make_archive('/content/coastal_training_results', 'zip', '/content/coastal', EXPERIMENT)
files.download('/content/coastal_training_results.zip')
''')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                              accelerator='GPU', colab=dict(name='해안쓰레기_GPU학습.ipynb')),
                    nbformat=4, nbformat_minor=5)
    (args.output / '해안쓰레기_GPU학습.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    if args.refine:
        # Keep Kaggle from unpacking and indexing the training archive during upload.
        shutil.copyfile(archive, archive.with_suffix('.bin'))
        kaggle = copy.deepcopy(notebook)
        kaggle['metadata'].pop('colab', None)
        kaggle['cells'][0]['source'] = ['# 해안쓰레기 무료 Kaggle GPU 보완 학습\n',
            '1. `coastal_training_bundle.bin`을 본인 계정의 **Private** 데이터로 올리고 Add Input으로 연결합니다. 노트북에서 압축을 풉니다.\n',
            '2. Notebook 설정에서 GPU와 Internet을 켠 뒤 Run All을 실행합니다.\n',
            '3. 완료 후 Save Version에서 출력 저장을 켜고 저장합니다. 저장된 버전의 Output 탭에서 제출 ZIP과 학습 기록을 다운로드합니다.\n',
            'Public 0.3913 모델에서 12회 보완 학습합니다. 학습용 오탐과 이미지별 Dice를 사용하며 대회에 자동 제출하지 않습니다.\n']
        for item in kaggle['cells']:
            source_code = ''.join(item['source'])
            if 'uploaded = files.upload()' in source_code:
                source_code = '''import torch, shutil, zipfile, json, subprocess, sys
from pathlib import Path
assert torch.cuda.is_available(), 'Notebook 설정에서 GPU를 켜세요.'
print('사용 GPU:', torch.cuda.get_device_name(0))
work = Path('/kaggle/working/coastal').resolve()
work.mkdir(parents=True, exist_ok=True)
roots = [p.parent.parent for p in Path('/kaggle/input').rglob('data.npz')
         if p.parent.name == 'data' and (p.parent.parent / 'train_spatial.py').is_file()]
if roots:
    assert len(roots) == 1, '이번 학습 데이터 하나만 연결하세요.'
    shutil.copytree(roots[0], work, dirs_exist_ok=True)
else:
    archives = [p for name in ('coastal_training_bundle.zip', 'coastal_training_bundle.bin')
                for p in Path('/kaggle/input').rglob(name)]
    assert len(archives) == 1, 'Add Input으로 이번 학습 ZIP을 연결하세요.'
    with zipfile.ZipFile(archives[0]) as z:
        assert {'data/data.npz', 'data/data_report.json', 'train_spatial.py'} <= set(z.namelist())
        assert all((work / name).resolve().is_relative_to(work) for name in z.namelist()), 'ZIP 내부 경로 오류'
        z.extractall(work)
import os
os.chdir(work)
print('학습 데이터:', json.loads(Path('data/data_report.json').read_text())['prepared'])
subprocess.run([sys.executable, 'test_pipeline.py'], check=True)
'''
            if "files.download('submission_gpu.zip')" in source_code:
                source_code = source_code.replace("files.download('submission_gpu.zip')", "print('출력을 포함하여 Save Version으로 저장한 뒤, Output → coastal → submission_gpu.zip을 다운로드하세요.')")
            if 'coastal_training_results' in source_code:
                source_code = '''import shutil
shutil.make_archive('/kaggle/working/coastal_training_results', 'zip', str(work), EXPERIMENT)
print('학습 기록: 저장된 버전의 Output → coastal_training_results.zip에서 다운로드하세요.')
'''
            if item['cell_type'] == 'code' and not source_code.startswith('%'):
                ast.parse(source_code)
            item['source'] = source_code.splitlines(keepends=True)
        (args.output / '해안쓰레기_Kaggle학습.ipynb').write_text(json.dumps(kaggle, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Created {archive} ({archive.stat().st_size / 1e6:.1f} MB) and GPU notebook', flush=True)


if __name__ == '__main__':
    main()
