"""Build georeferenced debris masks, an extraction manifest, and a spatial holdout."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def debris_features(doc):
    return [f for f in doc['features']
            if str(f['properties'].get('ANN_NM', '')).startswith('해안쓰레기')]


def points(coordinates):
    if coordinates and isinstance(coordinates[0], (float, int)):
        yield coordinates[:2]
    else:
        for part in coordinates:
            yield from points(part)


def bounds(doc):
    xy = np.array([p for f in doc['features'] for p in points(f['geometry']['coordinates'])])
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"Invalid geometry: {doc['name']}")
    return np.r_[xy.min(0), xy.max(0)].tolist()


def close(a, b, gap=0):
    return min(a[2], b[2]) - max(a[0], b[0]) >= -gap and min(a[3], b[3]) - max(a[1], b[1]) >= -gap


def read_labels(path):
    if path.is_dir():
        docs = [json.loads(p.read_bytes()) for p in sorted(path.rglob('*.json'))]
    else:
        with zipfile.ZipFile(path) as z:
            docs = [json.loads(z.read(n)) for n in sorted(z.namelist()) if n.lower().endswith('.json')]
    for doc in docs:
        if doc.get('crs', {}).get('properties', {}).get('name') != 'urn:ogc:def:crs:EPSG::32652':
            raise ValueError(f"Unsupported CRS: {doc.get('name')}")
    return docs


def rasterize(doc, size, tie, scale):
    """Use class names, never ambiguous ANN_CD values. Include polygon holes."""
    mask = np.zeros((size[1], size[0]), np.uint8)
    for feature in debris_features(doc):
        geometry = feature['geometry']
        polygons = geometry['coordinates']
        if geometry['type'] == 'Polygon':
            polygons = [polygons]
        elif geometry['type'] != 'MultiPolygon':
            raise ValueError(f"Unsupported geometry: {geometry['type']}")
        for polygon in polygons:
            part = np.zeros_like(mask)
            for index, ring in enumerate(polygon):
                xy = np.asarray(ring, dtype=float)[:, :2]
                xy[:, 0] = (xy[:, 0] - tie[3]) / scale[0] + tie[0] - .5
                xy[:, 1] = (tie[4] - xy[:, 1]) / scale[1] + tie[1] - .5
                if len(xy) < 3 or not np.isfinite(xy).all():
                    raise ValueError(f"Invalid polygon: {doc['name']}")
                cv2.fillPoly(part, [np.rint(xy * 256).astype(np.int32)], int(index == 0), shift=8)
            mask |= part
    return mask


def self_check():
    def feature(name, ring):
        return {'properties': {'ANN_CD': 80, 'ANN_NM': name},
                'geometry': {'type': 'Polygon', 'coordinates': ring}}
    doc = {'name': 'test', 'features': [
        feature('해안쓰레기_플라스틱', [[[2, 8], [8, 8], [8, 2], [2, 2]],
                                  [[4, 6], [6, 6], [6, 4], [4, 4]]]),
        feature('산림_침엽수림', [[[0, 10], [1, 10], [1, 9], [0, 9]]])]}
    m = rasterize(doc, (10, 10), (0, 0, 0, 0, 10, 0), (1, 1, 0))
    assert m[3, 3] and not m[5, 5] and not m[0, 0]
    assert not close([0, 0, 10, 10], [41, 0, 50, 10], 30)
    assert close([0, 0, 10, 10], [9, 9, 20, 20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, action='append', required=True, help='JSON ZIP or extracted directory; repeat for multiple sources')
    parser.add_argument('--images', type=Path, action='append', help='Image ZIP or directory, in the same order as --labels')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'experiment_v2')
    parser.add_argument('--holdout', type=Path, help='Existing data_report.json: preserve its validation region when adding data')
    parser.add_argument('--negative-ratio', type=float, default=2., help='Maximum training negatives per positive image')
    args = parser.parse_args()
    self_check()
    if args.images and len(args.images) != len(args.labels):
        parser.error('Pass one --images for each --labels')
    args.out.mkdir(parents=True, exist_ok=True)
    documents = {}
    for source, path in enumerate(args.labels):
        for doc in read_labels(path):
            stem = doc['name']
            if stem in documents:
                raise ValueError(f'Duplicate image ID across sources: {stem}')
            documents[stem] = (source, doc)
    positives = {stem for stem, (_, d) in documents.items() if debris_features(d)}
    if not positives:
        raise ValueError('No debris labels found')
    positive_groups = {s.split('_')[0] for s in positives}
    group_bounds = {}
    for stem in sorted(positives):
        group = stem.split('_')[0]
        b = bounds(documents[stem][1])
        if group in group_bounds:
            a = group_bounds[group]
            b = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        group_bounds[group] = b
    # ponytail: quadratic scan over coastal tiles; use a spatial index if the manifest reaches thousands of tiles.
    components = []
    for group, box in group_bounds.items():
        matches = [c for c in components if any(close(box, group_bounds[g], 25) for g in c)]
        merged = {group}
        for c in matches:
            merged |= c
            components.remove(c)
        components.append(merged)
    counts = Counter(s.split('_')[0] for s in positives)
    if args.holdout:
        previous = json.loads(args.holdout.read_text(encoding='utf-8'))
        holdout_boxes = previous['holdout_bounds']
        holdout_groups = set(previous['holdout_groups'])
    else:
        if len(components) < 2:
            raise ValueError('Only one geographic site; add data or supply an independent --holdout')
        holdout_groups = min(components, key=lambda c: (abs(sum(counts[g] for g in c) / len(positives) - .25), sorted(c)))
        holdout_boxes = [group_bounds[g] for g in sorted(holdout_groups)]
    rows = []
    for stem, (source, doc) in sorted(documents.items()):
        group = stem.split('_')[0]
        box = bounds(doc)
        # Same coastal tile across dates gives useful negatives without selecting by model predictions.
        if group not in positive_groups and not any(close(box, b, 0) for b in group_bounds.values()):
            continue
        split = 'val' if group in holdout_groups else 'train'
        if split == 'train' and any(close(box, b, 30) for b in holdout_boxes):
            split = 'excluded_buffer'
        rows.append(dict(id=stem, source=source, split=split, positive=int(stem in positives),
                         group=group, image=stem + '.tif', json=stem + '.json'))
    training_positive_count = sum(r['split'] == 'train' and r['positive'] for r in rows)
    negatives = sorted((r for r in rows if r['split'] == 'train' and not r['positive']),
                       key=lambda r: hashlib.sha256(r['id'].encode()).digest())
    selected_negatives = {r['id'] for r in negatives[:max(0, int(training_positive_count * args.negative_ratio))]}
    rows = [r for r in rows if r['split'] != 'train' or r['positive'] or r['id'] in selected_negatives]
    with (args.out / 'extraction_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(labels=[str(p.resolve()) for p in args.labels], images=[str(p.resolve()) for p in args.images or []],
                  total_json=len(documents), positive_images=len(positives), positive_tiles=len(positive_groups),
                  geographic_components=[dict(groups=sorted(c), positive_images=sum(counts[g] for g in c)) for c in components],
                  holdout_groups=sorted(holdout_groups), holdout_bounds=holdout_boxes, buffer_m=30,
                  class_counts=dict(Counter(f['properties']['ANN_NM'] for _, d in documents.values() for f in debris_features(d))),
                  manifest_splits=dict(Counter(r['split'] for r in rows)),
                  note='AIHub proxy validation, not competition score. Four overlapping fixed crops per validation image are correlated.')
    report_path = args.out / 'data_report.json'
    if not args.images:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=True), flush=True)
        return
    images, masks, splits, names, boxes, areas = [], [], [], [], [], []
    for source, path in enumerate(args.images):
        archive = zipfile.ZipFile(path) if path.is_file() else None
        try:
            files = archive.namelist() if archive else [str(p) for p in path.rglob('*') if p.is_file()]
            index = {}
            for name in files:
                if Path(name).suffix.lower() not in ('.tif', '.tiff'):
                    continue
                key = Path(name).stem
                if key in index:
                    raise ValueError(f'Duplicate image stem: {key}')
                index[key] = name
            chosen = [r for r in rows if r['source'] == source and r['split'] != 'excluded_buffer']
            missing = [r['id'] for r in chosen if r['id'] not in index]
            if missing:
                raise FileNotFoundError(f'{len(missing)} selected images missing; first IDs: {missing[:10]}. See extraction_manifest.csv')
            for number, row in enumerate(chosen):
                stem = row['id']
                payload = io.BytesIO(archive.read(index[stem])) if archive else index[stem]
                with Image.open(payload) as image:
                    if image.size != (1024, 1024):
                        raise ValueError(f'Expected 1024x1024: {stem} {image.size}')
                    tie, scale = image.tag_v2[33922], image.tag_v2[33550]
                    if not np.allclose(scale[:2], .1):
                        raise ValueError(f'Expected 0.1m imagery: {stem} {scale}')
                    rgb = np.asarray(image.convert('RGB'))[:1023, :1023]
                    mask = rasterize(documents[stem][1], image.size, tie, scale)[:1023, :1023]
                    box = [tie[3], tie[4] - 102.4, tie[3] + 102.4, tie[4]]
                    # Enforce the embargo using actual raster footprints, not only annotation bounds.
                    if row['split'] == 'train' and any(close(box, b, 30) for b in holdout_boxes):
                        continue
                    if image.mode == 'RGBA' and (np.asarray(image.getchannel('A'))[:1023, :1023] == 0).any():
                        report.setdefault('excluded_nodata', []).append(stem)
                        continue
                small = cv2.resize(rgb, (341, 341), interpolation=cv2.INTER_AREA)
                target = mask.reshape(341, 3, 341, 3).mean(axis=(1, 3)) >= .5
                images.append(small)
                masks.append(target)
                splits.append(row['split'])
                names.append(stem)
                boxes.append(box)
                areas.append(int(target.sum()))
                if number % 100 == 0:
                    print(f'source {source}: {number}/{len(chosen)}', flush=True)
        finally:
            if archive:
                archive.close()
    for a, split in zip(boxes, splits):
        if split == 'train':
            assert not any(close(a, b, 0) for b, s in zip(boxes, splits) if s == 'val'), 'Spatial leakage'
    if not images or not {'train', 'val'}.issubset(splits):
        raise ValueError('Need non-empty train and val sets')
    np.savez_compressed(args.out / 'data.npz', images=np.asarray(images), masks=np.asarray(masks),
                        splits=splits, names=names, bounds=boxes)
    report['prepared'] = {s: dict(images=splits.count(s), positives=sum(a > 0 for a, t in zip(areas, splits) if t == s),
                                positives_ge50=sum(a >= 50 for a, t in zip(areas, splits) if t == s)) for s in ('train', 'val')}
    report['positive_area_quantiles'] = np.quantile([a for a in areas if a], [0, .5, 1]).tolist()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
