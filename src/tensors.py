"""Deterministic EMPS tensor ETL. No augmentation or learned normalization at ETL time."""
import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'Data/Database/microscopy.db'
OUT = ROOT / 'data/inputs'
SEED = 42
SIDE = 256
FOLDS = 5


def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def image_fingerprint(path):
    # Identical image pixels must never be divided across train/eval.
    with Image.open(path) as im:
        pixels = np.asarray(im.convert('L'))
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def rows_from_database():
    if not DB.is_file():
        raise FileNotFoundError(f'Missing database: {DB}. Run just first.')
    with sqlite3.connect(DB) as con:
        rows = con.execute('''SELECT image_id, image_path, label_path, split
                              FROM images WHERE source_dataset = 'EMPS'
                              ORDER BY image_id''').fetchall()
    if len(rows) != 465:
        raise ValueError(f'Expected 465 EMPS records; got {len(rows)}')
    return rows


def build_manifest(rows):
    """Group exact-duplicate image pixels and deterministic filename-family prefixes.

    The original EMPS split is retained only if all groups are disjoint across it.
    Otherwise assign entire groups deterministically to train/eval. Grouping by
    filename prefix is a conservative heuristic, NOT proof of acquisition identity.
    """
    fingerprints = {}
    prefixes = {}
    for image_id, img, label, original_split in rows:
        p = ROOT / img
        q = ROOT / label
        if not p.is_file() or not q.is_file():
            raise FileNotFoundError(f'Missing image or mask for {image_id}: {p}, {q}')
        stem = Path(img).stem
        # Whole filename when there is no separator; otherwise shared prefix.
        prefixes[image_id] = stem.rsplit('_', 1)[0] if '_' in stem and stem.rsplit('_', 1)[-1].isdigit() else stem
        fingerprints[image_id] = image_fingerprint(p)

    # Union-find merges both filename families and exact image duplicates.
    parent = {r[0]: r[0] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen = {}
    for image_id, *_ in rows:
        for key in ('prefix:' + prefixes[image_id], 'pixels:' + fingerprints[image_id]):
            if key in seen:
                union(image_id, seen[key])
            else:
                seen[key] = image_id
    groups = defaultdict(list)
    for image_id, *_ in rows:
        groups[find(image_id)].append(image_id)
    if len(groups) < FOLDS + 1:
        raise ValueError(f'Only {len(groups)} independent groups: insufficient for {FOLDS}-fold CV')

    original = {r[0]: r[3] for r in rows}
    original_safe = all(len({original[i] for i in members}) == 1 for members in groups.values())
    original_safe &= {original[i] for i in original} == {'train', 'test'}
    if original_safe:
        allocation = {g: original[members[0]] for g, members in groups.items()}
        split_method = 'EMPS supplied split, verified disjoint across derived groups'
    else:
        ordered = sorted(groups, key=lambda g: digest(f'{SEED}:split:{g}'))
        eval_count = max(1, round(len(ordered) * 0.2))
        eval_groups = set(ordered[:eval_count])
        allocation = {g: ('test' if g in eval_groups else 'train') for g in groups}
        split_method = 'seeded 80/20 group split; original split conflicts with groups'
    training_groups = sorted((g for g in groups if allocation[g] == 'train'),
                             key=lambda g: digest(f'{SEED}:fold:{g}'))
    if len(training_groups) < FOLDS:
        raise ValueError('Insufficient training groups for 5 folds')
    folds = {g: i % FOLDS for i, g in enumerate(training_groups)}
    result = []
    for image_id, img, label, original_split in rows:
        g = find(image_id)
        split = 'eval' if allocation[g] == 'test' else 'train'
        result.append({'image_id': image_id, 'image_path': img, 'label_path': label,
                       'group_id': g, 'split': split,
                       'fold': folds[g] if split == 'train' else -1,
                       'original_split': original_split,
                       'pixel_sha256': fingerprints[image_id]})
    return result, split_method


def load_pair(item):
    with Image.open(ROOT / item['image_path']) as im:
        im = im.convert('L').resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        image = np.asarray(im, dtype=np.uint8)
    with Image.open(ROOT / item['label_path']) as im:
        # Convert instance IDs to binary BEFORE resizing; nearest preserves labels.
        mask = Image.fromarray((np.asarray(im) > 0).astype(np.uint8) * 255)
        mask = mask.resize((SIDE, SIDE), Image.Resampling.NEAREST)
        target = (np.asarray(mask) > 0).astype(np.uint8)
    return image[None], target[None]


def export():
    rows = rows_from_database()
    manifest, method = build_manifest(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'eval'):
        subset = [r for r in manifest if r['split'] == split]
        if not subset:
            raise ValueError(f'Empty {split} set')
        images, masks = [], []
        for item in subset:
            a, b = load_pair(item)
            images.append(a)
            masks.append(b)
        data = {'images': np.stack(images), 'masks': np.stack(masks),
                'folds': np.asarray([r['fold'] for r in subset], dtype=np.int32)}
        tmp = OUT / f'{split}.safetensors.part'
        save_file(data, str(tmp), metadata={'dataset': 'EMPS', 'seed': str(SEED),
                                            'side': str(SIDE), 'format': 'uint8 grayscale and binary masks'})
        tmp.replace(OUT / f'{split}.safetensors')
        print(f'{split}: {len(subset)} samples, shape={data["images"].shape}')
    report = {'seed': SEED, 'side': SIDE, 'folds': FOLDS,
              'split_method': method,
              'grouping_note': 'Filename prefix and identical pixels; acquisition-level relationships not guaranteed',
              'records': manifest}
    tmp = OUT / 'manifest.json.part'
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    tmp.replace(OUT / 'manifest.json')
    print(f'Wrote {OUT} using {method}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    export()
