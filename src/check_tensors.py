"""Fail-fast, independently auditable tensor and leakage checks."""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data/inputs'


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    records = manifest['records']
    assert len(records) == 465, 'Wrong number of EMPS samples'
    ids = [r['image_id'] for r in records]
    assert len(ids) == len(set(ids)), 'Duplicate sample IDs'
    assert ids == sorted(ids), 'Samples not deterministically ordered'
    assert manifest['seed'] == 42 and manifest['folds'] == 5
    train = [r for r in records if r['split'] == 'train']
    eval_ = [r for r in records if r['split'] == 'eval']
    assert train and eval_, 'Empty train or eval split'
    assert not ({r['group_id'] for r in train} & {r['group_id'] for r in eval_}), 'GROUP LEAKAGE'
    assert not ({r['pixel_sha256'] for r in train} & {r['pixel_sha256'] for r in eval_}), 'DUPLICATE IMAGE LEAKAGE'
    assert all(r['fold'] == -1 for r in eval_)
    assert set(r['fold'] for r in train) == set(range(5)), 'Missing CV folds'
    fold_by_group = {}
    for r in train:
        g = r['group_id']
        if g in fold_by_group:
            assert fold_by_group[g] == r['fold'], 'Cross-validation group leakage'
        fold_by_group[g] = r['fold']
    with sqlite3.connect(ROOT / 'Data/Database/microscopy.db') as con:
        expected = {row[0] for row in con.execute("SELECT image_id FROM images WHERE source_dataset='EMPS' ORDER BY image_id")}
    assert set(ids) == expected, 'Tensor manifest does not match SQLite records'
    for name, subset in [('train', train), ('eval', eval_)]:
        with safe_open(str(OUT / f'{name}.safetensors'), framework='np') as f:
            assert set(f.keys()) == {'images', 'masks', 'folds'}
            x, y, folds = f.get_tensor('images'), f.get_tensor('masks'), f.get_tensor('folds')
        assert x.shape == y.shape == (len(subset), 1, 256, 256), 'Bad tensor shapes'
        assert x.dtype == y.dtype == np.uint8, 'Unexpected tensor dtypes'
        assert set(np.unique(y)).issubset({0, 1}), 'Masks are not binary'
        assert folds.tolist() == [r['fold'] for r in subset], 'Fold ordering mismatch'
        print(f'PASS {name}: {len(subset)} image/mask pairs, shape={x.shape}')
    print('PASS: no derived-group overlap, no exact-image overlap, deterministic order, five group-aware folds')
    print('Note: source acquisition groups are not documented by EMPS; filename-family grouping is a heuristic.')


if __name__ == '__main__':
    main()
