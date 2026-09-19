"""Media decoding, collector discovery, and leakage-safe manifest validation."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def discover(root, output):
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in IMAGES | VIDEOS:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            raise ValueError(f'Media must be inside a class folder: {path}')
        metadata = path.parent / 'metadata.json'
        if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8'))['status'] != 'complete':
            continue
        # All camera views of one take must stay in the same partition.
        group = str(path.parent.relative_to(root)) if len(rel.parts) > 2 else str(rel)
        rows.append(dict(path=str(path), label=rel.parts[0], group=group, split=''))
    if not rows:
        raise ValueError(f'No supported media found under {root}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'group', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_manifest(filename):
    filename = Path(filename).resolve()
    with filename.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest is empty')
    seen = set()
    for row in rows:
        if not all(row.get(k, '').strip() for k in ('path', 'label', 'group')):
            raise ValueError('Every row needs path, label, and group')
        p = Path(row['path'])
        p = (filename.parent / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_file() or p.suffix.lower() not in IMAGES | VIDEOS:
            raise ValueError(f'Missing or unsupported media: {p}')
        if str(p).casefold() in seen:
            raise ValueError(f'Duplicate path: {p}')
        seen.add(str(p).casefold())
        row['path'] = str(p)
        row['split'] = row.get('split', '').strip()
    return rows


def partition(rows, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    labels = sorted({r['label'] for r in rows})
    if len(labels) < 2:
        raise ValueError('At least two classes are required')
    if any(r['split'] for r in rows):
        if any(r['split'] not in {'train', 'val', 'test'} for r in rows):
            raise ValueError('Fill every split with train, val, or test, or leave every split blank')
        parts = {s: [r.copy() for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    else:
        groups = np.array([r['group'] for r in rows])
        # Search deterministic group splits with all classes represented in each set.
        parts = None
        for attempt in range(300):
            try:
                trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed+attempt).split(rows, groups=groups))
                tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=seed+attempt).split(trainval, groups=groups[trainval]))
            except ValueError:
                continue
            candidate = dict(train=[rows[i].copy() for i in trainval[tr]],
                             val=[rows[i].copy() for i in trainval[va]],
                             test=[rows[i].copy() for i in test])
            if all({r['label'] for r in p} == set(labels) for p in candidate.values()):
                parts = candidate
                break
        if parts is None:
            raise ValueError('Cannot create three group-disjoint splits containing every class. Collect more independent groups or specify splits manually.')
    owner, hashes = {}, {}
    for split, records in parts.items():
        if {r['label'] for r in records} != set(labels):
            raise ValueError(f'{split} must contain every class')
        for row in records:
            row['split'] = split
            group = row['group']
            if group in owner and owner[group] != split:
                raise ValueError(f'Group leakage: {group}')
            owner[group] = split
            with open(row['path'], 'rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f'Identical media appears across splits: {row["path"]}')
            hashes[digest] = split
    return parts, labels


def load_clip(path, frames, size, bgr=False):
    """Uniform sampling over actual decoded frames, bounded memory, no padded steps."""
    path = Path(path)
    def resize(frame):
        if not bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    if path.suffix.lower() in IMAGES:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot decode image: {path}')
        return np.repeat(resize(frame)[None], frames, axis=0).astype(np.float32)
    # Two passes avoid trusting inaccurate container frame counts or keeping a video in RAM.
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    if count == 0:
        raise ValueError(f'Cannot decode video: {path}')
    indices = np.linspace(0, count-1, frames).round().astype(int)
    selected = {}
    cap = cv2.VideoCapture(str(path))
    try:
        needed = set(indices.tolist())
        for i in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Video changed or decoding failed: {path}, frame {i}')
            if i in needed:
                selected[i] = resize(frame)
    finally:
        cap.release()
    return np.stack([selected[i] for i in indices]).astype(np.float32)
