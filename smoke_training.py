"""Exercise both training phases, export, and inference on synthetic media.

This is an integration check, not a sign-recognition accuracy benchmark.
"""
import argparse
import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from train_signs import parser, train
from predict_signs import predict


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--legacy-model', help='Also check the actual supplied H5')
    options = cli.parse_args()
    with tempfile.TemporaryDirectory(prefix='bhsl_smoke_') as tmp:
        root = Path(tmp)
        rows = []
        rng = np.random.default_rng(42)
        for split in ['train', 'val', 'test']:
            for label in ['one', 'hello']:
                for kind in ['image', 'video']:
                    p = root/f'{split}_{label}_{kind}'
                    frames = rng.integers(0, 256, (3,64,64,3), dtype=np.uint8)
                    if kind == 'image':
                        p = p.with_suffix('.png')
                        assert cv2.imwrite(str(p), frames[0])
                    else:
                        p = p.with_suffix('.avi')
                        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,64))
                        assert writer.isOpened()
                        for frame in frames:
                            writer.write(frame)
                        writer.release()
                    rows.append(dict(path=str(p), label=label, group=f'{split}_{label}', split=split))
        manifest = root/'manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'label', 'group', 'split'])
            writer.writeheader()
            writer.writerows(rows)
        backbones = ['mobilenet'] + (['legacy'] if options.legacy_model else [])
        for backbone in backbones:
            args = parser().parse_args(['train', '--manifest', str(manifest), '--output', str(root/backbone),
                '--backbone', backbone, '--weights', 'none', '--size', '64', '--frames', '2',
                '--batch', '2', '--workers', '1', '--epochs', '1', '--fine-epochs', '1', '--unfreeze', '4']
                + (['--legacy-model', options.legacy_model] if backbone == 'legacy' else []))
            train(args)
            for row in rows[-2:]:
                result = predict(root/backbone, row['path'])
                assert len(result) == 2 and abs(sum(x['probability'] for x in result)-1) < 1e-5
            keras.backend.clear_session()
            print(f'PASS: {backbone} training, fine-tuning, save/reload, image and video inference')


if __name__ == '__main__':
    main()
