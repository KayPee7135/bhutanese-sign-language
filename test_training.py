"""Run with .venv-training/Scripts/python.exe -m unittest test_training -v."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sign_data import discover, load_clip, partition, read_manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def picture(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((40, 40, 3), value, np.uint8)
        image[..., 0] = 255
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_color_order_and_static_repeat(self):
        p = self.picture('a.png', 0)
        rgb = load_clip(p, 4, 32)
        bgr = load_clip(p, 4, 32, bgr=True)
        self.assertEqual(rgb.shape, (4,32,32,3))
        np.testing.assert_array_equal(rgb[0,0,0], [0,0,255])
        np.testing.assert_array_equal(bgr[0,0,0], [255,0,0])
        np.testing.assert_array_equal(rgb[0], rgb[-1])

    def test_video_order_and_short_clip(self):
        path = self.root/'test.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (40,40))
        self.assertTrue(writer.isOpened())
        for v in [0, 80, 160]:
            writer.write(np.full((40,40,3), v, np.uint8))
        writer.release()
        x = load_clip(path, 5, 32)
        self.assertEqual(x.shape, (5,32,32,3))
        self.assertTrue(np.all(np.diff(x.mean(axis=(1,2,3))) >= 0))
        self.assertGreater(x[-1].mean(), 150)

    def test_discovery_skips_incomplete_and_groups_views(self):
        self.picture('hello/take_001/front.png', 1)
        self.picture('hello/take_001/left.png', 2)
        self.picture('hello/take_002/front.png', 3)
        (self.root/'hello/take_002/metadata.json').write_text(json.dumps({'status':'incomplete'}))
        manifest = self.root/'manifest.csv'
        self.assertEqual(discover(self.root, manifest), 2)
        rows = read_manifest(manifest)
        self.assertEqual(rows[0]['group'], rows[1]['group'])

    def test_group_split_and_leakage(self):
        rows = []
        for i in range(20):
            for label in ['a','b']:
                p = self.picture(f'{label}/{i}.png', i*2+(label=='b'))
                rows.append(dict(path=str(p), label=label, group=str(i), split=''))
        parts, labels = partition(rows)
        self.assertEqual(labels, ['a','b'])
        groups = [{r['group'] for r in part} for part in parts.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        explicit = [r for part in parts.values() for r in part]
        duplicate = dict(parts['train'][0], split='val')
        with self.assertRaisesRegex(ValueError, 'Group leakage'):
            partition(explicit+[duplicate])
        duplicate['group'] = 'different-group'
        with self.assertRaisesRegex(ValueError, 'Identical media'):
            partition(explicit+[duplicate])


if __name__ == '__main__':
    unittest.main()
