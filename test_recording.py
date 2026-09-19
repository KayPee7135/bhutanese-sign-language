import importlib.util
import queue
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

with patch.dict(sys.modules, {name: MagicMock() for name in
                             ('cv2', 'PIL', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox')}):
    spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('collector.py'))
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)


class RecordingTests(unittest.TestCase):
    def camera(self):
        c = collector.Camera.__new__(collector.Camera)
        c.lock = threading.Lock()
        c.encoder_done = threading.Event()
        c.record_queue = queue.Queue(maxsize=3)
        c.writer = MagicMock()
        c.csv_file = MagicMock()
        c.timestamps = MagicMock()
        c.count = 0
        c.started = 100.0
        c.record_error = None
        c.encoder_thread = None
        return c

    def test_slow_writer_does_not_lock_preview_and_stop_drains(self):
        c = self.camera()
        entered, release = threading.Event(), threading.Event()
        def slow_write(frame):
            entered.set()
            release.wait(2)
        writer, csv_file = c.writer, c.csv_file
        writer.write.side_effect = slow_write
        c.record_queue.put(('first', 100.1))
        c.record_queue.put(('second', 100.2))
        c.encoder_thread = threading.Thread(target=c.encode, args=(c.record_queue,))
        c.encoder_thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(c.lock.acquire(timeout=0.2))
            c.lock.release()
            c.end_capture()
        finally:
            release.set()
        self.assertEqual(c.finish(), 2)
        self.assertEqual(writer.write.call_count, 2)
        writer.release.assert_called_once()
        csv_file.close.assert_called_once()
        self.assertEqual(c.timestamps.writerow.call_args_list[1].args[0], [1, '0.200000'])

    def test_encoding_failure_reported_and_resources_closed(self):
        c = self.camera()
        writer, csv_file = c.writer, c.csv_file
        writer.write.side_effect = RuntimeError('disk failure')
        c.record_queue.put(('frame', 100.1))
        c.encode(c.record_queue)
        self.assertIn('disk failure', c.record_error)
        self.assertIsNone(c.record_queue)
        writer.release.assert_called_once()
        csv_file.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
