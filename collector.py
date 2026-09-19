import csv
import json
import queue
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

ANGLES = ("front", "left", "right")


class Camera:
    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.received = 0.0
        self.status = "Connecting…"
        self.stop = threading.Event()
        self.writer = None
        self.csv_file = None
        self.count = 0
        self.error = None
        self.record_queue = None
        self.encoder_thread = None
        self.encoder_done = threading.Event()
        self.record_error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = None
        try:
            # Explicit FFmpeg backend supports network read/open timeouts.
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
            ])
            if not cap.isOpened():
                raise RuntimeError("Could not open stream. Check the URL and Wi-Fi.")
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Stream disconnected or timed out.")
                now = time.monotonic()
                with self.lock:
                    self.frame = frame
                    self.received = now
                    self.status = "Live"
                    if self.record_queue is not None:
                        if (frame.shape[1], frame.shape[0]) != self.size:
                            raise RuntimeError("Camera resolution changed during recording.")
                        try:
                            self.record_queue.put_nowait((frame, now))
                        except queue.Full:
                            self.record_error = "Recording cannot keep up. Lower phone resolution/FPS."
                            self.record_queue = None
                            self.encoder_done.set()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = self.error
        finally:
            if cap is not None:
                cap.release()

    def begin(self, folder, angle, fps, started):
        with self.lock:
            if self.frame is None or self.error or time.monotonic() - self.received > 3:
                raise RuntimeError(f"{angle}: stream is not live.")
            self.size = (self.frame.shape[1], self.frame.shape[0])
            writer = cv2.VideoWriter(str(folder / f"{angle}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Cannot create {angle}.mp4")
            try:
                self.csv_file = (folder / f"{angle}_timestamps.csv").open("w", newline="")
                self.timestamps = csv.writer(self.csv_file)
                self.timestamps.writerow(["frame_index", "seconds_since_take_start"])
            except Exception:
                writer.release()
                raise
            self.count = 0
            self.started = started
            self.writer = writer
            self.record_error = None
            # Bound both latency and memory (at most ~32 MB of queued raw frames).
            capacity = max(1, min(15, (32 * 1024 * 1024) // self.frame.nbytes))
            self.record_queue = queue.Queue(maxsize=capacity)
            self.encoder_done.clear()
            self.encoder_thread = threading.Thread(
                target=self.encode, args=(self.record_queue,), daemon=True)
            self.encoder_thread.start()

    def encode(self, frames):
        try:
            while not self.encoder_done.is_set() or not frames.empty():
                try:
                    frame, received = frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                # Never hold the preview/capture lock during encoding or disk I/O.
                self.writer.write(frame)
                self.timestamps.writerow([self.count, f"{received - self.started:.6f}"])
                self.count += 1
        except Exception as exc:
            with self.lock:
                self.record_error = f"Video writer failed: {exc}"
                self.record_queue = None
                self.encoder_done.set()
        finally:
            self.writer.release()
            self.csv_file.close()

    def end_capture(self):
        with self.lock:
            self.record_queue = None
            self.encoder_done.set()

    def finish(self):
        self.end_capture()
        if self.encoder_thread is not None:
            self.encoder_thread.join()
            self.encoder_thread = None
        with self.lock:
            self.writer = None
            self.csv_file = None
            return self.count


class Collector:
    def __init__(self, root):
        self.root = root
        root.title("BhSL — Three-angle video collector")
        root.geometry("1140x700")
        self.cameras = {}
        self.recording = False
        self.countdown = None
        self.folder = None
        self.urls = {}
        self.previews = {}
        self.states = {}
        self.images = {}
        self.word = tk.StringVar()
        self.seconds = tk.StringVar(value="5")
        self.fps = tk.StringVar(value="30")
        self.destination = tk.StringVar(value=str(Path(__file__).resolve().parent / "dataset"))
        self.message = tk.StringVar(value="Enter the actual video stream URLs, then connect.")

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Bhutanese Sign Language • Video collection",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(0, 12))
        grid = ttk.Frame(outer)
        grid.pack(fill="both", expand=True)
        for col, angle in enumerate(ANGLES):
            grid.columnconfigure(col, weight=1)
            panel = ttk.LabelFrame(grid, text=angle.title(), padding=8)
            panel.grid(row=0, column=col, sticky="nsew", padx=4)
            self.urls[angle] = tk.StringVar()
            ttk.Entry(panel, textvariable=self.urls[angle], width=34).pack(fill="x")
            self.previews[angle] = ttk.Label(panel, text="No camera connected", anchor="center")
            self.previews[angle].pack(fill="both", expand=True, pady=12)
            self.states[angle] = ttk.Label(panel, text="Disconnected", wraplength=300)
            self.states[angle].pack(fill="x")
        grid.rowconfigure(0, weight=1)
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=12)
        self.connect_button = ttk.Button(controls, text="Connect / Reconnect", command=self.connect)
        self.connect_button.pack(side="left", padx=(0, 18))
        for label, variable, width in (("Word:", self.word, 22), ("Seconds:", self.seconds, 6),
                                        ("Output FPS:", self.fps, 6)):
            ttk.Label(controls, text=label).pack(side="left", padx=(8, 4))
            ttk.Entry(controls, textvariable=variable, width=width).pack(side="left")
        self.record_button = ttk.Button(controls, text="Record (3-second countdown)", command=self.prepare)
        self.record_button.pack(side="left", padx=12)
        ttk.Button(controls, text="Stop / Cancel", command=self.stop_recording).pack(side="left")
        output = ttk.Frame(outer)
        output.pack(fill="x")
        ttk.Label(output, text="Save to:").pack(side="left")
        ttk.Entry(output, textvariable=self.destination).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(output, text="Browse", command=self.browse).pack(side="left")
        ttk.Label(outer, textvariable=self.message, wraplength=1050,
                  font=("Segoe UI", 11)).pack(anchor="w", pady=12)
        ttk.Label(outer, text="Keep phones streaming. Wi-Fi feeds are not frame-synchronized. "
                  "Timestamps measure arrival at this laptop. No participant fields are collected.",
                  wraplength=1050).pack(anchor="w")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def browse(self):
        path = filedialog.askdirectory()
        if path:
            self.destination.set(path)

    def connect(self):
        urls = {a: v.get().strip() for a, v in self.urls.items()}
        if not all(u.startswith(("http://", "https://", "rtsp://")) for u in urls.values()):
            messagebox.showerror("Stream URLs", "Enter an HTTP, HTTPS, or RTSP video URL for each angle.")
            return
        if len(set(urls.values())) != 3:
            messagebox.showerror("Stream URLs", "Use three different camera URLs.")
            return
        for camera in self.cameras.values():
            camera.stop.set()
        self.cameras = {a: Camera(urls[a]) for a in ANGLES}
        self.message.set("Connecting… Wait until all three previews are live.")

    def ready(self):
        return len(self.cameras) == 3 and all(
            c.frame is not None and not c.error and time.monotonic() - c.received < 3
            for c in self.cameras.values())

    def prepare(self):
        try:
            word = self.word.get().strip()
            if not word:
                raise ValueError("Enter a word label.")
            self.duration = float(self.seconds.get())
            self.output_fps = float(self.fps.get())
            if not 0 < self.duration <= 600 or not 1 <= self.output_fps <= 120:
                raise ValueError("Use 0–600 seconds (above zero) and 1–120 FPS.")
            if not self.ready():
                raise ValueError("All three cameras must be live before recording.")
            # Preserve Unicode labels while preventing paths and Windows reserved names.
            slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", word).strip(" .")[:100]
            if not slug:
                raise ValueError("The word needs at least one valid filename character.")
            if slug.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
                slug = "word_" + slug
            self.label = word
            self.word_folder = Path(self.destination.get()).expanduser().resolve() / slug
            self.word_folder.mkdir(parents=True, exist_ok=True)
            self.countdown = time.monotonic() + 3
            self.record_button.state(["disabled"])
            self.connect_button.state(["disabled"])
        except Exception as exc:
            messagebox.showerror("Cannot record", str(exc))

    def begin(self):
        self.countdown = None
        try:
            take = 1
            while True:
                self.folder = self.word_folder / f"take_{take:03d}"
                try:
                    self.folder.mkdir()
                    break
                except FileExistsError:
                    take += 1
            self.started = time.monotonic()
            self.metadata = {
                "word": self.label, "take": take, "angles": list(ANGLES),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "requested_seconds": self.duration, "output_fps": self.output_fps,
                "status": "incomplete", "timestamp_basis": "laptop frame arrival, monotonic",
                "note": "Fixed-FPS video; arrival timestamps are not camera exposure times.",
            }
            self.save_metadata()
            for angle, camera in self.cameras.items():
                camera.begin(self.folder, angle, self.output_fps, self.started)
            self.recording = True
        except Exception as exc:
            for camera in self.cameras.values():
                camera.finish()
            self.record_button.state(["!disabled"])
            self.connect_button.state(["!disabled"])
            self.message.set(f"Recording failed; any partial take is retained: {exc}")

    def save_metadata(self):
        path = self.folder / "metadata.json"
        temporary = self.folder / "metadata.tmp"
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def stop_recording(self, reason="stopped_early"):
        self.countdown = None
        if self.recording:
            self.recording = False
            elapsed = round(time.monotonic() - self.started, 3)
            for camera in self.cameras.values():
                camera.end_capture()
            counts = {a: c.finish() for a, c in self.cameras.items()}
            errors = {a: c.record_error for a, c in self.cameras.items() if c.record_error}
            if errors:
                reason = "incomplete_recording_overload"
            self.metadata.update(status=reason if all(counts.values()) else "incomplete",
                                 elapsed_seconds=elapsed, frames=counts, recording_errors=errors)
            try:
                self.save_metadata()
                self.message.set(f"{self.metadata['status']}: {self.folder} — frames: {counts}"
                                 + (". Lower phone resolution/FPS and retake." if errors else ""))
            except Exception as exc:
                self.message.set(f"Videos closed, but metadata could not be saved: {exc}")
        else:
            self.message.set("Countdown cancelled / ready.")
        self.record_button.state(["!disabled"])
        self.connect_button.state(["!disabled"])

    def tick(self):
        for angle, camera in self.cameras.items():
            with camera.lock:
                frame = camera.frame
                status = camera.status
            self.states[angle].configure(text=status)
            if frame is not None:
                scale = min(340 / frame.shape[1], 300 / frame.shape[0], 1)
                small = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)),
                                           max(1, int(frame.shape[0] * scale))))
                preview = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                self.images[angle] = ImageTk.PhotoImage(preview)
                self.previews[angle].configure(image=self.images[angle], text="")
        if self.countdown is not None:
            remaining = self.countdown - time.monotonic()
            if remaining <= 0:
                self.begin()
            else:
                self.message.set(f"Get ready: {int(remaining) + 1}…")
        if self.recording:
            if any(c.record_error for c in self.cameras.values()):
                self.stop_recording("incomplete_recording_overload")
            elif not self.ready():
                self.stop_recording("incomplete_stream_failure")
            else:
                remaining = self.duration - (time.monotonic() - self.started)
                if remaining <= 0:
                    self.stop_recording("complete")
                else:
                    self.message.set(f"Recording {self.label} — {remaining:.1f} seconds remaining")
        self.root.after(80, self.tick)

    def close(self):
        self.stop_recording()
        for camera in self.cameras.values():
            camera.stop.set()
        self.root.destroy()


if __name__ == "__main__":
    Collector(tk.Tk()).root.mainloop()


