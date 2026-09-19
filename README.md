# Bhutanese Sign Language video collector

For static-image and video model training with MobileNetV2 + LSTM or transfer from `BSLModel.h5`, see [TRAINING.md](TRAINING.md).

A Python desktop tool for recording three IP camera streams. It has front/left/right previews, a word label, a three-second countdown, timed recording, and automatic take folders. No participant fields or camera URLs are stored in the dataset. MediaPipe is not required: retain original videos and extract landmarks afterward.

## Windows setup

Install Python 3.11 or later with Tcl/Tk support (included with the normal Windows installer). In this folder run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe collector.py
```

## Phones

1. Install an IP camera streaming app on each phone, such as IP Webcam for Android.
2. Connect the laptop and phones to the same Wi-Fi network. The network must allow connections between devices; some guest networks block these. Internet access is not necessary for local streams.
3. Place the phones at the front, left, and right. Keep hands, face, and upper body in view. Use consistent resolution and frame rate; start with 720p at 30 FPS if supported.
4. Start each app's camera server. Find its **video stream URL**, not just its settings page. For IP Webcam for Android this is commonly `http://PHONE_IP:8080/video`; verify the address and port in your app. Other apps may provide RTSP addresses instead.
5. Enter each full stream URL in the matching angle box and click **Connect / Reconnect**. Wait for all three live previews.
6. Enter a word, clip duration, output FPS, and output directory. Press **Record**. Perform one word after the countdown. Recording stops and saves automatically. Press Record again for the next take, or change the word.

Keep the streaming apps active and the phones powered throughout collection. If a stream disconnects, reconnect and record a new take. Failed takes are retained and marked incomplete instead of silently discarded. Stop/Cancel ends a take early or cancels the countdown. There is no automatic deletion or overwrite of existing takes.

## Output

```text
dataset/hello/take_001/
  front.mp4
  left.mp4
  right.mp4
  front_timestamps.csv
  left_timestamps.csv
  right_timestamps.csv
  metadata.json
```

Each CSV row maps a zero-based video frame index to its arrival time at the laptop relative to the take start. Metadata records the word, take, frame counts, output FPS, duration, and completion status. Only takes with status `complete` finished their requested recording window with all feeds live; still review clips before training.

## Timing and limitations

The three independent workers keep streams open before recording, but Wi-Fi delays, phone buffering, and sequential writer activation mean **the views are not frame-synchronized**. Timestamps measure laptop arrival, not the actual camera exposure time. A shared visible cue can help estimate alignment; this does not replace hardware synchronization.

OpenCV writes fixed-frame-rate video using the Output FPS setting. It does not preserve variable network frame timing: if a phone delivers fewer frames per second, playback can be shorter or faster than real time. The CSV files preserve arrival intervals for later processing. Match output FPS to the actual delivered rate and check recorded duration before a large collection session.

This tool records video only. MediaPipe processing is intentionally separate to avoid slowing capture. Encoding happens on the laptop; lower the cameras' resolution if previews or recording lag. A hard crash or power loss may leave an MP4 unplayable. Disk write failures are not always reported by OpenCV, so check free space and review a short test take.

Capture and video encoding run in separate workers so disk writes do not hold up previews or network reads. Each camera has a bounded recording queue (up to 15 frames, reduced for large frames). If the queue fills, the take stops and is marked incomplete with recording errors in metadata. Reduce phone resolution/FPS and retake. Stop drains the queued frames before closing files, which may briefly pause the interface.

## Basic verification

```powershell
.\.venv\Scripts\python.exe -m py_compile collector.py
```

Then record a short take with the actual phones, play all three MP4 files, and verify that each view, label, and completion status is correct. Unplug one phone during another take and verify it is marked incomplete.
