# BhSL CNN + LSTM training

This pipeline classifies **one isolated sign per image or trimmed video**. It does not segment continuous signing or translate sentences. Both static and dynamic signs share one vocabulary and classifier.

## Setup

Use Python **3.11 or 3.12**, separately from the collector's Python 3.14 environment:

```powershell
py -3.12 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.txt
```

The CPU path works on Windows. For NVIDIA GPU training with modern TensorFlow, use Linux/WSL2 and the appropriate TensorFlow CUDA installation; native Windows TensorFlow 2.16 is CPU-only. Start with batch 4, 16 frames, 128 pixels; reduce batch or frames if memory is limited. ImageNet weights download on first use. `--weights none` is for pipeline tests, not pretrained training.

## Prepare data

Put media under class folders. Images and videos can coexist:

```text
dataset/
  hello/take_001/front.mp4
  hello/take_001/left.mp4
  hello/take_001/right.mp4
  hello/take_001/metadata.json
  one/session_001/image_001.jpg
  one/session_001/image_002.jpg
```

Use identical class folder names for images and videos of the same sign. Every media file is one labeled sample. Collector takes whose metadata status is not `complete` are excluded. Supported extensions: JPG, JPEG, PNG, BMP, WEBP, MP4, AVI, MOV, MKV, WEBM, subject to installed codecs.

```powershell
.\.venv-training\Scripts\python.exe train_signs.py manifest --data dataset --output manifest.csv
```

**Review `manifest.csv` before training.** Its columns are `path,label,group,split`. Paths may be absolute or relative to the manifest. Automatic groups keep all files in a take/session folder together; files directly inside a class folder get individual groups. If photos come from a burst or frames extracted from a video, give all related media the same group, including their source video.

For an evaluation on unseen signers, replace `group` with an anonymous signer ID consistently across every class, session, image, and camera for that person. The collector does not record signer identities, so the script cannot infer these. Without this change, evaluation measures held-out takes rather than unseen signers. Near-duplicates cannot be detected reliably by file hashes.

Leave all `split` fields blank for a deterministic approximate 70/15/15 group split. Or fill **every** row with `train`, `val`, or `test`. Every class must appear in each partition, and no group may cross partitions. Collect at least three independent groups per class, preferably many more; small or uneven collections may need manually assigned splits. Exact duplicate files across partitions cause an error. Generated split manifests are saved for reproducible comparisons.

Review labels, trim idle time, and keep the crop/framing consistent between collection and inference. The code resizes the full input to a square, so prepare comparable hand/upper-body crops yourself. Each video is decoded twice to uniformly sample over its actual frames without loading the full video into memory. Short videos repeat sampled frames; static images repeat a single image. Still images supply appearance information, not motion. Avoid flooding the dataset with many nearly identical photos, and ensure dynamic classes have genuine video examples.

## Train MobileNetV2 + LSTM (default)

```powershell
.\.venv-training\Scripts\python.exe train_signs.py train --manifest manifest.csv --output runs/mobilenet --epochs 20 --fine-epochs 15
```

Architecture: shared ImageNet MobileNetV2 (width multiplier 0.35) on each frame → global average pooling → 128-unit LSTM → dense/dropout → softmax over your labels. Input pixels are RGB 0–255; normalization to [-1,1] is inside the saved model. First train the new temporal/classification layers with the CNN frozen, then fine-tune the last 30 CNN layers at a lower learning rate. Batch normalization stays frozen in inference mode. Select the better phase using validation loss.

Training uses deterministic seeds, sequence-consistent brightness/contrast augmentation, inverse-frequency class weights, gradient clipping, early stopping, learning-rate reduction, best checkpoints, and non-finite loss detection. No horizontal flips or temporal reversal are applied. The full media set is decoded before fitting to catch unreadable files. A partially decodable video may still contain a usable prefix: review the original recordings as well.

## Transfer from your BSLModel.h5

```powershell
.\.venv-training\Scripts\python.exe train_signs.py train --manifest manifest.csv --output runs/legacy --backbone legacy --legacy-model "D:\BhSL\Sir's Model\BSLModel.h5" --size 64 --epochs 20 --fine-epochs 15 --unfreeze 4
```

The legacy path loads the original weights without its optimizer, extracts the last convolutional feature map, applies global average pooling, and adds a new LSTM/classifier. The old ten-digit output layer is discarded; your manifest defines the new class order. It preserves **BGR / 255**, following the supplied `predictDigits.py`. That script used a mirrored webcam hand crop, so match that orientation and crop in your legacy training and inference data. The original training preprocessing is not supplied, and the script alone cannot prove what preprocessing was used during training.

This transfers the supplied CNN into an LSTM model. It does not copy its weights into MobileNetV2, whose layer shapes differ. Run both approaches on the **same saved splits** to compare fairly, e.g. use `--manifest runs/mobilenet/splits.csv` for the legacy run. A digit CNN may transfer less effectively to full-body or moving signs; validation determines which model works better.

## Predict and inspect results

```powershell
.\.venv-training\Scripts\python.exe predict_signs.py --run runs/mobilenet --input "new_sign.mp4"
.\.venv-training\Scripts\python.exe predict_signs.py --run runs/legacy --input "new_static_sign.jpg"
```

Outputs include:

- `model.keras`: selected CNN + LSTM, with normalization embedded.
- `config.json`: labels in output order, dimensions, color order, arguments, framework version.
- `splits.csv`: exact samples and partitions.
- `head.keras`, `fine.keras`: best checkpoint from each completed phase.
- `head.csv`, `fine.csv`: epoch training/validation history.
- `test_metrics.json`: accuracy, per-class precision/recall/F1, macro F1, confusion matrix and class order.
- `test_predictions.npz`: probabilities and targets, ordered as test rows in `splits.csv`.
- `architecture.txt`: model summary.

Keep `model.keras` and `config.json` together. Use a fresh output directory for each run; existing runs are never overwritten. Checkpoints support recovery of weights but there is no automatic interrupted-run resume command. Predictions return top-five softmax scores, which are not calibrated confidence or an unknown-sign detector. Add a well-collected background/unknown class and validate rejection thresholds if deploying interactively.

Accuracy cannot be established without representative real training and held-out data. Use validation for architecture and hyperparameter decisions; reserve the test set for the final comparison. These commands train a model; no real-data trained model is bundled.

## Verification commands

```powershell
.\.venv-training\Scripts\python.exe -m unittest test_training -v
.\.venv-training\Scripts\python.exe smoke_training.py --legacy-model "D:\BhSL\Sir's Model\BSLModel.h5"
```

The smoke check uses tiny synthetic images/videos, runs both training phases, and checks exported image/video predictions. MobileNet weights are random for this offline integration test; no accuracy claim can be made from synthetic results.

## Technical references

- [MobileNetV2 inputs and pretrained model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- [Transfer learning and batch normalization](https://keras.io/guides/transfer_learning/)
- [TensorFlow platform installation](https://www.tensorflow.org/install/pip)
