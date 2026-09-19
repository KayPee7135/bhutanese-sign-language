"""Train a frame CNN + LSTM. See TRAINING.md for data and commands."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from sign_data import discover, load_clip, partition, read_manifest
from legacy_model import load_legacy_model


def make_encoder(backbone, size, legacy_model=None, weights='imagenet'):
    if backbone == 'mobilenet':
        base = keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=.35,
                                             include_top=False, weights=weights, pooling='avg')
        image = keras.Input((size, size, 3))
        x = keras.layers.Rescaling(1/127.5, offset=-1)(image)
    else:
        if not legacy_model:
            raise ValueError('--legacy-model is required for legacy transfer')
        source = load_legacy_model(legacy_model)
        if tuple(source.input_shape[1:]) != (size, size, 3):
            raise ValueError(f'Legacy input is {source.input_shape}; use its required --size')
        convs = [l for l in source.layers if isinstance(l, keras.layers.Conv2D)]
        if not convs:
            raise ValueError('Legacy model must have top-level Conv2D layers')
        # Transfer spatial filters, discard the old digit classifier and large dense layers.
        base = keras.Model(source.inputs, keras.layers.GlobalAveragePooling2D()(convs[-1].output), name='legacy_features')
        image = keras.Input((size, size, 3))
        x = keras.layers.Rescaling(1/255)(image)
    base.trainable = False
    encoder = keras.Model(image, base(x, training=False), name='frame_encoder')
    return encoder, base


def build_model(backbone, size, frames, classes, legacy_model=None, weights='imagenet'):
    encoder, base = make_encoder(backbone, size, legacy_model, weights)
    clip = keras.Input((frames, size, size, 3), name='clip')
    x = keras.layers.TimeDistributed(encoder, name='frame_features')(clip)
    x = keras.layers.LSTM(128, dropout=.25, name='temporal_features')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(.4)(x)
    output = keras.layers.Dense(classes, activation='softmax', dtype='float32', name='sign')(x)
    return keras.Model(clip, output), base


def dataset(rows, labels, args, training=False):
    indices = {label: i for i, label in enumerate(labels)}
    paths = [r['path'] for r in rows]
    targets = [indices[r['label']] for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training:
        ds = ds.shuffle(len(rows), seed=args.seed, reshuffle_each_iteration=True)
    def decode(path, label):
        def read(value):
            return load_clip(value.numpy().decode('utf-8'), args.frames, args.size, args.backbone == 'legacy')
        clip = tf.py_function(read, [path], tf.float32)
        clip.set_shape((args.frames, args.size, args.size, 3))
        if training:
            # One brightness/contrast transform for the whole sequence. Never flip handedness.
            contrast = tf.random.uniform([], .9, 1.1)
            brightness = tf.random.uniform([], -12., 12.)
            clip = tf.clip_by_value((clip-127.5)*contrast+127.5+brightness, 0., 255.)
        return clip, label
    ds = ds.map(decode, num_parallel_calls=args.workers, deterministic=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = args.workers
    return ds.batch(args.batch).with_options(options).prefetch(1)


def train(args):
    if min(args.frames, args.batch, args.workers, args.epochs) < 1 or args.size < 32 or args.fine_epochs < 0 or args.unfreeze < 1:
        raise ValueError('Invalid dimensions, batch size, workers, or epoch counts')
    if args.lr <= 0 or args.fine_lr <= 0:
        raise ValueError('Learning rates must be positive')
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    parts, labels = partition(read_manifest(args.manifest), args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['labels'] = labels
    config['tensorflow'] = tf.__version__
    config['color_order'] = 'BGR' if args.backbone == 'legacy' else 'RGB'
    (out/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out/'splits.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'group', 'split'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r for records in parts.values() for r in records)
    print('Checking all media before training...')
    for records in parts.values():
        for row in records:
            load_clip(row['path'], args.frames, args.size, args.backbone == 'legacy')
    ds = {s: dataset(records, labels, args, s == 'train') for s, records in parts.items()}
    model, base = build_model(args.backbone, args.size, args.frames, len(labels), args.legacy_model,
                              None if args.weights == 'none' else 'imagenet')
    with (out/'architecture.txt').open('w', encoding='utf-8') as f:
        model.summary(print_fn=lambda line, **kwargs: f.write(line+'\n'))
    counts = np.bincount([labels.index(r['label']) for r in parts['train']], minlength=len(labels))
    class_weight = {i: float(counts.sum()/(len(labels)*n)) for i, n in enumerate(counts)}
    def fit_phase(name, epochs, lr):
        model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callbacks = [keras.callbacks.ModelCheckpoint(str(out/f'{name}.keras'), monitor='val_loss', save_best_only=True),
                     keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=.5, min_lr=1e-7),
                     keras.callbacks.CSVLogger(str(out/f'{name}.csv')),
                     keras.callbacks.TerminateOnNaN()]
        history = model.fit(ds['train'], validation_data=ds['val'], epochs=epochs,
                            class_weight=class_weight, callbacks=callbacks)
        if not all(np.isfinite(v) for v in history.history['loss'] + history.history['val_loss']):
            raise RuntimeError('Non-finite loss; training aborted. Inspect data and learning rate.')
        return min(history.history['val_loss'])
    scores = {'head': fit_phase('head', args.epochs, args.lr)}
    if args.fine_epochs:
        model.load_weights(out/'head.keras')
        base.trainable = True
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= len(base.layers)-args.unfreeze and not isinstance(layer, keras.layers.BatchNormalization)
        scores['fine'] = fit_phase('fine', args.fine_epochs, args.fine_lr)
    best = min(scores, key=scores.get)
    model = keras.models.load_model(out/f'{best}.keras', compile=False)
    model.save(out/'model.keras')
    # Select by validation only; test data is evaluated exactly once after selection.
    probabilities = model.predict(ds['test'])
    actual = np.array([labels.index(r['label']) for r in parts['test']])
    predicted = probabilities.argmax(axis=1)
    report = classification_report(actual, predicted, labels=list(range(len(labels))),
                                    target_names=labels, output_dict=True, zero_division=0)
    report['confusion_matrix'] = confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist()
    report['class_order'] = labels
    report['selected_phase'] = best
    report['validation_losses'] = scores
    (out/'test_metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(out/'test_predictions.npz', probabilities=probabilities, targets=actual)
    print(f'Saved {out / "model.keras"}; test accuracy={report["accuracy"]:.4f}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('manifest')
    d.add_argument('--data', required=True)
    d.add_argument('--output', required=True)
    t = sub.add_parser('train')
    t.add_argument('--manifest', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--backbone', choices=['mobilenet', 'legacy'], default='mobilenet')
    t.add_argument('--legacy-model')
    t.add_argument('--weights', choices=['imagenet', 'none'], default='imagenet', help='none is for offline smoke tests only')
    for name, default in [('size',128), ('frames',16), ('batch',4), ('workers',2), ('epochs',20), ('fine-epochs',15), ('unfreeze',30), ('seed',42)]:
        t.add_argument('--'+name, type=int, default=default)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--fine-lr', type=float, default=1e-5)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'manifest':
        print(f'Wrote {discover(args.data, args.output)} records. Review groups before training.')
    else:
        train(args)
