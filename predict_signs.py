"""Predict one isolated sign from an image or trimmed video."""
import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from sign_data import load_clip


def predict(run, media, top_k=5):
    run = Path(run)
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    model = keras.models.load_model(run/'model.keras', compile=False)
    clip = load_clip(media, config['frames'], config['size'], config['color_order'] == 'BGR')
    probabilities = model.predict(clip[None], verbose=0)[0]
    return [{'label': config['labels'][int(i)], 'probability': float(probabilities[i])}
            for i in np.argsort(probabilities)[::-1][:top_k]]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--input', required=True)
    args = p.parse_args()
    print(json.dumps(predict(args.run, args.input), ensure_ascii=False, indent=2))
