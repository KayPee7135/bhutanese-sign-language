"""Read the supplied Keras 2 Sequential H5 with modern Keras, without modifying it."""
import copy
import json

import h5py
from tensorflow import keras


def load_legacy_model(path):
    with h5py.File(path, 'r') as f:
        config = json.loads(f.attrs['model_config'])
    if config['class_name'] != 'Sequential':
        return keras.models.load_model(path, compile=False)
    specs = config['config']['layers']
    # Keras 2.2 stored batch_input_shape on Conv2D, and BN axis as a list.
    shape = specs[0]['config'].get('batch_input_shape')
    if shape is None:
        return keras.models.load_model(path, compile=False)
    allowed = {'Conv2D', 'BatchNormalization', 'MaxPooling2D', 'Dropout', 'Flatten', 'Dense'}
    model = keras.Sequential(name=config['config'].get('name', 'legacy'))
    model.add(keras.Input(shape=tuple(shape[1:])))
    for original in specs:
        spec = copy.deepcopy(original)
        if spec['class_name'] not in allowed:
            raise ValueError(f'Unsupported legacy layer: {spec["class_name"]}')
        spec['config'].pop('batch_input_shape', None)
        if spec['class_name'] == 'BatchNormalization':
            axis = spec['config']['axis']
            if isinstance(axis, list):
                if len(axis) != 1:
                    raise ValueError('Only single-axis legacy BatchNormalization is supported')
                spec['config']['axis'] = axis[0]
        model.add(keras.layers.deserialize(spec))
    # Strict topology loading: mismatched/missing weights raise rather than being skipped.
    model.load_weights(path)
    return model
