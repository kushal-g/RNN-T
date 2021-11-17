from itertools import islice, chain, repeat
from config.config import speech_config
import numpy as np

def _create_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def _add_padding(window):
    for _ in range(len(window),speech_config["input_shape"]):
        window = np.append(window,[[0] * speech_config["feature_bins"]],axis=0)
    return window

def create_slices(l,n):
    slices = _create_chunks(l, n)
    slices = np.array(list(slices),dtype=object)
    slices[-1] = _add_padding(slices[-1])

    return slices
