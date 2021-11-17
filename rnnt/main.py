from preprocess import logmelspectogram
from preprocess.slice import create_slices
from config.config import speech_config
import numpy as np

mel_spec = logmelspectogram.get("/home/kushal/Desktop/RNN-T/music.mp3")

slices = create_slices(mel_spec,speech_config["input_shape"])

print(slices.shape)
print(slices[0].shape)
print(slices[-1].shape)