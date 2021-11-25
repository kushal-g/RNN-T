from json import encoder

from tensorflow import keras
from preprocess import logmelspectogram
from preprocess.slice import create_slices
from config.config import speech_config
import numpy as np

from model.rnnt import rnnt

mel_spec = logmelspectogram.get("/home/kushal/Desktop/RNN-T/music.mp3")

#slices = create_slices(mel_spec,speech_config["input_shape"])

dataset = np.array(object=object)
np.append(dataset,mel_spec)
print(np.array([mel_spec]).shape)

model  = rnnt()
model.compile(optimizer="adam",loss="mse")

print(model.summary())
keras.utils.plot_model(model, "my_first_model.png")