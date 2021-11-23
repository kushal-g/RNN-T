from json import encoder
from preprocess import logmelspectogram
from preprocess.slice import create_slices
from config.config import speech_config
from model.prediction_network import prediction_network
from model.encoder import encoder
import numpy as np

mel_spec = logmelspectogram.get("/home/kushal/Desktop/RNN-T/music.mp3")

#slices = create_slices(mel_spec,speech_config["input_shape"])

dataset = np.array(object=object)
np.append(dataset,mel_spec)
print(np.array([mel_spec]).shape)

model  = encoder()
model.compile(optimizer="adam",loss="mse")

print(model.summary())