from json import encoder
from numpy.random import randint
import tensorflow as tf
from tensorflow import keras
from preprocess import logmelspectogram
from preprocess.slice import create_slices
from config.config import speech_config
import numpy as np

from model.rnnt import RNNTransducer

# mel_spec = logmelspectogram.get("music.mp3")

# #slices = create_slices(mel_spec,speech_config["input_shape"])

# dataset = np.array(object=object)
# np.append(dataset,mel_spec)
# print(np.array([mel_spec]).shape)

# model  = rnnt()
# model.compile(optimizer="adam",loss="mse")

# print(model.summary())
from config.config import model_config, speech_config

model = RNNTransducer(
  B=model_config["batch_size"],
  T=model_config["timesteps"],
  U=model_config["label_length"],
  V=model_config["vocab_size"],
)
model.compile(optimizer="adam",loss="mse")

#Prediction
print(
  model(
    np.random.rand(model_config["batch_size"],model_config["timesteps"],80),
  )
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
x_train = np.random.rand(32,model_config["timesteps"],80)
y_train = np.array([randint(0,high=27,size=3) for _ in range(32)])

print(x_train.shape,y_train.shape)
model.fit(x_train, y_train, epochs=2, batch_size=32)

#keras.utils.plot_model(model, "rnnt.png")

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)