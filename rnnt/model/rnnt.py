from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from rnnt.model.encoder import encoder
from rnnt.model.decoder import decoder
import tensorflow as tf

def rnnt():
    enc = encoder()
    dec = decoder()
    joint_inp = (
        tf.expand_dims(dec.output, axis=2) +                 # [B, T, V] => [B, T, 1, V]
        tf.expand_dims(enc.output, axis=1)              # [B, U, V] => [B, 1, U, V]
    )
    merged = Dense(28, activation="tanh")(joint_inp)
    dec.layers[0](merged)

    return Model(inputs=[enc.input,dec.input],outputs=[merged], name="RNN-Transducer")