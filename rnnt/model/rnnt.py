from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Concatenate
from rnnt.model.prediction_network import prediction_network
from rnnt.model.encoder import encoder

def rnnt():
    pred_net = prediction_network()
    enc = encoder()

    merged = Concatenate()([pred_net.output,enc.output])
    merged = Dense(28, activation="tanh")(merged)

    return Model(inputs=[pred_net.input,enc.input],outputs=[merged], name="RNN-Transducer")