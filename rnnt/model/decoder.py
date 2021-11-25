from os import name

from tensorflow.keras import Model
from config.config import model_config
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, RNN, LayerNormalization

dec_config = model_config["decoder"]

def decoder():

    batch_size = model_config["batch_size"]
    # vocab_size = model_config["vocab_size"]
    # embedding_size = pred_config["embedding_size"]
    num_layers = dec_config["num_layers"]
    lstm_hidden_units = dec_config["lstm_hidden_units"]
    lstm_projection = dec_config["lstm_projection"]

    
    
    inp = Input(shape=[None,28], batch_size=batch_size, dtype=tf.float32)
    rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

    output = inp
    for i in range(num_layers):
        output = RNN(rnn_cell(),return_sequences=True, name=f"LSTM_DEC_{i+1}")(output)

    return Model(inputs=inp,outputs=output,name="decoder")
