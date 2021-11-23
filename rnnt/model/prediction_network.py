from os import name
from config.config import model_config
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, RNN, LayerNormalization

pred_config = model_config["prediction_network"]

def prediction_network():

    batch_size = model_config["batch_size"]
    # vocab_size = model_config["vocab_size"]
    # embedding_size = pred_config["embedding_size"]
    num_layers = pred_config["num_layers"]
    lstm_hidden_units = pred_config["lstm_hidden_units"]
    lstm_projection = pred_config["lstm_projection"]

    model = Sequential()
    model._init_set_name("prediction_network")
    
    model.add(Input(shape=[3,80], batch_size=batch_size, dtype=tf.float32))
    rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

    for i in range(num_layers):
        model.add(RNN(rnn_cell(),return_sequences=True, name=f"LSTM_{i+1}"))

    return model
