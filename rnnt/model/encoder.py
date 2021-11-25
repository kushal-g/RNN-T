from os import name
from config.config import model_config, speech_config
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, RNN, LayerNormalization

pred_config = model_config["encoder"]

def encoder():

    batch_size = model_config["batch_size"]
    num_layers = pred_config["num_layers"]
    lstm_hidden_units = pred_config["lstm_hidden_units"]
    lstm_projection = pred_config["lstm_projection"]
    frames_at_a_time = model_config["frames_at_a_time"]
    feature_bins = speech_config["feature_bins"]

    model = Sequential()
    model._init_set_name("encoder")
    
    model.add(Input(shape=[None,feature_bins], batch_size=batch_size, dtype=tf.float32))
    rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

    for i in range(num_layers):
        model.add(RNN(rnn_cell(),return_sequences=True, name=f"LSTM_ENC_{i+1}"))

    return model
