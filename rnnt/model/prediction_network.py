from config.config import model_config
import tensorflow as tf
from tf.keras import Input
from tf.keras.layers import Embedding, RNN, LayerNormalization

pred_config = model_config["prediction_network"]

def prediction_network():

    batch_size = model_config["batch_size"]
    # vocab_size = model_config["vocab_size"]
    # embedding_size = pred_config["embedding_size"]
    num_layers = pred_config["num_layers"]
    lstm_hidden_units = pred_config["lstm_hidden_units"]
    lstm_projection = pred_config["lstm_projection"]

    inputs = Input(shape=[None], batch_size=batch_size, dtype=tf.float32)
    #embed = Embedding(vocab_size, embedding_size)(inputs)

    rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

    outputs = inputs

    for _ in range(num_layers):

        outputs = RNN(rnn_cell(),return_sequences=True)(outputs)
        outputs = LayerNormalization(dtype=tf.float32)(outputs)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs], name='prediction_network')
