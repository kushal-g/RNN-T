from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
import tensorflow as tf
from tensorflow import keras
from config.config import model_config, speech_config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, RNN, LayerNormalization
from tensorflow.keras import Input

enc_config = model_config["encoder"]
dec_config = model_config["decoder"]

class RNNTransducer(keras.Model):
    def __init__(self,B,T,U,V,):
        super(RNNTransducer,self).__init__()
        self.B = lambda : B
        self.T = lambda : T
        self.U = lambda : U
        self.V = lambda : V

        self.enc = self.encoder()
        self.dec = self.decoder()

        print("TYPEEE",type(self.dec.input_shape),self.dec.input_shape)

        prev_pred_init = tf.zeros_initializer()
        self.prev_prediction = tf.Variable(
            initial_value=prev_pred_init(shape=self.dec.input_shape, dtype="float32"), trainable=False
        )

        self.joint = Dense(self.V(), activation="tanh")     
    
    def _summary(self):
        self.summary()
        self.enc.summary()
        self.dec.summary()

    def call(self,inputs):
        encoded_inp = self.enc(inputs)
        decoded_prev = self.dec(self.prev_prediction)

        joint_inp = (
            tf.expand_dims(encoded_inp, axis=2) +                 # [B, T, V] => [B, T, 1, V]
            tf.expand_dims(decoded_prev, axis=1)
        )
        print("JOINT INPut SHAPE",joint_inp.shape)
        output = self.joint(joint_inp)
        print("OUPUTSHAPE",output.shape)
        self.prev_prediction = output

        return output

    def encoder(self):
        num_layers = enc_config["num_layers"]
        lstm_hidden_units = enc_config["lstm_hidden_units"]
        lstm_projection = enc_config["lstm_projection"]
        feature_bins = speech_config["feature_bins"]

        model = Sequential()
        model._init_set_name("encoder")
        model.add(Input(shape=[self.T(),feature_bins], batch_size=self.B(), dtype=tf.float32))
        rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

        for i in range(num_layers):
            model.add(RNN(rnn_cell(),return_sequences=True, name=f"LSTM_ENC_{i+1}"))

        return model
    

    def decoder(self):
    
        num_layers = dec_config["num_layers"]
        lstm_hidden_units = dec_config["lstm_hidden_units"]
        lstm_projection = dec_config["lstm_projection"]
    
        
        inp = Input(shape=[self.U(),self.V()], batch_size=self.B(), dtype=tf.float32)
        rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)
    
        output = inp
        output = RNN(
            tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32),
            return_sequences=True, 
            name=f"LSTM_DEC_1")(output)
    
        for i in range(1,num_layers):
            output = RNN(rnn_cell(),return_sequences=True, name=f"LSTM_DEC_{i+1}")(output)
    
        return Model(inputs=inp,outputs=output,name="decoder")

