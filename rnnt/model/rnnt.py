from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
import tensorflow as tf
from tensorflow import keras
from config.config import model_config, speech_config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, RNN, LayerNormalization
from tensorflow.keras import Input
import numpy as np

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
        self.dec = Decoder()

        self.joint = Dense(self.V(), activation="tanh")     
    
    def _summary(self):
        self.summary()
        self.enc.summary()
        self.dec.summary()

    def call(self,x):
        y_batch = []
        B = self.B()
        print("ENCODER INPUT SHAPE",x.shape)
        encoder_out = self.enc(x)
        U = self.U()
        NULL_INDEX=0
        start_symbol = NULL_INDEX

        for b in range(B):
            t = 0
            u = 0
            y = [start_symbol]; 
            decoder_state = None
            record_timesteps = x[b].shape[0]
            while t < record_timesteps and u < U:
                g_u = self.dec(np.array([[[float(y[-1])]]]))  #TODO: update states ???
                f_t = encoder_out[b,t]
                joint_inp = f_t + g_u

                h_t_u = self.joint(joint_inp)

                argmax = tf.math.argmax(h_t_u,axis=2).numpy()[0][0]
                #argmax = h_t_u.max(-1)[1].item()
                
                if argmax == NULL_INDEX:
                    t += 1
                else: # argmax == a label
                    u += 1
                    t += 1   #IMPORTANT NOTE: DIFFERENT FROM NOTEBOOK CODE
                    y.append(argmax)
            #print(y)
            y_batch.append(y[1:]) # remove start symbol

        return y_batch

    def encoder(self):
        num_layers = enc_config["num_layers"]
        lstm_hidden_units = enc_config["lstm_hidden_units"]
        lstm_projection = enc_config["lstm_projection"]
        feature_bins = speech_config["feature_bins"]

        model = Sequential()
        model._init_set_name("encoder")
        model.add(Input(shape=[None,feature_bins], batch_size=self.B(), dtype=tf.float32))
        rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

        for i in range(num_layers):
            model.add(RNN(rnn_cell(),return_sequences=True, name=f"LSTM_ENC_{i+1}"))

        return model
    
       

class Decoder(keras.Model):
    def __init__(self):
        super(Decoder,self).__init__()
        num_layers = dec_config["num_layers"]
        lstm_hidden_units = dec_config["lstm_hidden_units"]
        lstm_projection = dec_config["lstm_projection"]
        
        rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_hidden_units, num_proj=lstm_projection, dtype=tf.float32)

        self.lstms = [RNN(rnn_cell(),return_sequences=True, name=f"LSTM_DEC_{i+1}") for i in range(num_layers)]


    def call(self, x):
        for lstm in self.lstms:
            x = lstm(x)
        
        return x
