import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import callbacks
import tensorflow as tf
np.random.seed(216)


class LowDimensionalEmbedding(object):
    """Returns a low-dimensional embedding of customer transactions"""
    
    def __init__(self, transaction_history, encoding_dimension=25, batch_size=500,
                 epochs=1000, early_stopping=10):
        self.transaction_history = transaction_history
        self.encoding_dimension = encoding_dimension
        self._check_input_params()
        self.autoencoder, self.encoder = self._deep_feedforward_network()
        self.encoder = self._training_deep_feedforward_network()
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        
        
    def _deep_feedforward_network(self):
        
        """ Returns the autoencoder model architecture to train. """
        
        DIM = self.encoding_dimension
        encoding_dim1 = DIM*5
        encoding_dim2 = DIM*4
        encoding_dim3 = DIM*3
        encoding_dim4 = DIM*2
        encoding_dim5 = DIM
            
        inputs = Input(shape=(self.transaction_history.shape[1],))
    
        encoded1 = Dense(encoding_dim1, activation='tanh')(inputs)
        encoded2 = Dense(encoding_dim2, activation='tanh')(encoded1)
        encoded3 = Dense(encoding_dim3, activation='tanh')(encoded2)
        encoded4 = Dense(encoding_dim4, activation='tanh')(encoded3)
        encoded5 = Dense(encoding_dim5, activation='tanh')(encoded4)
        decoded1 = Dense(encoding_dim4, activation='tanh')(encoded5)
        decoded2 = Dense(encoding_dim3, activation='tanh')(decoded1)
        decoded3 = Dense(encoding_dim2, activation='tanh')(decoded2)
        decoded4 = Dense(encoding_dim1, activation='tanh')(decoded3)
        decoded5 = Dense(self.transaction_history.shape[1], activation='softmax')(decoded4)
        
        autoencoder = Model(inputs, decoded5)
        encoder = Model(inputs, encoded5)
            
        return autoencoder, encoder
    
    def _training_deep_feedforward_network(self):
        
        """ Returns the trained autoencoder. """
        
        autoencoder, encoder = self._deep_feedforward_network(self.encoding_dimension)
        
        tf.set_random_seed(216) #for reproduceable results
        EPOCHS = self.epochs
        BATCH_SIZE = self.batch_size
        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
        autoencoder.fit(self.transaction_history, self.transaction_history, epochs=EPOCHS, verbose = 1, 
                                  batch_size=BATCH_SIZE, shuffle=True,
                          callbacks = [callbacks.EarlyStopping(monitor='loss', 
                                                               patience=self.early_stopping, 
                                                               mode='auto')])
        return encoder
        
    def encoding_creation(self):
        
        """ Returns encoded transaction history. """
        
        encoder = self._training_deep_feedforward_network()
        encoded_transaction_history = pd.DataFrame(encoder.predict(self.transaction_history))
        encoded_transaction_history.insert(loc=0, column='id', value=self.transaction_history.columns.values)
        encoded_transaction_history = encoded_transaction_history.set_index(encoded_transaction_history.id).drop(labels = ['id'], axis = 1)
        
        return encoded_transaction_history