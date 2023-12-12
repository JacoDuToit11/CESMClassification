#----- Attention mechanims -----#

# imports
import tensorflow as tf
import numpy as np

system = 'mac'
# system = 'ubuntu'

if system == 'mac':
    # MAC
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Flatten, concatenate, Layer, LSTM
    import tensorflow.keras.backend as K
else:
    # UBUNTU
    from keras_preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from keras.models import Model, clone_model, load_model
    from keras.layers import Dense, Embedding, GlobalMaxPool1D, MaxPool1D, Dropout, Conv1D, Input, Flatten, concatenate, Layer, LSTM
    from keras import Model
    from keras import utils 
    import keras.backend as K

# Label-wise attention layer
class embedded_label_wise_attention(Layer):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(embedded_label_wise_attention,self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape = (self.num_classes, input_shape[2]), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias',  shape = (self.num_classes, input_shape[1]), 
                               initializer='zeros', trainable=True)        
        super(embedded_label_wise_attention, self).build(input_shape)

    def call(self, x):
        output = tf.matmul(x, self.W, transpose_b = True)
        output = tf.transpose(output, perm=[0, 2, 1])
        output = tf.add(output, self.b)

        # Compute the weights
        alpha = K.softmax(output)

        # Compute the context vector
        context = K.batch_dot(alpha, x)
        return context

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

# Label embedding attention mechanism
class general_attention(Layer):
    def __init__(self, **kwargs):
        super(general_attention,self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape = (input_shape[1][2], input_shape[0][2]), 
                               initializer='random_normal', trainable = True)
        self.p = self.add_weight(name='linear_transformation_weight', shape = (input_shape[1][1], 1), 
                               initializer='random_normal', trainable = True)
        super(general_attention, self).build(input_shape)

    def call(self, inputs):
        x, query_matrix = inputs
        
        new_query_matrix = []
        for i in range(len(query_matrix)):
            temp = tf.matmul(tf.transpose(query_matrix[i]), self.p)
            new_query_matrix.append(temp)
        
        new_query_matrix = tf.squeeze(tf.convert_to_tensor(new_query_matrix))

        first = tf.matmul(new_query_matrix, self.W)
        output = tf.matmul(first, x, transpose_b = True)

        # Compute the weights
        # softmax of last layour because we want to know how much attention to assign to the ith feature for label j
        alpha = K.softmax(output)

        # Compute the context vector
        context = K.batch_dot(alpha, x)
        return context

    def get_config(self):
        config = super().get_config().copy()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Output layer of attention mechanisms
class output_layer(Layer):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(output_layer,self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='output_weight', shape = (self.num_classes, input_shape[2]), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='output_bias',  shape = (self.num_classes, 1), 
                               initializer='zeros', trainable=True)        
        super(output_layer, self).build(input_shape)

    def call(self, x):
        output = x * self.W
        ones = np.ones(shape = (x.shape[2], 1), dtype = np.float32)
        ones = tf.convert_to_tensor(ones)

        result = tf.matmul(output, ones)
        result = tf.add(result, self.b)
        result = tf.squeeze(result, axis = 2)
        result = tf.keras.activations.softmax(result)
        return result
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)