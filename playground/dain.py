

import tensorflow as tf

class Adaptive_Normalizer_Layer(tf.keras.layers.Layer):
    def __init__(self, mode = 'full', input_dim = 2):
        super(Adaptive_Normalizer_Layer, self).__init__()
        
        '''
        PARAMETERS
        
        :param mode: Type of normalization to be performed.
                        - 'adaptive_average' performs the adaptive average of the inputs
                        - 'adaptive_scale' performs the adaptive z-score normalization of the inputs
                        - 'full' (Default) performs the complete normalization process: adaptive_average + adaptive_scale + gating
        :param input_dim
        '''
        
        self.mode = mode
        self.x = None

        self.eps = 1e-8
        
        initializer = tf.keras.initializers.Identity()
        gate_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        self.linear_1 = tf.keras.layers.Dense(input_dim, kernel_initializer=initializer, use_bias=False)
        self.linear_2 = tf.keras.layers.Dense(input_dim, kernel_initializer=initializer, use_bias=False)
        self.linear_3 = tf.keras.layers.Dense(input_dim, kernel_initializer=gate_initializer)

    def call(self, inputs):
        # Expecting (n_samples, dim, n_feature_vectors)
        
        def adaptive_avg(inputs):
        
            avg = tf.keras.backend.mean(inputs, 2)
            adaptive_avg = self.linear_1(avg)
            adaptive_avg = tf.keras.backend.reshape(adaptive_avg, (tf.shape(inputs)[0].numpy(), tf.shape(inputs)[1].numpy(), 1))
            x = inputs - adaptive_avg
            
            return x
        
        def adaptive_std(x):
        
            std = tf.keras.backend.mean(x ** 2, 2)
            std = tf.keras.backend.sqrt(std + self.eps)
            adaptive_std = self.linear_2(std)
            adaptive_std = tf.where(tf.math.less_equal(adaptive_std, self.eps), 1, adaptive_std)
            adaptive_std = tf.keras.backend.reshape(adaptive_std, (tf.shape(inputs)[0].numpy(), tf.shape(inputs)[1].numpy(), 1))
            x = x / (adaptive_std)
            
            return x
        
        def gating(x):
            
            gate = tf.keras.backend.mean(x, 2)
            gate = self.linear_3(gate)
            gate = tf.math.sigmoid(gate)
            gate = tf.keras.backend.reshape(gate, (tf.shape(inputs)[0].numpy(), tf.shape(inputs)[1].numpy(), 1))
            x = x * gate
            
            return x
        
        if self.mode == None:
            pass
        
        elif self.mode == 'adaptive_average':
            self.x = adaptive_avg(inputs)
            
        elif self.mode == 'adaptive_scale':
            self.x = adaptive_avg(inputs)
            self.x = adaptive_std(x)
            
        elif self.mode == 'full':
            self.x = adaptive_avg(inputs)
            self.x = adaptive_std(self.x)
            self.x = gating(self.x)
        
        else:
            assert False

        return self.x



example_tensor = tf.constant([
  [[0.0, 1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0, 9.0]],
  [[10.0, 11.0, 12.0, 13.0, 14.0],
   [15.0, 16.0, 17.0, 18.0, 19.0]],
  [[20.0, 21.0, 22.0, 23.0, 24.0],
   [25.0, 26.0, 27.0, 28.0, 29.0]],])

keras_layer = Adaptive_Normalizer_Layer(input_dim = tf.shape(example_tensor)[1].numpy())
output = keras_layer(example_tensor)
print(output)