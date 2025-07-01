import tensorflow as tf
from tf_agents.trajectories.time_step import StepType

from keras import layers, models, activations
from tf_agents.keras_layers import dynamic_unroll_layer

class FSCLikeActorNetwork(models.Model):
    def __init__(self, observation_shape: tf.TensorShape,
                 action_range: int,
                 memory_len: int,
                 use_one_hot: bool = False,
                 use_residual_connection: bool = True):
        
        super(FSCLikeActorNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_range = action_range
        self.memory_len = memory_len
        self.dense1 = layers.Dense(128, activation='relu')
        self.grumender = layers.Dense(32, activation='tanh')
        self.simple_rnn_for_memory = layers.GRU(
            32, return_sequences=True, return_state=True)
        self.memory_dense = layers.Dense(memory_len, activation=None)
        self.action = layers.Dense(self.action_range, activation=None)
        gumbel_softmax_one_hot = False
        if gumbel_softmax_one_hot:
            self.projection_network = layers.Dense(memory_len, activation='relu')
            self.memory_function = layers.Lambda(
                lambda x: self.gumbel_softmax(x, temperature=0.5))
            self.quantization_layer = layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1),
                                                                         depth=self.memory_len, dtype=tf.float32))
            self.one_hot_constant = 1
        elif not use_one_hot:

            self.memory_function = layers.Lambda(
                lambda x: 1.5 * tf.tanh(x) + 0.5 * tf.tanh(-3 * x))
            self.quantization_layer = layers.Lambda(lambda x: tf.round(x))
            self.one_hot_constant = 0
        else:

            self.memory_function = layers.Lambda(
                lambda x: activations.sigmoid(x))
            self.quantization_layer = layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1),
                                                                         depth=self.memory_len, dtype=tf.float32))
            self.one_hot_constant = 1
        self.noise_level = 0.35

    def get_initial_state(self, batch_size):
        zeros_like = tf.zeros(
            (batch_size, self.memory_len - self.one_hot_constant))
        return tf.concat([tf.ones((batch_size, self.one_hot_constant)), zeros_like], axis=-1)
    
    def sample_gumbel(self, shape, eps = 1e-20):
        """
        Sample from the Gumbel-Softmax distribution.
        """
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax(self, logits, temperature=1.0):
        """
        Sample from the Gumbel-Softmax distribution and return one-hot encoding.
        """
        gumbel_noise = self.sample_gumbel(tf.shape(logits))
        y = logits + gumbel_noise
        y = tf.nn.softmax(y / temperature, axis=-1)
        return y
    
    @tf.function
    def generate_noise(self, inputs):
        """
        Generate noise for the memory function output.
        """
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.noise_level)
        return inputs + noise
    
    def set_noise_level(self, noise_level):
        """
        Set the noise level for the memory function output.
        """
        self.noise_level = noise_level

    @tf.function
    def call(self, inputs, step_type: StepType, old_memory=None):
        # inputs = tf.concat([inputs, tf.cast(old_memory, tf.float32)], axis=-1)
        inputs = tf.where(step_type == StepType.LAST, 
                          tf.zeros_like(inputs), inputs)
        x = self.dense1(inputs)
        if old_memory is None:
            old_memory = self.get_initial_state(tf.shape(x)[0])
        old_memory = self.grumender(old_memory)
        x, memory = self.simple_rnn_for_memory(x, initial_state=old_memory)

        # x2 = self.projection_network(x)
        # x = layers.concatenate(x1, axis=-1)
        memory = self.memory_dense(memory)
        memory = self.memory_function(memory)
        memory += self.generate_noise(memory)
        x_quantized = self.quantization_layer(memory)
        # State-through estimation, where we ignore round(x).
        memory = memory + tf.stop_gradient(x_quantized - memory)
        action = self.action(x)
        

        return action, memory
