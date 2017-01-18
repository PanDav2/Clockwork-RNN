import numpy as np
import tensorflow as tf

class ClockworkRNN():
    """
        Implementation of Clockwork RNN based on
        A Clockwork RNN - Koutnik et al. 2014 [arXiv, https://arxiv.org/abs/1402.3511]
    """

    def __init__(self, config):
        """
            Config parameters:
                - Input dimension: input_dim
                - Hidden state dimension: hidden_dim
                - Output dimension: output_dim
                - Module periods: periods

                - Number of steps: num_steps
        """
        print('Config: {}'.format(config))

        # Check config
        for v in ['input_dim', 'hidden_dim', 'output_dim', 'periods', 'num_steps']:
            if v not in config:
                print('Missing config[\'{}\']'.format(v))
                exit(1)

        if config['hidden_dim'] % len(config['periods']) != 0:
            print('Cannot divide the hidden state into {} blocks'.format(len(config['periods'])))

        self.config = config

        # Create placeholders for input & targets
        self.inputs = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['input_dim']], name = 'inputs')
        self.initial_state = tf.placeholder(tf.float32, shape = [self.config['hidden_dim']], name = 'initial_state')
        self.targets = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['output_dim']], name = 'targets')

        self.create_model()

    def create_model(self):
        # Model options
        weights_initializer = tf.random_normal_initializer(stddev = 0.1)
        biases_initializer = tf.zeros_initializer

        hidden_activation = tf.tanh
        output_activation = tf.sigmoid

        # Create weights & biases

        with tf.variable_scope('input'):
            Wi = tf.get_variable('weights', [self.config['input_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bi = tf.get_variable('biases', self.config['hidden_dim'], initializer = biases_initializer)

        with tf.variable_scope('hidden'):
            Wh = tf.get_variable('weights', [self.config['hidden_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bh = tf.get_variable('biases', self.config['hidden_dim'], initializer = biases_initializer)

            clockwork_mask = np.zeros((self.config['hidden_dim'], self.config['hidden_dim']))
            block_size = self.config['hidden_dim'] / len(self.config['periods'])
            for i in range(len(self.config['periods'])):
                clockwork_mask[i * block_size:(i+1) * block_size, i * block_size:] = 1.0
            clockwork_mask = tf.constant(clockwork_mask, name = 'clockwork_mask')

            Wh = tf.matmul(clockwork_mask, Wh)

        with tf.variable_scope('output'):
            Wo = tf.get_variable('weights', [self.config['hidden_dim'], self.config['output_dim']], initializer = weights_initializer)
            bo = tf.get_variable('biases', self.config['output_dim'], initializer = biases_initializer)


        # return tf.get_variable('weights', shape, initializer = tf.random_normal_initializer(stddev=0.01))
        # return tf.get_variable('biases', shape, initializer = tf.zeros_initializer)
