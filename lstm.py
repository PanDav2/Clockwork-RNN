import numpy as np
import tensorflow as tf


class LSTM():
	"""
        Implementation of Clockwork RNN based on
        Long Short Term Memory - Hochreiter, Schmidhuber et al. 2014 [paper ,http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf]
    """


    def __init__(self,config):
    	"""
            Config parameters:
                - Input dimension: input_dim
                - Hidden state dimension: hidden_dim
                - Output dimension: output_dim
                - Module periods: periods
                - Number of unfolded steps: num_steps

                - Learning rate: learning_rate
                - Learning rate: learning_rate_step
                - Learning rate: learning_rate_decay
                - Momentum: momentum
        """

        # Check config
        for v in ['input_dim', 'hidden_dim', 'output_dim', 'num_steps', 'learning_rate', 'learning_rate_step', 'learning_rate_decay', 'momentum','optimizer']:
            if v not in config:
                print('Missing config[\'{}\']'.format(v))
                exit(1)

        self.config = config

        self.inputs = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['input_dim']], name = 'inputs')
        self.initial_state = tf.placeholder(tf.float32, shape = [self.config['hidden_dim']], name = 'initial_state')
        self.targets = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['output_dim']], name = 'targets')

        self.create_model()
        self.create_optimizer()
        self.create_summaries()

        # Count number of parameters used
        self.num_parameters = 0
        self.num_parameters += self.config['input_dim'] * (self.config['hidden_dim'] + 1) # Input weights/biases
        self.num_parameters += self.config['output_dim'] * (self.config['hidden_dim'] + 1) # Output weights/biases

        self.num_parameters += self.config['hidden_dim'] # Hidden biases
        # Hidden weights: upper-triangular matrix of num_periods blocks, each of size block_size*block_size
        self.num_parameters += (self.num_periods * (self.num_periods + 1) / 2) * (self.block_size * self.block_size)

	def create_model(self):
		# Model options
		# Create weights & biases

        with tf.variable_scope('input_parameters'):
            Wi = tf.get_variable('weights', [self.config['input_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bi = tf.get_variable('biases', self.config['hidden_dim'], initializer = biases_initializer)

        with tf.variable_scope('hidden_parameters'):
            Wh = tf.get_variable('weights', [self.config['hidden_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bh = tf.get_variable('biases', self.config['hidden_dim'], initializer = biases_initializer)

	def create_optimizer(self):
	        self.global_step = tf.Variable(0, trainable = False, name = 'global_step')
	        self.learning_rate = tf.train.exponential_decay(
	                                self.config['learning_rate'],       # Base learning rate.
	                                self.global_step,                   # Current index into the dataset.
	                                self.config['learning_rate_step'],  # Decay step.
	                                self.config['learning_rate_decay'], # Decay rate.
	                                staircase = True)

	        self.optimizer = tf.train.MomentumOptimizer(self.config['learning_rate'], self.config['momentum'], use_nesterov = True)

    def create_summaries(self):
        with tf.variable_scope('train'):
            learning_rate_summary = tf.scalar_summary('learning_rate', self.learning_rate)
            loss_summary = tf.scalar_summary('loss', self.loss)

        self.summaries = tf.merge_summary([learning_rate_summary, loss_summary])

