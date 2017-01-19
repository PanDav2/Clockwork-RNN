import numpy as np
import tensorflow as tf

class ClockworkRNN():
    """
        Implementation of Clockwork RNN based on
        A Clockwork RNN - Koutnik et al. 2014 [arXiv, https://arxiv.org/abs/1402.3511]

        Recommended learning rate of 1e-2
    """

    def __init__(self, config, initial_state = None):
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
                - Optimizer used (momentum, rmsprop): optimizer
                - Momentum: momentum
        """

        # Check config
        for v in ['input_dim', 'hidden_dim', 'output_dim', 'periods', 'num_steps', 'learning_rate', 'learning_rate_step', 'learning_rate_decay', 'optimizer', 'momentum']:
            if v not in config:
                print('Missing config[\'{}\']'.format(v))
                exit(1)

        if config['hidden_dim'] % len(config['periods']) != 0:
            print('Cannot divide the hidden state into {} blocks'.format(len(config['periods'])))

        self.config = config
        self.num_periods = len(self.config['periods'])
        self.block_size = self.config['hidden_dim'] / self.num_periods

        # Create placeholders for inputs & targets
        self.inputs = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['input_dim']], name = 'inputs')
        self.initial_state = initial_state
        self.targets = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['output_dim']], name = 'targets')

        self.create_model()

        self.create_optimizer()

        self.create_summaries()

        # Count number of parameters used
        self.num_parameters = 0
        self.num_parameters += self.config['input_dim'] * (self.config['hidden_dim'] + 1) # Input weights/biases
        self.num_parameters += self.config['output_dim'] * (self.config['hidden_dim'] + 1) # Output weights/biases
        self.num_parameters += self.num_periods # Periods

        self.num_parameters += self.config['hidden_dim'] # Hidden biases
        # Hidden weights: upper-triangular matrix of num_periods blocks, each of size block_size*block_size
        self.num_parameters += (self.num_periods * (self.num_periods + 1) / 2) * (self.block_size * self.block_size)

    def create_model(self):
        # Model options
        weights_initializer = tf.random_normal_initializer(stddev = 0.1)
        biases_initializer = tf.zeros_initializer

        hidden_activation = tf.tanh
        output_activation = tf.tanh

        # Create weights & biases

        if self.config['input_dim'] != 0:
            with tf.variable_scope('input_parameters'):
                Wi = tf.get_variable('weights', [self.config['hidden_dim'], self.config['input_dim']], initializer = weights_initializer)
                bi = tf.get_variable('biases', [self.config['hidden_dim'], 1], initializer = biases_initializer)
        else:
            Wi_x = tf.constant(0, tf.float32)

        with tf.variable_scope('hidden_parameters'):
            Wh = tf.get_variable('weights', [self.config['hidden_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bh = tf.get_variable('biases', [self.config['hidden_dim'], 1], initializer = biases_initializer)

            clockwork_mask = np.zeros((self.config['hidden_dim'], self.config['hidden_dim']))
            for i in range(self.num_periods):
                clockwork_mask[i * self.block_size:(i+1) * self.block_size, i * self.block_size:] = 1.0
            clockwork_mask = tf.constant(clockwork_mask, name = 'clockwork_mask', dtype = tf.float32)

            Wh = tf.mul(clockwork_mask, Wh)

        with tf.variable_scope('output_parameters'):
            Wo = tf.get_variable('weights', [self.config['output_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bo = tf.get_variable('biases', [self.config['output_dim'], 1], initializer = biases_initializer)

        if self.initial_state:
            state = self.initial_state
        else:
            state = tf.zeros((self.config['hidden_dim'], 1))
        outputs = []

        with tf.variable_scope('clockwork_rnn'):
            for t in range(self.config['num_steps']):
                for i in range(self.num_periods):
                    if t % self.config['periods'][::-1][i] == 0:
                        active_rows = self.block_size * (self.num_periods-i)
                        break

                x = tf.reshape(tf.slice(self.inputs, [t, 0], [1, -1]), (-1, 1), name = 'input')

                # Compute the partial new state
                if self.config['input_dim'] != 0:
                    Wi_x = tf.matmul(tf.slice(Wi, [0, 0], [active_rows, -1]), x)
                    Wi_x = tf.add(tf.slice(bi, [0, 0], [active_rows, -1]), Wi_x, name = 'Wi_x')

                Wh_y = tf.matmul(tf.slice(Wh, [0, 0], [active_rows, -1]), state)
                Wh_y = tf.add(tf.slice(bh, [0, 0], [active_rows, -1]), Wh_y, name = 'Wh_y')

                partial_state = hidden_activation(tf.add(Wi_x, Wh_y))

                # Concatenate the partial state with old state values
                # for unactivated blocks
                state = tf.concat(0, [partial_state, tf.slice(state, [active_rows, 0], [-1, -1])], name = 'new_state')

                # Compute the new output
                Wo_s = tf.matmul(Wo, state)
                Wo_s = tf.add(bo, Wo_s, name = 'Wo_s')

                output = output_activation(Wo_s, name = 'output')
                outputs.append(output)

            self.outputs = tf.concat(0, outputs, name = 'outputs')

            # Compute the loss
            self.errors = tf.reduce_sum(tf.square(self.targets - self.outputs), reduction_indices = 1)
            self.loss  = tf.reduce_mean(self.errors, name = 'loss')

    def create_optimizer(self):
        self.global_step = tf.Variable(0, trainable = False, name = 'global_step')
        self.learning_rate = tf.train.exponential_decay(
                                self.config['learning_rate'],       # Base learning rate.
                                self.global_step,                   # Current index into the dataset.
                                self.config['learning_rate_step'],  # Decay step.
                                self.config['learning_rate_decay'], # Decay rate.
                                staircase = False)

        if self.config['optimizer'] == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.config['momentum'], use_nesterov = True)
        elif self.config['optimizer'] == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            print('Unknown optimizer {}'.format(self.config['optimizer']))
            exit(1)

        self.train_step = self.optimizer.minimize(self.loss, global_step = self.global_step)

    def create_summaries(self):
        with tf.variable_scope('train'):
            learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)
            loss_summary = tf.summary.scalar('loss', self.loss)

        self.summaries = tf.summary.merge([learning_rate_summary, loss_summary])
