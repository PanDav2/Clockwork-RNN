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
                - Number of unfolded steps: num_steps

                - Learning rate: learning_rate
                - Learning rate: learning_rate_step
                - Learning rate: learning_rate_decay
                - Momentum: momentum
        """

        # Check config
        for v in ['input_dim', 'hidden_dim', 'output_dim', 'periods', 'num_steps', 'learning_rate', 'learning_rate_step', 'learning_rate_decay', 'momentum']:
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

        self.create_optimizer()

        self.create_summaries()

        # Count number of parameters used
        self.num_parameters = 0
        for v in tf.trainable_variables():
            parameters = 1
            for dim in v.get_shape():
                parameters *= int(dim)
            self.num_parameters += parameters

    def create_model(self):
        # Model options
        weights_initializer = tf.random_normal_initializer(stddev = 0.1)
        biases_initializer = tf.zeros_initializer

        hidden_activation = tf.tanh
        output_activation = tf.sigmoid

        # Create weights & biases

        with tf.variable_scope('input_parameters'):
            Wi = tf.get_variable('weights', [self.config['input_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bi = tf.get_variable('biases', self.config['hidden_dim'], initializer = biases_initializer)

        with tf.variable_scope('hidden_parameters'):
            Wh = tf.get_variable('weights', [self.config['hidden_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bh = tf.get_variable('biases', self.config['hidden_dim'], initializer = biases_initializer)

            clockwork_mask = np.zeros((self.config['hidden_dim'], self.config['hidden_dim']))
            block_size = self.config['hidden_dim'] / len(self.config['periods'])
            for i in range(len(self.config['periods'])):
                clockwork_mask[i * block_size:(i+1) * block_size, i * block_size:] = 1.0
            clockwork_mask = tf.constant(clockwork_mask, name = 'clockwork_mask', dtype = tf.float32)

            Wh = tf.mul(clockwork_mask, Wh)

        with tf.variable_scope('output_parameters'):
            Wo = tf.get_variable('weights', [self.config['hidden_dim'], self.config['output_dim']], initializer = weights_initializer)
            bo = tf.get_variable('biases', self.config['output_dim'], initializer = biases_initializer)

        state = tf.expand_dims(self.initial_state, 0, name = 'state')
        outputs = []

        with tf.variable_scope('clockwork_rnn'):
            for t in range(self.config['num_steps']):
                for i in range(len(self.config['periods'])):
                    if t % self.config['periods'][::-1][i] == 0:
                        active_rows = block_size * (len(self.config['periods'])-i)
                        break

                x = tf.slice(self.inputs, [t, 0], [1, -1])

                # Compute the partial new state
                Wi_x = tf.matmul(x, tf.slice(Wi, [0, 0], [-1, active_rows]))
                Wi_x = tf.nn.bias_add(Wi_x, tf.slice(bi, [0], [active_rows]), name = 'Wi_x')

                Wh_y = tf.matmul(state, tf.slice(Wh, [0, 0], [-1, active_rows]))
                Wh_y = tf.nn.bias_add(Wh_y, tf.slice(bh, [0], [active_rows]), name = 'Wh_y')

                partial_state = hidden_activation(tf.add(Wi_x, Wh_y))

                # Concatenate the partial state with old state values
                # for unactivated blocks
                state = tf.concat(1, [partial_state, tf.slice(state, [0, active_rows], [-1, -1])], name = 'new_state')

                # Compute the new output
                Wo_s = tf.matmul(state, Wo)
                Wo_s = tf.nn.bias_add(Wo_s, bo, name = 'Wo_s')

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
                                staircase = True)

        self.optimizer = tf.train.MomentumOptimizer(self.config['learning_rate'], self.config['momentum'], use_nesterov = True)

        self.train_step = self.optimizer.minimize(self.loss, global_step = self.global_step)

    def create_summaries(self):
        with tf.variable_scope('train'):
            learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)
            loss_summary = tf.summary.scalar('loss', self.loss)

        self.summaries = tf.summary.merge([learning_rate_summary, loss_summary])
