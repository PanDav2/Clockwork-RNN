import numpy as np
import tensorflow as tf

class LSTM():
    """
        Implementation of Long-Short Term Memory cells
        using TensorFlow's BasicLSTMCell

        Can be difficult to train, use a learning rate of 1e-3
    """

    def __init__(self, config, initial_state = None):
        """
            Config parameters:
                - Input dimension: input_dim
                - Hidden state dimension: hidden_dim
                - Output dimension: output_dim
                - Number of unfolded steps: num_steps

                - Learning rate: learning_rate
                - Learning rate: learning_rate_step
                - Learning rate: learning_rate_decay
                - Optimizer used (momentum, rmsprop): optimizer
                - Momentum: momentum
        """

        # Check config
        for v in ['input_dim', 'hidden_dim', 'output_dim', 'num_steps', 'learning_rate', 'learning_rate_step', 'learning_rate_decay', 'optimizer', 'momentum']:
            if v not in config:
                print('Missing config[\'{}\']'.format(v))
                exit(1)

        self.config = config

        # Create placeholders for inputs & targets
        self.inputs = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['input_dim']], name = 'inputs')
        self.initial_state = initial_state
        self.targets = tf.placeholder(tf.float32, shape = [self.config['num_steps'], self.config['output_dim']], name = 'targets')

        self.create_model()

        self.create_optimizer()

        self.create_summaries()

        # Count number of parameters used
        self.num_parameters = 0
        for v in tf.trainable_variables():
            if str(v.name).startswith('lstm') or str(v.name).startswith('output_parameters'):
                parameters = 1
                for dim in v.get_shape():
                    parameters *= int(dim)
                self.num_parameters += parameters

    def create_model(self):
        # Model options
        weights_initializer = tf.random_normal_initializer(stddev = 0.1)
        biases_initializer = tf.zeros_initializer

        forget_bias = 5.0
        hidden_activation = tf.tanh
        output_activation = tf.tanh

        input_dropout = 0.9
        output_dropout = 0.9

        # Create LSTM cell

        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config['hidden_dim'], forget_bias = forget_bias, activation = hidden_activation)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = input_dropout, output_keep_prob = output_dropout)

        with tf.variable_scope('output_parameters'):
            Wo = tf.get_variable('weights', [self.config['output_dim'], self.config['hidden_dim']], initializer = weights_initializer)
            bo = tf.get_variable('biases', [self.config['output_dim'], 1], initializer = biases_initializer)

        if self.initial_state:
            state = self.initial_state
        else:
            state = lstm.zero_state(1, tf.float32)
        outputs = []

        with tf.variable_scope('lstm') as scope:
            for t in range(self.config['num_steps']):
                if t > 0:
                    scope.reuse_variables()

                if self.config['input_dim'] == 0:
                    x = tf.zeros((1, 1), name = 'input')
                else:
                    x = tf.slice(self.inputs, [t, 0], [1, -1], name = 'input')

                h, state = lstm(x, state)

                Wo_s = tf.matmul(Wo, tf.reshape(h, (self.config['hidden_dim'], 1)))
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

        # Gradient clipping for stability
        gradients = self.optimizer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_norm(grad, 500.0), var) for grad, var in gradients]
        self.train_step = self.optimizer.apply_gradients(clipped_gradients, global_step = self.global_step)

        # self.train_step = self.optimizer.minimize(self.loss, global_step = self.global_step)

    def create_summaries(self):
        with tf.variable_scope('train'):
            learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)
            loss_summary = tf.summary.scalar('loss', self.loss)

        self.summaries = tf.summary.merge([learning_rate_summary, loss_summary])
