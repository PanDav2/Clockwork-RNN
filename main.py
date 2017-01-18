#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from models.clockwork_rnn import ClockworkRNN

if __name__ == '__main__':
    ### Create the Clockwork RNN ###
    config = {
        'input_dim': 2,
        'hidden_dim': 90,
        'output_dim': 1,
        'periods': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'num_steps': 100,

        'learning_rate': 3e-4,
        'learning_rate_step': 100,
        'learning_rate_decay': 0.95,
        'momentum': 0.95,
        'max_epochs': 200
    }

    clockworkRNN = ClockworkRNN(config)

    ### Create a session ###
    with tf.Session() as sess:
        # Initialize variables
        tf.initialize_all_variables().run()

        # Create a writer
        log_writer = tf.train.SummaryWriter('log', sess.graph, flush_secs = 2)

        for epoch in range(config['max_epochs']):
            results = sess.run([clockworkRNN.train_step, clockworkRNN.summaries],
                feed_dict = {clockworkRNN.initial_state: np.ones((config['hidden_dim'],)),
                            clockworkRNN.inputs: np.ones((config['num_steps'], config['input_dim'])),
                            clockworkRNN.targets: np.zeros((config['num_steps'], config['output_dim']))})

            log_writer.add_summary(results[1], global_step = tf.train.global_step(sess, clockworkRNN.global_step))
