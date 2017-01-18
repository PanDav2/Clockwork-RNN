#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import argparse
from preprocess import preprocess

from models.clockwork_rnn import ClockworkRNN
NUM_STEP = 100
if __name__ == '__main__':
    ### Create the Clockwork RNN ###
    config = {
        'input_dim': 0,
        'hidden_dim': 90,
        'output_dim': 1,
        'periods': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'num_steps': NUM_STEP,

        'learning_rate': 3e-4,
        'learning_rate_step': 200,
        'learning_rate_decay': 0.95,
        'momentum': 0.95,
        'max_epochs': 2000
    }

    clockworkRNN = ClockworkRNN(config)
    # We are feeding the wav file
    p = preprocess() ## DEFAULT FILENAME IS IN PREPROCESS
    p.normalize()
    # plotting the signal we are operating on
    p.show_signal()
    signal = p.seek(1,NUM_STEP)
    # plotting the signal we will feed into the RNN
    p.show_signal(signal)
    

    ### Create a session ###
    with tf.Session() as sess:
        # Initialize variables
        tf.initialize_all_variables().run()

        # Create a writer
        log_writer = tf.train.SummaryWriter('log', sess.graph, flush_secs = 2)

        # Load data


        # Dummy data for now
        data_dict = {
            clockworkRNN.initial_state: np.zeros((config['hidden_dim'],)),
            clockworkRNN.inputs: np.zeros((config['num_steps'], config['input_dim'])),
            #clockworkRNN.targets: np.reshape(np.sin(np.arange(config['num_steps'])), (config['num_steps'], config['output_dim']))
            clockworkRNN.targets: signal.reshape((-1,1))
        }

        for epoch in range(config['max_epochs']):
            results = sess.run([clockworkRNN.train_step, clockworkRNN.summaries],
                feed_dict = data_dict)

            log_writer.add_summary(results[1], global_step = tf.train.global_step(sess, clockworkRNN.global_step))
