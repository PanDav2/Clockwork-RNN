#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import argparse
from preprocess import preprocess
import matplotlib.pyplot as plt

from models.clockwork_rnn import ClockworkRNN
if __name__ == '__main__':
    ### Create the Clockwork RNN ###
    config = {
        'input_dim': 0,
        'hidden_dim': 9,
        'output_dim': 1,
        'periods': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'num_steps': 320,

        'learning_rate': 3e-4,
        'learning_rate_step': 200,
        'learning_rate_decay': 0.95,
        'momentum': 0.95,
        'max_epochs': 5000
    }

    ### Create the model ###
    clockworkRNN = ClockworkRNN(config)
    print('The Clockwork RNN has {} parameters'.format(clockworkRNN.num_parameters))

    # Change this for a different log/ subfolder
    experiment_name = 'clockwork_rnn_{}params'.format(clockworkRNN.num_parameters)

    ### Load data ###

    # Use fake data: generated from a sinusoid
    # targets = np.reshape(np.sin(np.arange(config['num_steps'])), (config['num_steps'], config['output_dim']))

    # Use a WAV file and normalize it
    p = preprocess()
    p.slice(0, config['num_steps']) # Select values BEFORE normalization
    p.normalize()
    # Reshape signal for model
    targets = p.get_signal().reshape((-1, 1))

    # Plot signal
    # p.show_signal()

    ### Create a session ###
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()

        # Create a writer
        log_writer = tf.summary.FileWriter('log/' + experiment_name, sess.graph, flush_secs = 2)

        data_dict = {
            clockworkRNN.initial_state: np.zeros((config['hidden_dim'],)),
            clockworkRNN.inputs: np.zeros((config['num_steps'], config['input_dim'])),
            clockworkRNN.targets: targets
        }

        for epoch in range(config['max_epochs']):
            results = sess.run([clockworkRNN.train_step, clockworkRNN.summaries],
                feed_dict = data_dict)

            log_writer.add_summary(results[1], global_step = tf.train.global_step(sess, clockworkRNN.global_step))

        # After training, do a final pass evaluate & plot the result
        error, outputs = sess.run([clockworkRNN.loss, clockworkRNN.outputs], feed_dict = data_dict)
        outputs = outputs.reshape(-1)
        ground_truth = targets.reshape(-1)

        print(outputs)
        # Final result
        print('')
        print('')
        print('After {} epochs, error is {}'.format(config['max_epochs'], error))

        plt.plot(ground_truth, '--')
        plt.plot(outputs)
        plt.legend(['Target signal', 'Clockwork RNN output'])
        plt.show()
