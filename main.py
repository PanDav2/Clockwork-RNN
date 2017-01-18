#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from models.clockwork_rnn import ClockworkRNN

if __name__ == '__main__':
    print('Clockwork-RNN')

    config = {
        'input_dim': 1,
        'hidden_dim': 90,
        'output_dim': 1,
        'periods': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'num_steps': 100
    }
    clockworkRNN = ClockworkRNN(config)
