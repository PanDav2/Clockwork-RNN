# Clockwork RNN

This project is an implementation of the Clockwork RNN
(see [paper](https://arxiv.org/abs/1402.3511)).

### People : Paul Mustière, Pandav2 aka David Panou
### Organization : UPMC - Master Data Science

## Clockwork-RNN

The model can be found under `models/clockwork_rnn.py`.

The current `main.py` replicates the sequence generation task
described in the paper, but the implementation should be able
to handle other tasks.

To monitor the training, you can use TensorBoard:
`̀``
tensorboard --reload_interval 2 --logdir log
`̀``
