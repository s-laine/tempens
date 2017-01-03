Implementation of temporal ensembling and Pi-model.
Samuli Laine and Timo Aila, NVIDIA.

Released as part of ICLR 2017 paper submission "Temporal Ensembling for Semi-Supervised Learning".

Additional code (report.py, theano_utils.py, thread_utils.py) by Tero Karras, NVIDIA.
Code in zca_bn.py adapted from Tim Salimans' repository at:
  https://github.com/TimSalimans/weight_norm/blob/master/nn.py

Package versions used when preparing the paper:
- Theano 0.9.0.dev4
- Lasagne 0.2.dev1
- CUDA toolkit 8.0, CUDNN 5105

To configure a training run, edit config.py. To execute, run train.py.
