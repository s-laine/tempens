Implementation of temporal ensembling and Pi-model.
Samuli Laine and Timo Aila, NVIDIA.

Released as part of ICLR 2017 paper submission "Temporal Ensembling for Semi-Supervised Learning".

Additional code (report.py, theano_utils.py, thread_utils.py) by Tero Karras, NVIDIA.
Code in zca_bn.py adapted from Tim Salimans' repository at:
  https://github.com/TimSalimans/weight_norm/blob/master/nn.py

Package versions used when preparing the paper:
- Theano 0.8.0.dev0
- Lasagne 0.2.dev1
- CUDA toolkit 7.5, CUDNN 4004

To configure a training run, edit config.py. To execute, run train.py.
