# [WIP] Reproducible Experiments of "The Statistical Recurrent Unit"

- authors: Junier B. Oliva, Barnabas Poczos, Jeff Schneider
- arxiv: https://arxiv.org/abs/1703.00381


## Description

- This repository is designed for implementing reproducible experiments of the SRU experiment with pixel-by-pixel sequential MNIST.
- environment: python3.5


## Implement

- `python tune_params.py sru` : tuning hyper parameters with hyperopt.
- Choose the model to be tuned from [sru, gru, lstm]
- If you need more information, please run `python tune_params.py --help`.


## notes

- I choose Adam for optimization, though SGD is used in the paper. (It might converge faster)
- weight_decay is used. (The paper doesn't refer to it)
