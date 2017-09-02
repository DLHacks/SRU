# The Statistical Recurrent Unit

- authors: Junier B. Oliva, Barnabas Poczos, Jeff Schneider
- arxiv: https://arxiv.org/abs/1703.00381
- Pytorch implemention of the experiment of SRU with pixel-by-pixel sequential MNIST.
- Powered by [DL HACKS](http://deeplearning.jp/hacks/)


## Requirements

- environment: python3.5
- pytorch 0.2.0
- hyperopt 0.1
- numpy 1.13.1
- scikit-learn 0.18.2

## Implement

- `python main.py sru`: trainning RNNs with fixed parameters.
- `python tune_params.py sru` : tuning hyper parameters with hyperopt.
- Choose your model from [sru, gru, lstm]
- If you need more information, please run `python tune_params.py --help`.


## notes

- I choose Adam for optimization, though SGD is used in the paper. (It might converge faster)
- weight_decay is used. (The paper doesn't refer to it)
