# KGD
A Python implementation of the KGD optimization algorithms and accompanying experiments introduced in https://arxiv.org/pdf/1810.12273.pdf. 

Depdendencies are the usual scientific computing packages, as well as Tensorflow and the package Autograd (https://github.com/HIPS/autograd).

#### `optimization.py`
`optimization.py` provides three optimizers: `gd(..)`, `momgd(...)`, and `rmsprop(...)`. For each optimizer, the user can select `how=regular` for the usual verson of gradient descent, gradient descent with momentum, and RMSProp respectively. Alternately, the user can select `how='filtered'` to use Kalman filtering (using `kalman_filter.py`) for adaptive gradient variance reduction.

`opt_tests.py` provides simple unit tests for all three optimizers described above. 

#### `dist_optimization.py`
`dist_optimization.py` provides the distributed variant of Kalman RMSProp described in the paper. There is no option for unfiltered optimization, nor are the other optimization algorithms implemented at present. The algorithm does not (currently) take advantage of multiprocessing or multi-machine architectures, though the code has been written with this possibility in mind. 

`packunpack.py` provides some helper methods used to package and un-package the objective function and its arguments for compatibility with `dist_optimization.py`. `dist_opt_tests.py` provides some unit tests for the `dist_optimization.py` package.

### Examples
The Examples folder contains the experiments from the paper. They cover simple optimization, black box variational inference, neural network regression, and MNIST digit recognition. These can be used to understand how the optimization algorithms provided may be used in practice (particularly the distributed version).
