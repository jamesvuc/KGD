"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam, rmsprop

# from data import load_mnist
import tensorflow as tf
from tensorflow import keras
mnist=tf.keras.datasets.mnist

import sys
sys.path.append('../')
from dist_optimization import distributed_opt

np.random.seed(0)

"""
    This implements a distributed Kalman RMSProp optimizer from 
    https://arxiv.org/pdf/1810.12273.pdf. 

    The neural network methods below that are not specific to Kalman RMSProp are
    heavily based on https://github.com/HIPS/autograd/blob/master/examples/neural_net.py.

"""

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == '__main__':
    # Model parameters
    layer_sizes = [784, 10, 10, 10]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    # batch_size = 256# big
    batch_size = 32# small
    num_epochs = 2
    # num_epochs = 5
    # num_epochs = 8
    step_size = 0.001

    print("Loading training data...")
    # N, train_images, train_labels, test_images,  test_labels = load_mnist()
    (train_images, train_labels), (test_images,  test_labels) = mnist.load_data()
    
    train_images = train_images.reshape(train_images.shape[0], 28*28)/255.0
    test_images = test_images.reshape(test_images.shape[0], 28*28)/255.0

    # print(train_images.shape)
    train_labels=keras.utils.to_categorical(train_labels, 10)
    test_labels=keras.utils.to_categorical(test_labels, 10)

    init_params = init_random_params(param_scale, layer_sizes)

    num_batches = int(np.ceil(len(train_images) / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    hist={
        'iters':[],
        'testacc':[],
        'trainacc':[],
        'loss':[]
    }
    print("     Epoch     |    Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        # if iter % num_batches == 0:
        if iter % 5 == 0 :
            print(iter)
            train_acc = accuracy(params, train_images, train_labels)
            test_acc  = accuracy(params, test_images, test_labels)
            loss = objective(params, iter)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))
            print('loss=',loss)

            hist['iters'].append(iter)
            hist['testacc'].append(test_acc)
            hist['trainacc'].append(train_acc)
            hist['loss'].append(loss)
    
    
    
    import datetime as dt

    # """ COMMENT THIS LINE FOR FILTERED OPTIMIZATION
    from packingtools import package_params
    from packingtools import unpackage_params as _unpackage_params

    param_shapes = []
    for param in init_params:
        tmp = ()
        for p in param:
            tmp +=(p.shape,)
        param_shapes.append(tmp)

    unpackage_params = lambda d:_unpackage_params(d, shapes=param_shapes)
    
    init_params_pack = package_params(init_params)
    objective_grad_pack = grad(lambda d,i:objective(unpackage_params(d), i))
    print_perf_pack = lambda d, i , g: print_perf(unpackage_params(d), i , None)
    
    step_size, gamma, eps = (lambda t:0.001, lambda t:0.9, 1e-8)

    print('num iters:',num_batches * num_epochs)

    s = dt.datetime.now()
    optimized_params_pack = distributed_opt(objective_grad_pack, init_params_pack, 
        callback = print_perf_pack, num_iters = num_epochs * num_batches , method='RMSProp',
        opt_params=(step_size, gamma, eps))
    
    print('done in ', dt.datetime.now()- s)

    optimized_params = unpackage_params(init_params_pack)
    print_perf(optimized_params, num_batches, None)
   # """

    """ COMMENT THIS LINE FOR REGULAR RMSPROP
    s = dt.datetime.now()
    optimized_params = rmsprop(objective_grad, init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)
    print('done in ', dt.datetime.now()- s)


    # """
    # import pickle
    # pickle.dump(hist, open('filt_hist.pkl', 'wb'))
    # pickle.dump(hist, open('reg_hist.pkl', 'wb'))

    # pickle.dump(hist, open('filt_hist_smallbatch.pkl', 'wb'))
    # pickle.dump(hist, open('reg_hist_smallbatch.pkl', 'wb'))

