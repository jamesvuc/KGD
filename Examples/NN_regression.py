from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc import flatten
from autograd.core import getval

import numpy as npx

"""
	This demonstrates the use of Kalman RMSProp (https://arxiv.org/pdf/1810.12273.pdf)
	on a MLP regression problem. 

	The neural network script below is heavily based on the example at 
	https://github.com/HIPS/autograd/blob/master/examples/neural_net_regression.py.

"""

# from autograd.misc.optimizers import adam, rmsprop
import sys
sys.path.append('../')
from optimization import rmsprop

npx.random.seed(0)

def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
	"""These functions implement a standard multi-layer perceptron,
	vectorized over both training examples and weight samples."""
	shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
	num_weights = sum((m+1)*n for m, n in shapes)

	#don't need vectorized weights (only one sample...)
	def unpack_layers(weights):
		num_weight_sets = len(weights)
		for m, n in shapes:
			# yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
			#       weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
			yield (
					weights[:m*n]     .reshape((m, n)),
					# weights[m*n:m*n+n].reshape((num_weight_sets, 1, n))
					weights[m*n:m*n+n].reshape((1, n))
				)
			# weights = weights[:, (m+1)*n:]
			weights = weights[(m+1)*n:]

	def predictions(weights, inputs):
		# inputs = np.expand_dims(inputs, 0)
		for W, b in unpack_layers(weights):
			outputs = np.einsum('nd,do->no', inputs, W) + b
			inputs = nonlinearity(outputs)
		return outputs

	def logprob(weights, inputs, targets):
		# log_prior = -L2_reg * np.sum(weights**2, axis=1)
		log_prior = -L2_reg * np.sum(weights**2)
		preds = predictions(weights, inputs)
		# log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
		log_lik = -np.sum((preds - targets)**2) / noise_variance
		return log_prior + log_lik

	return num_weights, predictions, logprob

def build_toy_dataset(n_data=80, noise_std=0.1):
	rs = npr.RandomState(0)
	inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
							  np.linspace(6, 8, num=n_data/2)])
	targets = np.cos(inputs) + rs.randn(n_data) * noise_std
	inputs = (inputs - 4.0) / 2.0
	inputs  = inputs[:, np.newaxis]
	targets = targets[:, np.newaxis] / 2.0
	return inputs, targets


if __name__ == '__main__':

	init_scale = 0.1
	weight_prior_variance = 1.0
	# init_params = init_random_params(init_scale, layer_sizes=[1, 4, 4, 1])
	rbf = lambda x: np.exp(-x**2)#deep basis function model
	num_weights, predictions, logprob = \
		make_nn_funs(layer_sizes=[1, 4, 4, 1], L2_reg=weight_prior_variance,
						noise_variance=0.01)

	init_params=np.random.randn(num_weights)*init_scale

	nn_predict=predictions


	inputs, targets = build_toy_dataset()

	def generate_minibatch(inputs, targets, batch_size=32):
		idxs=npx.random.choice(len(inputs), size=batch_size, replace=False)
		return inputs[idxs], targets[idxs]

	def objective(weights, t):
		inputs_batch, targets_batch=generate_minibatch(inputs, targets, batch_size=8)
		return (
			-logprob(weights, inputs_batch, targets_batch)
			)

	print(grad(objective)(init_params, 0))


	# Set up figure.
	f, axes=plt.subplots(2)
	plt.show(block=False)
	traj=[]
	def callback(params, t, g):
		if t % 50 == 0:
			print("Iteration {} log likelihood {}".format(t, -objective(params, t)))
			print('Stepsize=',0.01 * 1.001**(-t))

		traj.append(objective(params, 0))
		for ax in axes:
			ax.cla()
		axes[0].plot(inputs.ravel(), targets.ravel(), 'bx', ms=12)
		plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1))
		outputs = nn_predict(params, plot_inputs)
		axes[0].plot(plot_inputs, outputs.reshape(plot_inputs.shape), 'r', lw=3)
		axes[0].set_ylim([-1, 1])

		axes[1].plot(traj)

		plt.draw()
		plt.pause(1.0/60.0)

	print("Optimizing network parameters...")



	# optimized_params = adam(grad(objective), init_params,
	# 						step_size=0.01, num_iters=1000, callback=callback)

	alpha_t=lambda t:0.01 * 1.001**(-t)
	optimized_params = rmsprop(grad(lambda w:objective(w, 0)), init_params,
							# step_size=0.01,
							step_size=alpha_t,
							gamma=0.9,
							num_iters=1000,
							callback=callback,
							# how='regular')
							how='filtered')

	# import pickle
	# pickle.dump(traj,open('reg_NN_small_traj.pkl', 'wb'))
	# pickle.dump(optimized_params,open('reg_NN_small_params.pkl', 'wb'))

	# pickle.dump(traj,open('filt_NN_small_traj.pkl', 'wb'))
	# pickle.dump(optimized_params,open('filt_NN_small_params.pkl', 'wb'))


	"""
	With very small batch size (e.g. 8), the filtering makes a huge difference 
	in convergence speed (3x!)

	"""


