import autograd.numpy as np

from collections import OrderedDict

import math

"""
	The unpack and pack could be built into tensorwise_opt
"""


def flatten(x):
	return x.ravel()

#might be nice to thinly wrap these objects in a class that remembers its shape...
def unflatten(x, shape=(2,2)):
	return x.reshape(*shape)

def split(x, split_size=50):
	dim=len(x)
	num_splits, remainder= math.floor(dim / split_size), dim % split_size
	# d = OrderedDict({i:x[i*split_size : (i+1)*split_size] for i in range(num_splits)})
	d = {i:x[i*split_size : (i+1)*split_size] for i in range(num_splits)}
	if remainder > 0:
		d[num_splits] = x[num_splits*split_size :]

	return d

def pack(X):
	#step 1: flatten
	Xflat=flatten(X)

	#step2: split
	max_size=50
	if len(Xflat) > max_size:
		d = split(Xflat)
	else:
		d = {0:Xflat}

	return d

def unpack(d):
	# return np.hstack(d.values())
	return np.hstack([d[k] for k in sorted(d)])

def package_params(params):
	#step 1: convert to a single vector
	flat_params = []
	for W,b in params:
		flat_params.append(flatten(W))
		flat_params.append(flatten(b))

	flat_params = np.hstack(flat_params)

	#step 2: use pack
	d = pack(flat_params)

	return d

def unpackage_params(d, shapes=[]):
	flat_params = unpack(d)

	params = []
	idx = 0
	for W_shape, b_shape in shapes:

		W_len = np.product(W_shape)
		b_len = np.product(b_shape)

		W = (flat_params[idx : idx + W_len]).reshape(*W_shape)
		idx += W_len

		b = (flat_params[idx : idx + b_len]).reshape(*b_shape)
		idx += b_len

		params.append((W, b))

	return params



