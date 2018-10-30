import autograd.numpy as np

from collections import OrderedDict

import math

"""
	The unpack/pack, split, flatten/unflatten could be built into tensorwise_opt
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
	"""
	X is a numpy tensor to be (possibly) split and 
		packaged for distributed optimization.
	Returns a dict of {i:subvector}.
	"""
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
	"""
	Unpacks a dict of {i:subjector} into a flattened tensor.
	"""
	return np.hstack([d[k] for k in sorted(d)])