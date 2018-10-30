import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian
from autograd.core import getval

from dist_optimization import distributed_opt
from optimization import rmsprop

import datetime as dt
import math

def test1():
	def obj(d, i):
		v = np.hstack(list(d.values()))
		v -= np.arange(len(v))
		return np.sum(v**2)

	g = grad(obj)

	dim = 250
	max_size = 50
	for dim in [1, 10, 100, 250]:
		d = {'0':np.random.randn(dim) * 5.0 }

		if dim > max_size:
			num_splits, remainder= math.floor(dim / max_size), dim % max_size
			d = {str(i):np.random.randn(max_size) *5.0 for i in range(num_splits)}
			if remainder > 0:
				d[str(num_splits)] = np.random.randn(remainder) * 5.0
			# dtemp = {str()}

			print(sum([len(d[name]) for name in d ]))

		s=dt.datetime.now()
		step_size, gamma, eps = (lambda t:0.05, lambda t:0.9, 1e-8)

		d = distributed_opt(g, d, num_iters = 500, method='RMSProp',
			opt_params=(step_size, gamma, eps)
			)

		print('f(d)=',obj(d, 0))
		print('d[0][:10]', d['0'][:10] )
		print('done in', dt.datetime.now()-s)
		print()


def test2():

	def flatten(x):
		return x.ravel()

	#might be nice to thinly wrap these objects in a class that remembers its shape...
	def unflatten(x, shape=(2,2)):
		return x.reshape(*shape)

	def split(x, split_size=50):
		dim=len(x)
		num_splits, remainder= math.floor(dim / split_size), dim % split_size
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

	def unpack(d, shape=(2,2)):
		# Xflat = np.hstack(sorted(d).values()))
		Xflat = np.hstack([d[k] for k in sorted(d)])
		return Xflat.reshape(*shape)


	# A=np.arange(9).reshape(3,3)
	A=np.random.randn(100,100)
	Apack=pack(A)
	Anew=unpack(Apack, A.shape)
	# print(Apack)
	# print(Anew)
	Abool= A== unpack(pack(A), shape=A.shape)
	print(False in Abool)

def test3():

	def flatten(x):
		return x.ravel()

	#might be nice to thinly wrap these objects in a class that remembers its shape...
	def unflatten(x, shape=(2,2)):
		return x.reshape(*shape)

	def split(x, split_size=50):
		dim=len(x)
		num_splits, remainder= math.floor(dim / split_size), dim % split_size
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

	def unpack(d, shape=(2,2)):
		# Xflat = np.hstack(sorted(d).values()))
		Xflat = np.hstack([d[k] for k in sorted(d)])
		return Xflat.reshape(*shape)

	def obj(X):
		# return np.sum(np.dot(X - np.eye(X.shape[0]), X))
		# Y = X-np.eye(X.shape[0])
		# return np.sum(np.dot(X, X.T))
		# return np.sum(np.dot(Y, Y.T))
		return np.sum(X**2)

	dim=10
	# A = np.random.randn(100,100)
	A = np.random.randn(dim,dim)
	Apack = pack(A)
	gpack = grad(lambda d,i:obj(unpack(d, shape=A.shape)))
	# print(type(Apack))
	step_size, gamma, eps = (lambda t:0.01, lambda t:0.9, 1e-8)
	Apack_opt = distributed_opt(gpack, Apack, num_iters = 500, method='RMSProp',
			opt_params=(step_size, gamma, eps))

	Aopt = unpack(Apack_opt, shape=A.shape)

	print(Aopt)

if __name__=='__main__':
	test1()
	# test2()
	# test3()




	