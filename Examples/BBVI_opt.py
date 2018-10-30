# JV_BBVI.py

import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

from profilehooks import profile
import datetime as dt

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad, jacobian
from autograd.misc.optimizers import adam
from autograd.core import getval

import sys
sys.path.append('../')
from optimization import rmsprop

"""
	This implements a generic BBVI model with several options for gradient estimators.
	See https://github.com/jamesvuc/BBVI for details (this is a copy from that repo).
"""

# ======Some Helper Methods=====
#X,Y are random vectors, with each having shape dim x N (or N x dim) if rowvar is False
def vector_cov(X, Y, rowvar=True):
	if rowvar:
		d=X.shape[0]
	else:
		d=X.shape[1]
	_cov=np.cov(X,Y, rowvar=rowvar)

	return _cov[d:,d:]

def _resize_for_cov(X):
	return X.reshape(X.shape[0],X.shape[1]*X.shape[2])

def np_cov(m, rowvar=False):
    # Handles complex arrays too
    # m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError('m has more than 2 dimensions')
    dtype = np.result_type(m, np.float64)
    m = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and m.shape[0] != 1:
        m = m.T
    if m.shape[0] == 0:
        return np.array([], dtype=dtype).reshape(0, 0)

    # Determine the normalization
    fact = m.shape[1] - 1
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    m -= np.mean(m, axis=1, keepdims=1)
    # c = np.dot(m, m.T.conj())
    c = np.dot(m, m.T)
    c *= np.true_divide(1, fact)
    return c.squeeze()


# ======Model====
class BaseBBVIModel(metaclass=ABCMeta):
	def __init__(self):
		#setup for later
		self._score=None
		self._init_var_params=None
		self._var_params=None

		self.N_SAMPLES=None

	# =======User-specified methods=====
	# All signatures must be preserved exactly. Check this with a parse_args?

	# Variational approx
	@abstractmethod
	def unpack_params(self, params):
		pass

	@abstractmethod
	def log_var_approx(self, z, params):
		pass

	@abstractmethod
	def sample_var_approx(self, params, n_samples=1000):
		pass

	# Joint Distribution
	@abstractmethod
	def log_prob(self, z):
		pass

	def callback(self, *args):
		pass

	"""
	=======-Generic VI methods=======
	"""
	"""
	------Stochastic Search-------
	"""
	def _objfunc(self, params, t):
		samps=self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)

		return np.mean(self.log_var_approx(samps, params)*(self.log_prob(samps)-self.log_var_approx(samps, getval(params))))

	def _objfuncCV(self, params, t):
		samps=self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)

		a_hat=np.mean(self.log_prob(samps)-self.log_var_approx(samps, getval(params)))

		return np.mean(self.log_var_approx(samps, params)*(self.log_prob(samps)-self.log_var_approx(samps, getval(params))-a_hat))

	"""
	-----Reparameterization Trick--------
	"""
	def _estimate_ELBO(self, params, t):
		samps=self.sample_var_approx(params, n_samples=self.N_SAMPLES)

		# estimates E[log p(z)-log q(z)]
		return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, params), axis=0)#this one appears to be correct

	#use reduced-variance method
	# (https://papers.nips.cc/paper/7268-sticking-the-landing-simple-lower-variance-gradient-estimators-for-variational-inference.pdf)
	def _estimate_ELBO_noscore(self, params, t):
		samps=self.sample_var_approx(params, n_samples=self.N_SAMPLES)

		#eliminates the score function
		return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params)), axis=0)#this one appears to be correct


	def run_VI(self, init_params, num_samples=50, step_size=0.01, num_iters=2000, how='stochsearch'):
		hows=['stochsearch', 'reparam']
		if how not in hows:
			raise KeyError('Allowable VI methods are', hows)

		self.N_SAMPLES=num_samples

		if how == 'stochsearch':	
			#not CV
			_tmp_gradient=grad(self._objfunc)

			#CV
			# _tmp_gradient=grad(self._objfuncCV)
		elif how == 'reparam':
			_tmp_gradient=grad(self._estimate_ELBO)
			# _tmp_gradient=grad(self._estimate_ELBO_CV)
			# _tmp_gradient=grad(self._estimate_ELBO_noscore)

		else:
			raise Exception("I don't know what to do!")

		self._init_var_params=init_params

		s=dt.datetime.now()


		"""
		TOGGLE how='regular' or how='filtered' here to compare KGD optimization
		performance.
		"""
		self._var_params=rmsprop(lambda x:_tmp_gradient(x,0), self._init_var_params,
			step_size=step_size,
			num_iters=num_iters,
			callback=self.callback,
			# how='regular'
			how='filtered'
		)

		print('done in:',dt.datetime.now()-s)
		return self._var_params

if __name__=='__main__':
	from JV_BBVI_test import run_BBVI_test1

	run_BBVI_test()


