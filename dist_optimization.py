
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian
from autograd.core import getval

from kalman_filter import Kalman

from copy import copy

"""
	This provides persistent optimizer objects, and a wrapper for doing
	distributed optimization
"""

class KGD:
	def __init__(self, x0, dx0, sigma_Q=0.01, sigma_R=2.0):
		#setup filter:
		self._N=len(x0)
		N=self._N

		z_0=np.hstack([x0, dx0]).reshape(2*N,1)

		Q=sigma_Q*np.eye(2*N)#dynamics noise
		R=sigma_R*np.eye(N)#observation noise
		C=np.hstack([np.zeros((N,N)), np.eye(N)])

		self.kf=Kalman(init_state=z_0,
				init_cov=np.eye(2*N)*0.01,
				C_t=C, Q_t=Q, R_t=R
			)

	def step(self, x, dx, alpha=0.01, callback=None):
		N=self._N

		A_t=np.block(
			[[np.eye(N) , -alpha*np.eye(N)],
			 # [np.zeros((N,N)) , 1-alpha*ddx]]
			 [np.zeros((N,N)) , np.eye(N)]]
		)

		z_hat = self.kf.filter(dx.T.reshape(N,1), A_t=A_t)
		x_hat, dx_hat=z_hat[:N], z_hat[N:]
		
		if callback: callback(x, t, dx_hat.reshape(x.shape))

		# x+=-alpha*dx_hat.flatten()
		return x - alpha*dx_hat.flatten()


class KRMSProp:
	"""
	A stateful version of Kalman RMSProp from https://arxiv.org/pdf/1810.12273.pdf
	which is used in distributed Kalman RMSProp.
	"""
	def __init__(self, x0, dx0, sigma_Q=0.01, sigma_R=2.0):
		self._N=len(x0)
		N=self._N

		r=np.ones(len(x0))#should be ones.
		z_0=np.hstack([r, x0, dx0]).reshape(3*N,1)

		Q=sigma_Q * np.eye(3*N)#dynamics noise
		R=sigma_R*np.eye(N)#observation noise
		C=np.hstack([np.zeros((N,2*N)), np.eye(N)])

		self.kf=Kalman(init_state=z_0,
				init_cov=np.eye(3*N)*0.01,
				C_t=C, Q_t=Q, R_t=R
			)

		self._r = r

	def step(self, x, dx, callback=None, 
		alpha=0.01, gamma=0.9, eps=1e-8):

		r = self._r
		N = self._N

		beta_t= alpha / np.sqrt((gamma*r + (1-gamma)*(dx**2) + eps ))

		A_t=np.block(
			[[np.eye(N) * gamma , np.zeros((N,N)),  (1-gamma)*np.diag(dx)],
			 [np.zeros((N,N)), np.eye(N), -beta_t * np.eye(N)],
			 [np.zeros((N,N)) , np.zeros((N,N)), np.eye(N)]]
		)

		z_hat=self.kf.filter(dx.T.reshape(N,1), A_t=A_t)
		r_hat, x_hat, dx_hat=z_hat[:N], z_hat[N:2*N], z_hat[2*N:]
		
		r = gamma * r + (1-gamma)* dx_hat.flatten()**2

		self._r = r


		return x - alpha/(np.sqrt(r) + eps) * dx_hat.flatten()

def distributed_opt(vectordict_g, vectordict_x, callback=None, 
	sigma_Q=0.01, sigma_R=2.0, method ='RMSProp', num_iters = 1000,
	opt_params=(lambda t:0.01, lambda t:0.9, 1e-8)):

	"""
	This wraps the distributed version of KGD from https://arxiv.org/pdf/1810.12273.pdf.
	Only Kalman RMSProp is available at present.
	
	vectordict_x is a dictionary of {name:flat_vector}
	vectordict_g is a function which takes a vectordict_x-like object and returns
		the gradient of a function w.r.t. this vectordict. This can be available through
		Autograd or assembled outside of this function.
	"""
	
	if method == 'RMSProp':
		_alpha, _gamma, _eps = opt_params
	else:
		# could add a dictionary of methods:persisitent optimizers
		raise NotImplementedError('Only RMSProp is supported right now')


	#initial gradient
	# vectordict_dx = vectordict_g(vectordict_x)
	vectordict_dx = vectordict_g(vectordict_x, 0)

	#setup optimizers
	_optimizers = {}
	for name in vectordict_x:
		_optimizers[name] = KRMSProp(vectordict_x[name], vectordict_dx[name],
			sigma_Q=sigma_Q, sigma_R=sigma_R)

	#do the optimization
	for t in range(num_iters):
		# vectordict_dx = vectordict_g(vectordict_x)
		vectordict_dx = vectordict_g(vectordict_x, t)
		for name in vectordict_x:
			vectordict_x[name] = _optimizers[name].step(
				vectordict_x[name], 
				vectordict_dx[name],
				callback=callback,
				alpha=_alpha(t),
				gamma=_gamma(t),
				eps=_eps
			)

		#since the gradients are all distributed, no gradients are provided to callback
		if callback: callback(vectordict_x, t, None)

	return vectordict_x
