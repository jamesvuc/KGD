
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian
from autograd.core import getval

from kalman_filter import Kalman

from copy import copy
from matplotlib import pyplot as plt

"""
	This package provides KGD, KGD + Momentum, and Kalman RMSProp
	from the paper https://arxiv.org/pdf/1810.12273.pdf.
"""

"""
	------------------------------
	Interface methods.
"""
def gd(g, x, callback=None, num_iters=200, 
	step_size=0.1, how='regular',
	sigma_Q=0.01, sigma_R=2.0):
	"""
	Performs vanilla gradient descent with how='regular'
	or Kalman Gradient Descent using how='filtered'.
	"""

	if type(step_size) is type(0.1):
		alpha=lambda t:step_size
	else:
		alpha=step_size

	if how=='regular':
		return _gd_regular(g, x,
			callback=callback,
			num_iters=num_iters,
			alpha=alpha,
		)
	elif how=='filtered':
		return _gd_filtered(g, x,
			callback=callback,
			num_iters=num_iters,
			alpha=alpha,
			sigma_Q=sigma_Q,
			sigma_R=sigma_R
		)
	else:
		raise ValueError("Allowable hows are 'regular' and 'filtered'.")



def momgd(g, x, callback=None, num_iters=200, 
	step_size=0.1, mass=0.9, how='regular',
	sigma_Q=0.01, sigma_R=2.0):
	"""
	Performs gradient descent with momentum using how='regular'
	or Kalman gradient descent with momentum using how='filtered'.
	"""
	if type(step_size) is type(0.1):
		alpha=lambda t:step_size
	else:
		alpha=step_size

	if type(mass) is type(0.1):
		mu=lambda t:mass
	else:
		mu=mass

	if how=='regular':
		return _momgd_regular(g, x,
			callback=callback,
			num_iters=num_iters,
			alpha=alpha,
			mu=mu
		)

	elif how=='filtered':
		return _momgd_filtered(g, x,
			callback=callback,
			num_iters=num_iters,
			alpha=alpha,
			mu=mu,
			sigma_Q=sigma_Q,
			sigma_R=sigma_R
		)
	else:
		raise ValueError("Allowable hows are 'regular' and 'filtered'.")

def rmsprop(g, x, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8, how='regular',
        	sigma_Q=0.01, sigma_R=2.0):
	
	"""
	Performs RMSProp using how='regular'
	or Kalman RMSProp using how='filtered'.
	"""

	if type(step_size) is type(0.1):
		alpha=lambda t:step_size
	else:
		alpha=step_size

	if type(gamma) is type(0.1):
		gamma_=lambda t:gamma
	else:
		gamma_=gamma

	if how=='regular':
		return _rmsprop_regular(g, x,
			callback=callback,
			num_iters=num_iters,
			alpha=alpha,
			gamma=gamma_,
			eps=eps
		)

	elif how=='filtered':
		return _rmsprop_filtered(g, x,
			callback=callback,
			num_iters=num_iters,
			alpha=alpha,
			gamma=gamma_,
			eps=eps,
			sigma_Q=sigma_Q,
			sigma_R=sigma_R
		)
	else:
		raise ValueError("Allowable hows are 'regular' and 'filtered'.")

"""
	------------------------------------
	Benchmark unfiltered optimization algorithms
"""

def _gd_regular(g, x, callback=None, num_iters=200, 
	alpha=lambda t:0.1):
	for t in range(num_iters):
		dx=g(x)
		if callback: callback(x, t, dx)
		x+=-alpha(t)*dx

	return x

def _momgd_regular(g, x, callback=None, num_iters=200, 
	alpha=lambda t:0.1, mu=lambda t:0.9):
	v=np.zeros(len(x))
	for t in range(num_iters):
		dx=g(x)
		if callback: callback(x, t, dx)
		v=mu(t) * v - (1-mu(t))* dx
		x+= alpha(t) * v

	return x


def _rmsprop_regular(g, x, callback=None, num_iters=200, 
	alpha=lambda t:0.1, gamma=lambda t:0.9, eps=1e-8):

	r=np.ones(len(x))
	for t in range(num_iters):
		dx=g(x)
		if callback: callback(x, t, dx)
		r = gamma(t) * r + (1-gamma(t))* dx**2
		x += - alpha(t)/(np.sqrt(r)+ eps) * dx 

	return x

"""
	--------------------------------------
	Filtered optimization algorithms.
"""

def _gd_filtered(g, x, callback=None, num_iters=200, 
	alpha=lambda t:0.1, 
	sigma_Q=0.01, sigma_R=2.0):
	"""
	Implements the Kalman Gradient Descent algorithm from 
	https://arxiv.org/pdf/1810.12273.pdf.
	"""

	# Initialize the Kalman filter
	N=len(x)

	dx=g(x)
	z_0=np.hstack([x, dx]).reshape(2*N,1)

	Q=sigma_Q*np.eye(2*N)#dynamics noise
	R=sigma_R*np.eye(N)#observation noise
	C=np.hstack([np.zeros((N,N)), np.eye(N)])

	kf=Kalman(init_state=z_0,
			init_cov=np.eye(2*N)*0.01,
			C_t=C, Q_t=Q, R_t=R
		)

	for t in range(num_iters):
		# evaluate the gradient
		dx=g(x)

		#construct transition matrix
		A_t=np.block(
			[[np.eye(N) , -alpha(t)*np.eye(N)],
			 [np.zeros((N,N)) , np.eye(N)]]
		)

		#increment the kalman filter
		z_hat=kf.filter(dx.T.reshape(N,1), A_t=A_t)
		x_hat, dx_hat=z_hat[:N], z_hat[N:]
		
		if callback: callback(x, t, dx_hat.reshape(x.shape))

		#perform the KGD update
		x+=-alpha(t)*dx_hat.flatten()

	return x


def _momgd_filtered(g, x, callback=None, num_iters=200, 
	alpha=lambda t:0.1, mu=lambda t:0.9, 
	sigma_Q=0.01, sigma_R=2.0):
	"""
	Implements the Kalman Gradient Descent + momentum algorithm from 
	https://arxiv.org/pdf/1810.12273.pdf.
	"""

	# Initialize the Kalman filter
	N=len(x)

	dx=g(x)
	v=np.zeros(len(x))
	z_0=np.hstack([v, x, dx]).reshape(3*N,1)

	Q=sigma_Q * np.eye(3*N)#dynamics noise
	R=sigma_R*np.eye(N)#observation noise
	C=np.hstack([np.zeros((N,2*N)), np.eye(N)])

	kf=Kalman(init_state=z_0,
			init_cov=np.eye(3*N)*0.01,
			C_t=C, Q_t=Q, R_t=R
		)

	for t in range(num_iters):
		# evaluate the gradient
		dx=g(x)

		#construct transition matrix
		A_t=np.block(
			[[np.eye(N) * mu(t) , np.zeros((N,N)),  (1-mu(t))*np.eye(N)],
			 [-alpha(t)* np.eye(N), np.eye(N), -alpha(t)*(1-mu(t)) * np.eye(N)],
			 [np.zeros((N,N)) , np.zeros((N,N)), np.eye(N)]]
		)

		#increment the kalman filter
		z_hat=kf.filter(dx.T.reshape(N,1), A_t=A_t)
		v_hat, x_hat, dx_hat=z_hat[:N], z_hat[N:2*N], z_hat[2*N:]
		
		if callback: callback(x, t, dx_hat.reshape(x.shape))

		#perform the KGD + momentum update
		v=mu(t) * v - (1-mu(t))* dx_hat.flatten()#don't flatte, use .reshpape(x.shape)
		x+= alpha(t) * v

	return x


def _rmsprop_filtered(g, x, callback=None, num_iters=200, 
	alpha=lambda t:0.1, gamma=lambda t:0.9, eps=1e-8, 
	sigma_Q=0.01, sigma_R=2.0):
	"""
	Implements the Kalman RMSProp algorithm from 
	https://arxiv.org/pdf/1810.12273.pdf.
	"""

	# Initialize the Kalman filter
	N=len(x)

	dx=g(x)
	r=np.ones(len(x))#should be ones.
	z_0=np.hstack([r, x, dx]).reshape(3*N,1)

	Q=sigma_Q * np.eye(3*N)#dynamics noise
	R=sigma_R*np.eye(N)#observation noise
	C=np.hstack([np.zeros((N,2*N)), np.eye(N)])

	kf=Kalman(init_state=z_0,
			init_cov=np.eye(3*N)*0.01,
			C_t=C, Q_t=Q, R_t=R
		)

	for t in range(num_iters):
		# evaluate the gradient
		dx=g(x)

		#construct transition matrix
		beta_t= alpha(t) / np.sqrt((gamma(t)*r + (1-gamma(t))*(dx**2))+ eps)
		
		A_t=np.block(
			[[np.eye(N) * gamma(t) , np.zeros((N,N)),  (1-gamma(t))*np.diag(dx)],
			 [np.zeros((N,N)), np.eye(N), -beta_t * np.eye(N)],
			 [np.zeros((N,N)) , np.zeros((N,N)), np.eye(N)]]
		)

		#increment the kalman filter
		z_hat=kf.filter(dx.T.reshape(N,1), A_t=A_t)
		r_hat, x_hat, dx_hat=z_hat[:N], z_hat[N:2*N], z_hat[2*N:]
		
		#perform the Kalman RMSProp update
		r = gamma(t) * r + (1-gamma(t))* dx_hat.flatten()**2
		x += - alpha(t)/(np.sqrt(r)+ eps) * dx_hat.flatten()

		if callback: callback(x, t, dx_hat.reshape(x.shape))

	return x




if __name__=='__main__':
	pass



