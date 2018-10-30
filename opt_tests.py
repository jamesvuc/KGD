import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian

from copy import copy
from matplotlib import pyplot as plt

from optimization import gd, momgd, rmsprop

np.random.seed(1)

def main():
	"""
	Simple unit-test for the gradient optimization algorithms in optimization.py
	"""
	d=2
	w=np.array([1.0,2.0])
	f = lambda x: 0.1 * np.sum(x**2) + np.sin(np.dot(x,w))
	
	sigma=1.0
	g=grad(f)
	grad_f=lambda x: g(x) + sigma*np.random.randn(d)

	x_0=np.array([10.0, 8.0])
	"""this just tests the functionality with clean gradients."""

	## gd
	x_opt_reg=gd(grad_f, copy(x_0), callback=None, num_iters=500, step_size=0.1, how='regular')
	x_opt_filt=gd(grad_f, copy(x_0), callback=None, num_iters=500, step_size=0.1, how='filtered')

	print('Regular GD:')
	print('x_opt_reg=',x_opt_reg, 'f(x_opt_reg)=',f(x_opt_reg))
	print('x_opt_filt=',x_opt_filt, 'f(x_opt_filt)=',f(x_opt_filt))
	print()

	#gd + momentum
	x_opt_reg=momgd(grad_f, copy(x_0), callback=None, num_iters=500, step_size=0.1, mass=0.95, how='regular')
	x_opt_filt=momgd(grad_f, copy(x_0), callback=None, num_iters=500, step_size=0.1, mass=0.95, how='filtered')

	print('Momentum GD')
	print('x_opt_reg=',x_opt_reg, 'f(x_opt_reg)=',f(x_opt_reg))
	print('x_opt_filt=',x_opt_filt, 'f(x_opt_filt)=',f(x_opt_filt))
	print()

	# rmsprop
	x_opt_reg=rmsprop(grad_f, copy(x_0), callback=None, num_iters=500, step_size=0.1, gamma=0.99, how='regular')
	x_opt_filt=rmsprop(grad_f, copy(x_0), callback=None, num_iters=500, step_size=0.1, gamma=0.99, how='filtered')

	print('RMSProp')
	print('x_opt_reg=',x_opt_reg, 'f(x_opt_reg)=',f(x_opt_reg))
	print('x_opt_filt=',x_opt_filt, 'f(x_opt_filt)=',f(x_opt_filt))
	print()


if __name__=='__main__':
	main()