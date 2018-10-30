import autograd.numpy as np
from autograd import grad

from copy import copy
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from optimization import gd, momgd, rmsprop

np.random.seed(1)

def main():
	d=2
	w=np.array([1.0,2.0])
	f = lambda x: 0.1 * np.sum(x**2) + np.sin(np.dot(x,w))
	
	sigma=1.0
	g=grad(f)
	#uses different noise for different runs.
	grad_f=lambda x: g(x) + sigma*np.random.randn(d)

	x_0=np.array([10.0, 8.0])

	hist={
		'x':[],
		'f':[],
		'dx':[],
		'x_noise':[],
		'f_noise':[],
		'dx_noise':[],
		'x_kf':[],
		'f_kf':[],
		'dx_kf':[],
	}

	"""
	want to test 3 versions:
		1. Noiseless (ideal case)
		2. Noisy, no filtering
		3. Noisy, filtering
	"""

	def logger(args, suffix=''):
		x, t, dx = args

		x_name, f_name, dx_name = 'x'+suffix, 'f'+suffix, 'dx'+suffix

		hist[x_name].append(copy(x))
		hist[f_name].append(f(x))
		hist[dx_name].append(np.sum(dx**2))

	"""
	===========
	Ordinary Gradient Descent
	-------
	"""


	""" True Gradient """
	_logger=lambda *x:logger(x, suffix='')
	# x_opt_reg=gd(g, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, how='regular')
	# x_opt_reg=momgd(g, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, mass=0.75, how='regular')
	x_opt_reg=rmsprop(g, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, gamma=0.99, how='regular')

	""" Noisy """
	_logger=lambda *x:logger(x, suffix='_noise')
	# x_opt_reg=gd(grad_f, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, how='regular')
	# x_opt_reg=momgd(grad_f, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, mass=0.75, how='regular')
	x_opt_reg=rmsprop(grad_f, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, gamma=0.99, how='regular')

	""" Filtered """
	_logger=lambda *x:logger(x, suffix='_kf')
	# x_opt_reg=gd(grad_f, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, how='filtered')
	# x_opt_reg=momgd(grad_f, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, mass=0.75, how='filtered')	
	x_opt_reg=rmsprop(grad_f, copy(x_0), callback=_logger, num_iters=500, step_size=0.1, gamma=0.99, how='filtered')	

	def do_plotting():
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		fig, axes=plt.subplots(2,2)

		#"""
		xs=np.arange(-10, 10, 0.05)
		ys=np.arange(-10, 10, 0.05)
		X, Y = np.meshgrid(xs, ys)
		Z=np.zeros((xs.shape[0], ys.shape[0]))
		for i,_x in enumerate(xs):
			for j,_y in enumerate(ys):
				Z[i,j]=f(np.array([_x, _y]))

		axes[0,0].contour(X,Y, Z, 20, alpha=0.5)

		axes[0,0].plot(np.array(hist['x'])[:,0],np.array(hist['x'])[:,1], alpha=0.75, color='blue')
		axes[0,0].plot(np.array(hist['x_noise'])[:,0],np.array(hist['x_noise'])[:,1], alpha=0.75, color='orange')
		axes[0,0].plot(np.array(hist['x_kf'])[:,0],np.array(hist['x_kf'])[:,1], alpha=0.75, color='green')


		axes[0,0].scatter(hist['x'][-1][0],hist['x'][-1][1], color='black')#'blue')
		axes[0,0].scatter(hist['x_noise'][-1][0],hist['x_noise'][-1][1], color='black')#'orange')
		axes[0,0].scatter(hist['x_kf'][-1][0],hist['x_kf'][-1][1], color='black')#'green')
		
		axes[0,0].legend(['noise-free', 'unfiltered', 'filtered'])
		axes[0,0].set_xlabel(r'$x^1$')
		axes[0,0].set_ylabel(r'$x^2$', rotation=0)

		axes[0,1].plot(np.array(hist['x']), label='noise-free', alpha=0.5)
		axes[0,1].plot(hist['x_noise'], label='unfiltered', alpha=0.5)
		axes[0,1].plot(hist['x_kf'], label='filtered', alpha=0.5)
		axes[0,1].set_title(r'$x_t$')
		axes[0,1].legend([
			r'$x^1_t$ noise-free',
			r'$x^2_t$ noise-free',
			r'$x^1_t$ unfiltered',
			r'$x^2_t$ unfiltered',
			r'$x^1_t$ filtered',
			r'$x^2_t$ filtered'])

		# axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

		axes[1,0].plot(hist['f'], label='noise-free', alpha=0.5)
		axes[1,0].plot(hist['f_noise'], label='unfiltered', alpha=0.5)
		axes[1,0].plot(hist['f_kf'], label='filtered', alpha=0.5)
		axes[1,0].set_title(r'$f(x_t)$')
		axes[1,0].set_xlabel(r'$t$')
		axes[1,0].legend()

		axes[1,1].plot(hist['dx'], label='noise-free', alpha=0.5)
		axes[1,1].plot(hist['dx_noise'], label='unfiltered', alpha=0.5)
		axes[1,1].plot(hist['dx_kf'], label='filtered', alpha=0.5)
		axes[1,1].set_title(r"$\|\nabla f(x_t)\|^2$")
		axes[1,1].set_xlabel(r'$t$')
		axes[1,1].legend()

		plt.subplots_adjust(
			left  = 0.10,  # the left side of the subplots of the figure
			bottom = 0.07,   # the bottom of the subplots of the figure
			right = 0.93,    # the right side of the subplots of the figure
			top = 0.93,      # the top of the subplots of the figure
			wspace = 0.10,   # the amount of width reserved for space between subplots,
			hspace = 0.25 	# expressed as a fraction of the average axis width
		)

		plt.show()

	do_plotting()


if __name__=='__main__':
	main()
