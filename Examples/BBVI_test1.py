import matplotlib.pyplot as plt
from copy import copy

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

import pickle

from BBVI_opt import BaseBBVIModel

"""
This implements the example in "Black-Box Stochastic Variational Inference
in Five Lines of Python" By D. Duvenaud and R. Adams 
using the Kalman RMSProp algorithm from https://arxiv.org/pdf/1810.12273.pdf.
"""

np.random.seed(0)

class TestModel1(BaseBBVIModel):
	def __init__(self, D=2):
		self.dim=D
		plt.show(block=False)
		self.fig, self.ax=plt.subplots(2)
		self.elbo_hist=[]

		BaseBBVIModel.__init__(self)

	# specify the variational approximator
	def unpack_params(self, params):
		return params[:2], params[2:]

	def log_var_approx(self, z, params):
		mu, log_sigma=self.unpack_params(params)
		sigma=np.diag(np.exp(2*log_sigma))+1e-6
		return mvn.logpdf(z, mu, sigma)

	def sample_var_approx(self, params, n_samples=2000):
		mu, log_sigma=self.unpack_params(params)
		return npr.randn(n_samples, mu.shape[0])*np.exp(log_sigma)+mu

	# specify the distribution to be approximated
	def log_prob(self, z):
		mu, log_sigma = z[:, 0], z[:, 1]#this is a vectorized extraction of mu,sigma
		sigma_density = norm.logpdf(log_sigma, 0, 1.35)
		mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))

		return sigma_density + mu_density

	def plot_isocontours(self, ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
		x = np.linspace(*xlimits, num=numticks)
		y = np.linspace(*ylimits, num=numticks)
		X, Y = np.meshgrid(x, y)
		zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
		Z = zs.reshape(X.shape) 
		# plt.contour(X, Y, Z)
		ax.contour(X, Y, Z)
		ax.set_yticks([])
		ax.set_xticks([])


	def callback(self, *args):
		self.elbo_hist.append(self._estimate_ELBO(args[0], 0))
		if args[1]%50==0:
			print(args[1])
			curr_params=args[0]
			for a in self.ax:
				a.cla()
			self.plot_isocontours(self.ax[0], lambda z:np.exp(self.log_prob(z)))
			self.plot_isocontours(self.ax[0], lambda z:np.exp(self.log_var_approx(z, curr_params)))
			self.ax[1].plot(self.elbo_hist)
			self.ax[1].set_title('elbo estimate='+str(round(self.elbo_hist[-1],4)))
			plt.pause(1.0/30.0)

			plt.draw()


def run_BBVI_test():
	init_params=np.array([[3.0, 1.0],
						  [3.0, -5.0]]).ravel()

	mod=TestModel1()

	alpha_t=lambda t:0.01 * 1.001**(-t)
	
	"""
	BBVI optimization is done here. The 'how' parameter must be set in BBVI_opt.py.
	"""
	var_params=mod.run_VI(copy(init_params),
		step_size=alpha_t,
		num_iters=1500,
		num_samples=1,
		how='reparam'
	)

	print('init params=')
	print(init_params)
	print('final params=')
	print(var_params)
	print('elbo=',mod._estimate_ELBO(var_params, 0))

	# ===================

	# pickle.dump(var_params, open('regular_params_s1.pkl', 'wb'))
	# pickle.dump(mod.elbo_hist, open('regular_hist_s1.pkl', 'wb'))

	# pickle.dump(var_params, open('filtered_params_s1.pkl', 'wb'))
	# pickle.dump(mod.elbo_hist, open('filtered_hist_s1.pkl', 'wb'))

	plt.show()

if __name__=='__main__':

	run_BBVI_test()