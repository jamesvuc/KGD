import numpy as np
from numpy.linalg import inv

class Kalman:

	def __init__(self, init_state=None, init_cov=None, A_t=None, C_t=None, Q_t=None, R_t=None):
		self.state_dim=len(init_state)

		self.A_realTime=False
		self.C_realTime=False
		self.Q_realTime=False
		self.R_realTime=False

		self.A_t=A_t
		self.C_t=C_t
		self.Q_t=Q_t
		self.R_t=R_t


		if self.A_t is None:
			self.A_realTime=True

		if self.C_t is None:
			self.C_realTime=True

		if self.Q_t is None:
			raise Exception('Realtime covariances are not supported yet')
			self.Q_realTime=True

		if self.R_t is None:
			raise Exception('Realtime covariances are not supported yet')
			self.R_realTime=True

		# should check for consistency of the dimensions


		# initialize persistent variables.
		self.x_hat_t=init_state
		self.P_t=init_cov


	def increment_filter(self, y_t, A_t=None, C_t=None):
		#handle realtime vs. not real-time in filter funciton
		

		#prediction
		self.x_hat_t=np.dot(A_t, self.x_hat_t)
		self.P_t=np.dot(A_t, np.dot(self.P_t, A_t.T))+self.Q_t

		#calculate
		z_t=y_t-np.dot(C_t, self.x_hat_t)
		# y_t should be a matrix
		#can you combine these steps to speed up?
		S_t=self.R_t+np.dot(C_t, np.dot(self.P_t, C_t.T))

		K_t=np.dot(self.P_t, np.dot(C_t.T, inv(S_t)))

		#update
		self.x_hat_t=self.x_hat_t+np.dot(K_t, z_t)
		self.P_t=np.dot((np.eye(self.state_dim)-np.dot(K_t, C_t)), self.P_t)


	def filter(self, y_t, A_t=None, C_t=None):
		if self.A_realTime:
			A_temp=A_t
		else:
			A_temp=self.A_t


		if self.C_realTime:
			C_temp=C_t
		else:
			C_temp=self.C_t

		self.increment_filter(y_t, A_t=A_temp, C_t=C_temp)

		return self.x_hat_t



"""
Model is:

x_{t+1}=A_t * x_t + w_t
  y_t=C_t * x_t + u_t

  w_t~N(0, Q_t)
  u_t~N(0, R_t)

"""

