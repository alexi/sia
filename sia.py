import numpy as np
import math
from scipy.optimize import minimize
import cvxopt.modeling

def context_vector(age, gender):
	pass

class Bandit:

	def __init__(self, regularization_param=0.1, num_items=3, num_positions=2, len_contexts=3):
		self.K = num_items
		self.M = num_positions
		self.S = num_positions
		self.C = len_contexts
		self.D = len_contexts + num_positions * 2 + num_items
		self.lamda = regularization_param
		self.m = cvxopt.matrix(np.zeros(self.D))
		self.q = cvxopt.matrix(np.full(self.D, regularization_param))

	def sample_posterior(self):
		w = np.zeros(self.D)
		for j in xrange(self.D):
			print("q:", self.q[j])
			w[j] = np.random.normal(self.m[j], 1/math.sqrt(self.q[j]))
		return w

	def expected_values(self, context):
		e = np.random.rand(self.K,self.M)
		return e

	def optimizer(self, expected_values):
		f = []
		for j in range(self.M):
			f.append(cvxopt.modeling.variable(self.K, "position_%d" % j))
		get_column = lambda i: cvxopt.matrix(expected_values[:, i])
		constraints = []
		for vec in f:
			constraints.append(vec >= 0)
			constraints.append(vec <= 1)
			constraints.append(sum(vec) <= 1)
		constraints.append(sum(sum(x) for x in f) == self.S)
		for k in xrange(self.K):
			constraints.append(sum([f[m][k] for m in xrange(len(f))]) <= 1)
		op = cvxopt.modeling.op(sum(cvxopt.modeling.dot(f[i], get_column(i)) for i in xrange(len(f))), constraints)
	 	op.solve()
	 	return f


	def update_posterior(self, event, reward):
		print(len(event))
		
		def objective(w):
			expInside = -1 * reward * np.dot(w, event)
			term1 = -.5 * sum(self.q[i] * (w[i]-self.m[i]) * (w[i]-self.m[i]) for i in range(self.D))
			term2 = -math.log(1+math.exp(expInside))
			result = -(term1 + term2)
			return result

		x0 = np.zeros(len(event))
		res = minimize(objective, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

		print("res: ", res)

		sigmoid = lambda x: 1 / (1 + math.exp(-x))

		for j in range(self.D):
			self.m[j] = res.x[j]
			cw_std = sigmoid(np.dot(res.x, event))
			print("original: ", self.q[j])
			print("cw_std: ", cw_std)
			self.q[j] = self.q[j] + cw_std * (1-cw_std)*event[j]**2
			print("updated: ", self.q[j])
			if (self.q[j] < 0): raise Exception("AAAH")

	def get_items(self, context):
		w = self.sample_posterior()
		# select actions
		e = self.expected_values(context)
		f = self.optimizer(e)
		return f

	def handle_user_action(self, context, item, pos, reward):
		gamma = np.zeros(self.M)
		gamma[pos] = 1
		psi = np.zeros(self.K + self.M)
		psi[item] = 1
		psi[pos + self.K] = 1

		phi = np.zeros(self.C)
		phi[0] = context["age"]
		phi[1] = 1 if context["gender"] == "male" else 0
		phi[2] = 1 if context["gender"] == "female" else 0
		self.update_posterior(np.concatenate((phi,psi,gamma), axis=0), reward)
		# m = np.zeros(self.M)
		# Get position item indeces
		# for j in xrange

b = Bandit()
ctx = np.array([33,0])
items = b.get_items(ctx)

print([str(x.value) for x in items])
b.handle_user_action({"age": 55, "gender": "female"}, 0, 1, 1)
items = b.get_items(ctx)
print([str(x.value) for x in items])
# server