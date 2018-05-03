import cvxopt
import cvxopt.modeling
import numpy as np
import math
from scipy.optimize import minimize
import sys

# Pick the right range to use (Python 3 range = Python 2 xrange)
range = range if sys.version_info >= (3, 0) else xrange

class Bandit:
    def __init__(self, regularization_param=0.1, num_items=3, num_positions=2, len_contexts=3):
        self.K = num_items
        self.M = num_positions
        self.S = num_positions
        self.C = len_contexts
        self.D = len_contexts + num_positions * 2 + num_items
        self.m = np.zeros(self.D)
        self.q = np.full(self.D, regularization_param)

    def handle_user_action(self, context, item, pos, reward):
        # n-hot vector saying which item(s) were picked (currently just one)
        gamma = np.zeros(self.M)
        gamma[pos] = 1

        # Two one-hot vectors stuck together, encoding putting "item" at "pos"
        # TODO(tom): I think this might be wrong, it seems like we should be putting *all* the
        # items in from the superarm that the user saw
        phi = np.zeros(self.K + self.M)
        phi[item] = 1
        phi[pos + self.K] = 1

        # Context vector
        psi = np.zeros(self.C)
        psi[0] = context["age"]
        psi[1] = 1 if context["gender"] == "male" else 0
        psi[2] = 1 if context["gender"] == "female" else 0

        (new_m, new_q) = self.update_posterior(np.concatenate((phi, psi, gamma)), reward, self.m, self.q)

        # Update the m and q vectors
        self.m = new_m
        self.q = new_q

    def get_items(self, context):
        w = self.sample_posterior(self.m, self.q)
        # select actions
        e = self.expected_values(self.K, self.M, context)
        f = self.compute_superarm(e, self.S)
        return f

    @staticmethod
    def sample_posterior(m, q):
        """Return a sample of the parameters (a vector of independent Gaussians with means m and precisions q)"""
        return [np.random.normal(m[j], 1/math.sqrt(q[j])) for j in range(len(m))]

    @staticmethod
    def expected_values(K, M, context):
        e = np.random.rand(K, M)
        # TODO(tom): we still need to fill this in. The expected values need to be
        # \sigma(\Psi . w), where capital Psi is the big concatenated vector we make in
        # handle_user_action. We should probably extract a helper method to generate the big vector
        # given a set of actions and context
        return e

    @staticmethod
    def compute_superarm(expected_values, S):
        """Given a matrix of expected rewards, solve the LP to get the best super-arm with S items"""
        (K, M) = expected_values.shape

        # Variables
        f = [cvxopt.modeling.variable(K, "position_%d" % j) for j in range(M)]

        # Objective
        objective = sum(cvxopt.modeling.dot(f[i], cvxopt.matrix(expected_values[:, i])) for i in range(len(f)))

        # Constraints
        constraints = []
        for vec in f: constraints.append(vec >= 0)
        for vec in f: constraints.append(vec <= 1)
        constraints.append(sum(sum(x) for x in f) == S)
        for vec in f: constraints.append(sum(vec) <= 1) # Column sums don't exceed 1
        for k in range(K): constraints.append(sum([f[m][k] for m in range(len(f))]) <= 1) # Row sums don't exceed 1

        cvxopt.modeling.op(objective, constraints).solve()

        return f

    @staticmethod
    def update_posterior(event, reward, m, q):
        """Run a solver to find an updated m and q given the event and reward."""
        def objective(w):
            term1 = -0.5 * sum(q[i] * (w[i]-m[i]) * (w[i]-m[i]) for i in range(len(event)))
            term2 = -math.log(1+math.exp(-1 * reward * np.dot(w, event)))
            result = -(term1 + term2)
            return result

        x0 = np.zeros(len(event))
        res = minimize(objective, x0, method='nelder-mead', options={'xtol': 1e-6, 'disp': False})

        prob = 1 / (1 + math.exp(-np.dot(res.x, event)))
        new_q = q + prob * (1 - prob) * (event**2)

        return (res.x, new_q)


b = Bandit()
ctx = np.array([33,0])
items = b.get_items(ctx)

print([str(x.value) for x in items])
b.handle_user_action({"age": 55, "gender": "female"}, 0, 1, 1)
items = b.get_items(ctx)
print([str(x.value) for x in items])
# server
