import numpy as np

class PILCO:
    def __init__(self, plant, policy, cost, initial_random_rollouts=4):
        self.x = numpy.array([])
        self.x_cov = numpy.array([])
        self.u = numpy.array([])
        self.u_cov = numpy.array([])
        self.y = numpy.array([])
        self.L = numpy.array([])

        self.initial_random_rollouts = initial_random_rollouts

        self.cost = cost
        self.plant = plant
        self.policy = policy

        if ( self.policy.angi is None or len(self.policy.angi) == 0 ):
            self.policy.angi = self.plant.angi

    def execute_training_iteration(self, states, controls, succesor_states):
        self.plant.update(states, controls, successor_states)
        self.policy.optimize(self.plant, self.cost)

    def rollout(self, m0, S0, H):
        # sample initial state
        m = np.random.multivariate_normal(m0,s0)
        s = S0

        for i in xrange(H):
            # apply policy (covariance empty for deterministic policies)
            u = policy.fcn(m)
            np.append(self.x,x)
            np.append(self.x_cov,x_cov)
            np.append(self.u,u)
            np.append(self.u_cov,u_cov)

            # compute cost ( need not depend on the control input or covariance)
            np.append(L,cost.fcn(x,u,x_cov,u_cov))

            # step the plant (covariance empty for deterministic systems)
            x,x_cov = plant.fcn(x,u,x_cov,u_cov)
            np.append(self.y,x)
            np.append(self.y_cov,x_cov)

    def value(self):
        # TODO implement this
        pass

    def minimize(self):
        # TODO implement this

