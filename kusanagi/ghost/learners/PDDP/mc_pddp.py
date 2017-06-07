from kusanagi.ghost.learners.PILCO import MC_PILCO
import theano.tensor as tt
import theano
import numpy as np

class MC_PDDP(MC_PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None,
                 dynmodel_class=kreg.GP_UI, n_samples=10, experience=None, async_plant=False,
                 name='MC_PILCO', filename_prefix=None):
        super(MC_PDDP, self).__init__(params, plant_class, policy_class, cost_func,
                                      viz_class, dynmodel_class, experience, async_plant,
                                      name, filename_prefix)
        self.forward_step_fn = None
        self.x_t = theano.shared(self.x0.get_value().copy())
        self.u_t = theano.shared(np.zeros((n_samples,len(self.maxU)))

        def forward_step(self):
            if not self.forward_step_fn:
                x = tt.matrix('x')
                u = tt.matrix('u')
                x_next = self.propagate_state(x,u)
                Fx, Fu = theano.jacobian(x_next, [x, u])

                self.forward_step_fn = theano.function([x_next, Fx, Fu],[x,u])
            return self.forward_step