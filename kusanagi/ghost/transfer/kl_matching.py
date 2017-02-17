
class KLMatching(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=kreg.GP_UI, invdynmodel_class=kreg.GP_UI, experience = None, async_plant=False, name='TrajectoryMatching', wrap_angles=False, filename_prefix=None):
        self.angle_idims = params['angle_dims']
        self.maxU = params['policy']['maxU']
        # initialize source policy
        params['policy']['source_policy'] = params['source_policy']
        # initialize source policy
        self.source_experience = params['source_experience']
        
        # initialize dynamics model
        # input dimensions to the dynamics model are (state dims - angle dims) + 2*(angle dims) + control dims
        x0 = np.array(params['x0'],dtype='float64').squeeze()
        S0 = np.array(params['S0'],dtype='float64').squeeze()
        self.mx0 = theano.shared(x0.astype('float64'))
        self.Sx0 = theano.shared(S0.astype('float64'))
        dyn_idims = len(x0) + len(self.angle_idims) + len(self.maxU)
        # output dimensions are state dims
        dyn_odims = len(x0)
        # initialize dynamics model (TODO pass this as argument to constructor)
        if 'inv_dynmodel' not in params:
            params['invdynmodel'] = {}
        params['invdynmodel']['idims'] = 2*dyn_odims
        params['invdynmodel']['odims'] = len(self.maxU)#dyn_odims

        self.inverse_dynamics_model = invdynmodel_class(**params['invdynmodel'])
        self.next_episode_inv = 0
        super(KLMatching, self).__init__(params, plant_class, policy_class, cost_func,viz_class, dynmodel_class,  experience, async_plant, name, filename_prefix)

    def train_forward_dynamics(exp,model=None,dynmodel_class=kreg.GP_UI,params={}):
        ''' Trains a dynamics model using the current experience dataset '''
        utils.print_with_stamp('Training dynamics model',self.name)

        X = []
        Y = []
        n_episodes = len(.states)
        
        if n_episodes>0:
            # construct training dataset
            for i in xrange(self.next_episode,n_episodes):
                x = np.array(self.experience.states[i])
                u = np.array(self.experience.actions[i])

                # inputs are states, concatenated with actions ( excluding the last entry) 
                x_ = utils.gTrig_np(x, self.angle_idims)
                X.append( np.hstack((x_[:-1],u[:-1])) )
                # outputs are changes in state
                Y.append( x[1:] - x[:-1] )

            self.next_episode = n_episodes 
            X = np.vstack(X)
            Y = np.vstack(Y)
            
            # wrap angles if requested (this might introduce error if the angular velocities are high )
            if self.wrap_angles:
                # wrap angle differences to [-pi,pi]
                Y[:,self.angle_idims] = (Y[:,self.angle_idims] + np.pi) % (2 * np.pi ) - np.pi

            # get distribution of initial states
            x0 = np.array([x[0] for x in self.experience.states])
            if n_episodes > 1:
                self.mx0.set_value(x0.mean(0).astype('float64'))
                self.Sx0.set_value(np.cov(x0.T).astype('float64'))
            else:
                self.mx0.set_value(x0.astype('float64').flatten())
                self.Sx0.set_value(1e-2*np.eye(x0.size).astype('float64'))

            # append data to the dynamics model
            self.dynamics_model.append_dataset(X,Y)
        else:
            x0 = np.array(self.plant.x0, dtype='float64').squeeze()
            S0 = np.array(self.plant.S0, dtype='float64').squeeze()
            self.mx0.set_value(x0)
            self.Sx0.set_value(S0)
        
        #utils.print_with_stamp('%s, \n%s'%(self.mx0.get_value(), self.Sx0.get_value()),self.name)
        utils.print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(self.dynamics_model.X.get_value(borrow=True).shape,self.dynamics_model.Y.get_value(borrow=True).shape),self.name)
        if self.dynamics_model.should_recompile:
            # reinitialize log likelihood
            self.dynamics_model.init_loss()
            self.should_recompile = True
 
        self.dynamics_model.train()
        utils.print_with_stamp('Done training dynamics model',self.name)

