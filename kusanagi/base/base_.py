# pylint: disable=C0103
from kusanagi import utils
import numpy as np


def preprocess_angles(x_t, angle_dims=[]):
    x_t_ = utils.gTrig_np(x_t[None, :], angle_dims)
    return x_t_


def apply_controller(env, policy, max_steps, preprocess=None, callback=None):
    '''
        Starts the env and applies the current policy to the env for a duration
        specified by H (in seconds). If  H is not set, it will run for self.H
        seconds.
        @param env interface to the system being controller
        @param policy Interface to the controller to be applied to the system
        @param max_steps Horizon for applying controller (in seconds)
        @param callback Callable object to be called after every time step
    '''
    fnname = 'apply_controller'
    # initialize policy if needed
    if hasattr(policy, 'get_params'):
        p = policy.get_params()
        if len(p) == 0:
            policy.init_params()
        # making sure we initialize the policy before resetting the plant
        policy(np.zeros((policy.D,)))

    # start robot
    utils.print_with_stamp('Starting run', fnname)
    if hasattr(env, 'dt'):
        H = max_steps*env.dt
        utils.print_with_stamp('Running for %f seconds' % (H), fnname)
    else:
        utils.print_with_stamp('Running for %d steps' % (max_steps), fnname)
    x_t = env.reset()

    # data corresponds to state at time t, action at time t, reward after
    # applying action at time t
    data = []

    # do rollout
    for t in range(max_steps):
        # preprocess state
        x_t_ = preprocess(x_t) if callable(preprocess) else x_t

        #  get command from policy
        u_t = policy(x_t_, t=t)
        if isinstance(u_t, list) or isinstance(u_t, tuple):
            u_t = u_t[0].flatten()
        else:
            u_t = u_t.flatten()

        # apply control and step the env
        x_next, c_t, done, info = env.step(u_t)
        info['done'] = done

        # append to dataset
        data.append((x_t, u_t, c_t, info))

        # send data to callback
        if callable(callback):
            callback(x_t, u_t, c_t, info)

        # break if done
        if done:
            break

        # replace current state
        x_t = x_next

    states, actions, costs, infos = zip(*data)

    msg = 'Done. Stopping robot.'
    if all([v is not None for v in costs]):
        run_value = np.array(costs).sum()
        msg += ' Value of run [%f]' % run_value
    utils.print_with_stamp(msg, fnname)

    # stop robot
    if hasattr(env, 'stop'):
        env.stop()

    return states, actions, costs, infos


def train_dynamics(dynmodel, data, angle_dims=[],
                   init_episode=0, max_episodes=None,
                   max_dataset_size=0,
                   wrap_angles=False, append=False):
    ''' Trains a dynamics model using the data dataset '''
    utils.print_with_stamp('Training dynamics model', 'train_dynamics')

    X = []
    Y = []
    n_episodes = len(data.states)
    if n_episodes > init_episode:
        # get dataset for dynamics model
        episodes = list(range(init_episode, n_episodes))\
            if max_episodes is None or n_episodes < max_episodes\
            else list(range(max(0, n_episodes-max_episodes), n_episodes))

        X, Y = data.get_dynmodel_dataset(filter_episodes=episodes,
                                         angle_dims=angle_dims,
                                         deltas=True)
        X = X[-max_dataset_size:]
        Y = Y[-max_dataset_size:]
        # wrap angles if requested
        # (this might introduce error if the angular velocities are high)
        if wrap_angles:
            # wrap angle differences to [-pi,pi]
            Y[:, angle_dims] = (Y[:, angle_dims] + np.pi) % (2 * np.pi) - np.pi

        if append:
            # append data to the dynamics model
            dynmodel.append_dataset(X, Y)
        else:
            dynmodel.set_dataset(X, Y)

    i_shp = dynmodel.X.get_value(borrow=True).shape
    o_shp = dynmodel.Y.get_value(borrow=True).shape
    msg = 'Dataset size:: Inputs: [ %s ], Targets: [ %s ] ' % (i_shp, o_shp)
    utils.print_with_stamp(msg, 'train_dynamics')

    # finally, train the dynamics model
    dynmodel.train()
    utils.print_with_stamp('Done training dynamics model', 'train_dynamics')

    return dynmodel
