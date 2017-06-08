import numpy as np
import time
from kusanagi import utils

def apply_controller(env, agent, dt=None, H=None, realtime=False):
    '''
    Starts the environment and applies the current policy in it for a duration specified by H
    (in seconds). If  H is not set, it will run for agent.H seconds. Timestamps should be provided
    by the environment. If this is not true, agent.dt will be used to produce timestamps (this might
    not be consistent with the actual control rate!)

    @param env Environment where the agent will be tested
    @param agent Policy/Controller/Algorithm that defines the agents behavior in the environment
    @param H Horizon for applying controller (in seconds)
    '''
    # initialize policy parameters
    p = agent.policy.get_params()
    if len(p) == 0:
        agent.policy.init_params()

    # mark the start of the episode
    agent.experience.new_episode(policy_params=p)

    # set control horizon
    utils.print_with_stamp('Starting data collection run', agent.name)
    if not H:
        H = agent.H
    utils.print_with_stamp('Running for %f seconds'%(H), agent.name)
    # set dt
    if not dt:
        dt = agent.dt
    H_steps = int(np.ceil(H/dt))

    # start robot if asynchronous
    if hasattr(env,'asynchronous') and env.asynchronous:
        env.start()

    exec_time = time.time()
    # read initial state (noisy measurement)
    x = env.reset()
    # get initial timestamp
    t = env.get_time() if hasattr(env,'get_time') else 0

    # do rollout
    for i in range(H_steps):
        # preprocess state for the agent's policy
        x_ = agent.preprocess(x)

        # get command from policy
        u = agent.policy.evaluate(x_)[0].flatten()

        # send command to robot. The internal logic for
        # step should deal with the synchronization so that
        # r corresponds to the reward AFTER applying the action
        x_next, r, done, info = env.step(u)

        # get timestamp
        t = info['time'] if 'time' in info else t+dt

        # append to experience dataset
        # time stamp, state, action, reward after applying action
        agent.experience.add_sample(t, x, u, r)

        # sleep to match the desired sample rate
        if realtime:
            exec_time = time.time() - exec_time
            if exec_time < dt:
                time.sleep(dt-exec_time)

        x = x_next

        if done:
            break

    # stop robot
    run_value = np.array(agent.experience.immediate_cost[-1][:-1])
    utils.print_with_stamp('Done. Stopping robot. Value of run [%f]'%(run_value.sum()), self.name)
    if hasattr(env, 'stop'):
        env.stop()

    agent.n_episodes = len(agent.experience.states)

    return agent.experience
