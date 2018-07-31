import argparse
import dill
import os
import pickle as pkl

from functools import partial

import kusanagi
from kusanagi.base import ExperienceDataset
from kusanagi.ghost import control
from kusanagi.shell import experiment_utils
from kusanagi import utils


def recursive_getattr(obj, attr):
    val = getattr(obj, attr, None)
    if val is None:
        attr1, attr_next = attr.split('.')
        next_mod = getattr(obj, attr1)
        #print next_mod
        if attr == attr1 or next_mod is None:
            raise AttributeError('Can\'t find %s in module %s' % (obj, attr))
        else:
            return recursive_getattr(next_mod, attr_next)
    else:
        return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_folder', type=str,
                        default=utils.get_output_dir(),
                        help='where to load the results of the experiment')
    parser.add_argument('-r', '--render', type=bool,
                        default=False,
                        help='whether to call env.render')
    parser.add_argument(
        '-k', '--kwarg', nargs=2, action='append', default=[],
        help='additional arguments for the experiment [name value]')
    parser.add_argument(
        '-e', '--env', type=str, default='cartpole.Cartpole',
        help='environment from kusanagi.shell')
    parser.add_argument(
        '-c', '--cost', type=str, default='cartpole.cartpole_loss',
        help='cost funciton from kusanagi.shell')   
    parser.add_argument('-p', '--policy_class', type=str,
                        default='NNPolicy',
                        help='Policy class (in kusanagi.ghost.control)')

    args = parser.parse_args()

    odir = args.dataset_folder
    kwargs = dict(args.kwarg)
    n_trials = int(kwargs.get('n_trials', 5))
    last_iteration = int(kwargs.get('last_iteration', 5))
    config_path = os.path.join(odir, 'initial_config.dill')
    exp_path = os.path.join(odir, 'experience_%d' % (last_iteration))
    pol_path = os.path.join(odir, 'policy_%d' % (last_iteration))
    env_class = recursive_getattr(kusanagi.shell, args.env)
    cost_func = recursive_getattr(kusanagi.shell, args.cost)
    policy_class = getattr(control, args.policy_class)

    with open(config_path, 'rb') as f:
        config_dict = dill.load(f)

    params = config_dict['params']
    p0 = params['state0_dist']
    exp = ExperienceDataset(filename=exp_path)
    if args.policy_class == 'NNPolicy':
        pol = policy_class(p0.mean.size, filename=pol_path, **params['policy'])
    else:
        pol = policy_class(filename=pol_path, **params['policy'])

    # init cost model
    cost = partial(cost_func, **params['cost'])
    # init environment
    env = env_class(loss_func=cost, **params['plant'])

    # evaluate policy
    results = experiment_utils.evaluate_policy(
        env, pol, exp, params, n_trials, render=args.render)

    # dump results to file
    results_path = os.path.join(
        odir, 'results_%d_%d' % (last_iteration, n_trials))
    with open(results_path, 'wb+') as f:
        utils.print_with_stamp('Dumping results to [%s]' % (results_path))
        pkl.dump(results, f, 2)
