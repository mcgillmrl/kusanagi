'''
Example of how to use the library for learning using the PILCO learner
on the double_cartpole task
'''
# pylint: disable=C0103
import argparse
import dill
import os
import sys
import numpy as np
import lasagne
import theano

from functools import partial
from kusanagi import utils
from kusanagi.ghost import regression, control
from kusanagi.shell import experiment_utils, double_cartpole

# np.random.seed(1337)
np.set_printoptions(linewidth=500)


def eval_str_arg(arg):
    if type(arg) is str:
        arg = eval(arg)
    return arg


def experiment1_params(n_rnd=1, n_opt=100, dynmodel_class=regression.SSGP_UI,
                       **kwargs):
    ''' pilco with rbf controller'''
    params = double_cartpole.default_params()
    params['n_rnd'] = int(n_rnd)
    params['n_opt'] = int(n_opt)
    for key in kwargs:
        if key in params:
            params[key] = eval(kwargs[key])
    params['dynmodel_class'] = dynmodel_class

    loss_kwargs = {}
    polopt_kwargs = {}
    extra_inps = []

    return params, loss_kwargs, polopt_kwargs, extra_inps


def experiment2_params(n_rnd=1, n_opt=100,
                       mc_samples=10, learning_rate=1e-3,
                       polyak_averaging=None,
                       min_method='adam', max_evals=1000,
                       mm_state=True,
                       mm_cost=True,
                       noisy_policy_input=False,
                       noisy_cost_input=False,
                       heteroscedastic_dyn=False,
                       crn=True, crn_dropout=True,
                       clip_gradients=1.0, **kwargs):
    ''' mc-pilco with rbf controller'''
    mc_samples = int(mc_samples)
    n_rnd = int(n_rnd)
    n_opt = int(n_opt)
    max_evals = int(max_evals)
    learning_rate = float(learning_rate)
    heteroscedastic_dyn = eval_str_arg(heteroscedastic_dyn)
    mm_state = eval_str_arg(mm_state)
    mm_cost = eval_str_arg(mm_cost)
    noisy_policy_input = eval_str_arg(noisy_policy_input)
    noisy_cost_input = eval_str_arg(noisy_cost_input)
    clip_gradients = eval_str_arg(clip_gradients)
    polyak_averaging = eval_str_arg(polyak_averaging)
    crn = eval_str_arg(crn)
    crn_dropout = eval_str_arg(crn_dropout)

    scenario_params = experiment1_params(n_rnd, n_opt)
    params, loss_kwargs, polopt_kwargs, extra_inps = scenario_params

    # params for the dynamics model
    params['dynamics_model']['heteroscedastic'] = heteroscedastic_dyn
    params['dynamics_model']['n_samples'] = mc_samples

    # parameters for building loss function
    loss_kwargs['n_samples'] = mc_samples
    loss_kwargs['mm_state'] = mm_state
    loss_kwargs['mm_cost'] = mm_cost
    loss_kwargs['noisy_policy_input'] = noisy_policy_input
    loss_kwargs['noisy_cost_input'] = noisy_cost_input
    loss_kwargs['crn'] = crn

    # init symbolic learning rate parameter
    lr = theano.tensor.scalar('lr')
    extra_inps += [lr]

    # optimizer parameters
    params['optimizer']['min_method'] = min_method
    params['optimizer']['max_evals'] = max_evals
    polopt_kwargs['learning_rate'] = lr
    polopt_kwargs['clip'] = clip_gradients
    polopt_kwargs['polyak_averaging'] = polyak_averaging
    params['learning_rate'] = learning_rate
    params['crn_dropout'] = crn_dropout

    return params, loss_kwargs, polopt_kwargs, extra_inps


def get_scenario(experiment_id, *args, **kwargs):
    pol = None
    dyn = None

    if experiment_id == 1:
        # PILCO with rbf controller
        scenario_params = experiment1_params(*args, **kwargs)
        learner_setup = experiment_utils.pilco_double_cartpole_experiment
        params = scenario_params[0]

        pol = control.RBFPolicy(**params['policy'])

    elif experiment_id == 2:
        # PILCO with nn controller 1
        scenario_params = experiment1_params(*args, **kwargs)
        learner_setup = experiment_utils.pilco_double_cartpole_experiment
        params = scenario_params[0]
        p0 = params['state0_dist']

        pol = control.NNPolicy(p0.mean.size, **params['policy'])
        pol_spec = regression.mlp(
            input_dims=pol.D,
            output_dims=pol.E,
            hidden_dims=[50]*4,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            output_nonlinearity=pol.sat_func,
            name=pol.name)
        pol.build_network(pol_spec)

    elif experiment_id == 3:
        # mc PILCO with RBF controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        pol = control.RBFPolicy(**params['policy'])

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=0.1, p_input=0.0,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

    elif experiment_id == 4:
        # mc PILCO with NN controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        p0 = params['state0_dist']

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=0.1, p_input=0.0,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

        # init policy
        pol = control.NNPolicy(p0.mean.size, **params['policy'])
        pol_spec = regression.mlp(
            input_dims=pol.D,
            output_dims=pol.E,
            hidden_dims=[50]*4,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            output_nonlinearity=pol.sat_func,
            name=pol.name)
        pol.build_network(pol_spec)

    elif experiment_id == 5:
        # mc PILCO with RBF controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        pol = control.RBFPolicy(**params['policy'])

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        # for the log normal dropout layers, the dropout probabilities 
        # are dummy variables to enable dropout (not actual dropout probs)
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=True, p_input=True,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseLogNormalDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

    elif experiment_id == 6:
        # mc PILCO with NN controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        p0 = params['state0_dist']

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=True, p_input=True,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseLogNormalDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

        # init policy
        pol = control.NNPolicy(p0.mean.size, **params['policy'])
        pol_spec = regression.mlp(
            input_dims=pol.D,
            output_dims=pol.E,
            hidden_dims=[50]*4,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            output_nonlinearity=pol.sat_func,
            name=pol.name)
        pol.build_network(pol_spec)

    elif experiment_id == 7:
        # mc PILCO with dropout controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        p0 = params['state0_dist']

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=0.1, p_input=0.0,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

        # init policy
        pol = control.NNPolicy(p0.mean.size, **params['policy'])
        pol_spec = regression.dropout_mlp(
            input_dims=pol.D,
            output_dims=pol.E,
            hidden_dims=[50]*4,
            p=0.1, p_input=0.0,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            output_nonlinearity=pol.sat_func,
            dropout_class=regression.layers.DenseDropoutLayer,
            name=pol.name)
        pol.build_network(pol_spec)

    elif experiment_id == 8:
        # mc PILCO with dropout controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        p0 = params['state0_dist']

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=True, p_input=True,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseLogNormalDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

        # init policy
        pol = control.NNPolicy(p0.mean.size, **params['policy'])
        pol_spec = regression.dropout_mlp(
            input_dims=pol.D,
            output_dims=pol.E,
            hidden_dims=[50]*4,
            p=0.1, p_input=0.0,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            output_nonlinearity=pol.sat_func,
            dropout_class=regression.layers.DenseDropoutLayer,
            name=pol.name)
        pol.build_network(pol_spec)

    elif experiment_id == 9:
        # mc PILCO with dropout controller and dropout mlp dynamics
        scenario_params = experiment2_params(*args, **kwargs)
        learner_setup = experiment_utils.mcpilco_double_cartpole_experiment
        params = scenario_params[0]
        p0 = params['state0_dist']

        # init dyn to use dropout
        dyn = regression.BNN(**params['dynamics_model'])
        odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=odims,
            hidden_dims=[200]*4,
            p=0.1, p_input=0.01,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            dropout_class=regression.layers.DenseConcreteDropoutLayer,
            name=dyn.name)
        dyn.build_network(dyn_spec)

        # init policy
        pol = control.NNPolicy(p0.mean.size, **params['policy'])
        pol_spec = regression.dropout_mlp(
            input_dims=pol.D,
            output_dims=pol.E,
            hidden_dims=[50]*4,
            p=0.1, p_input=0.0,
            nonlinearities=regression.nonlinearities.rectify,
            W_init=lasagne.init.GlorotNormal(),
            output_nonlinearity=pol.sat_func,
            dropout_class=regression.layers.DenseDropoutLayer,
            name=pol.name)
        pol.build_network(pol_spec)

    return scenario_params, pol, dyn, learner_setup


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--exp', type=int, default=1,
        help='id of experiment to run')
    parser.add_argument(
        '-n', '--name', type=str, default='cartpole',
        help='experiment name')
    parser.add_argument(
        '-o', '--output_folder', type=str, default=utils.get_output_dir(),
        help='where to save the results of the experiment')
    parser.add_argument(
        '-r', '--render', type=bool, default=False,
        help='whether to call env.render')
    parser.add_argument(
        '-d', '--debug_plot', type=int, default=0,
        help='whether to plot rollouts and other debugging information')
    parser.add_argument(
        '-k', '--kwarg', nargs=2, action='append', default=[],
        help='additional arguments for the experiment [name value]')
    args = parser.parse_args()

    e_id = args.exp
    odir = args.output_folder
    name = args.name+'_'+str(e_id)
    output_folder = os.path.join(odir, name)
    kwargs = dict(args.kwarg)

    try:
        os.mkdir(output_folder)
    except:
        # move the old stuff
        target_dir = output_folder+'_'+str(os.stat(output_folder).st_ctime)
        os.rename(output_folder, target_dir)
        os.mkdir(output_folder)
        utils.print_with_stamp(
            'Moved old results from [%s] to [%s]' % (output_folder, 
                                                     target_dir))

    utils.print_with_stamp('Results will be saved in [%s]' % (output_folder))

    scenario_params, pol, dyn, learner_setup = get_scenario(e_id, **kwargs)

    params, loss_kwargs, polopt_kwargs, extra_inps = scenario_params

    # write the inital configuration to disk
    params_path = os.path.join(output_folder, 'initial_config.dill')
    with open(params_path, 'wb+') as f:
        config_dict = dict(params=params, loss_kwargs=loss_kwargs,
                           polopt_kwargs=polopt_kwargs, extra_inps=extra_inps)

        dill.dump(config_dict, f)

    scenario = partial(
        learner_setup, policy=pol, dynmodel=dyn)

    # callback executed after every learning iteration
    def iter_cb(exp, dyn, pol, polopt, params):
        i = exp.curr_episode
        # setup output directory
        exp.save(output_folder, 'experience_%d' % (i))
        pol.save(output_folder, 'policy_%d' % (i))
        dyn.save(output_folder, 'dynamics_%d' % (i))
        # TODO save state of the optimizer

    # run pilco
    experiment_utils.run_pilco_experiment(
        scenario, params, loss_kwargs, polopt_kwargs, extra_inps,
        learning_iteration_cb=iter_cb, render=args.render,
        debug_plot=args.debug_plot)

    print('Finished experiment')
    sys.exit(0)
