#!/usr/bin/env python
import sys
import pickle
import numpy as np
from collections import OrderedDict
from flask import Flask, request
from werkzeug.utils import secure_filename

from kusanagi.ghost.algorithms import mc_pilco
from kusanagi import utils
from kusanagi.base import train_dynamics


ALLOWED_EXTENSIONS = set(['zip', 'pkl'])
DEBUG = True

task_spec_dict = {}


def mc_pilco_polopt(task_name, task_spec):
    '''
    executes one iteration of mc_pilco (model updating and policy optimization)
    '''
    # get task specific variables
    dyn = task_spec['transition_model']
    exp = task_spec['experience']
    pol = task_spec['policy']
    plant_params = task_spec['plant']
    immediate_cost = task_spec['cost']['graph']
    H = int(np.ceil(task_spec['horizon_secs']/plant_params['dt']))
    n_samples = task_spec.get('n_samples', 100)

    # if state != 'init':
    # train dynamics model. TODO block if training multiple tasks with
    # the same model
    train_dynamics(
        dyn, exp, pol.angle_dims, wrap_angles=task_spec['wrap_angles'])

    # init policy optimizer if needed
    optimizer = task_spec['optimizer']
    if optimizer.loss_fn is None:
        # task_state[task_name] = 'compile_polopt'

        # get policy optimizer options
        split_H = task_spec.get('split_H', 1)
        noisy_policy_input = task_spec.get('noisy_policy_input', False)
        noisy_cost_input = task_spec.get('noisy_cost_input', False)
        truncate_gradient = task_spec.get('truncate_gradient', -1)
        learning_rate = task_spec.get('learning_rate', 1e-3)
        gradient_clip = task_spec.get('gradient_clip', 1.0)

        # get extra inputs, if needed
        import theano.tensor as tt
        ex_in = OrderedDict([(k, v) for k, v in immediate_cost.keywords.items()
                            if type(v) is tt.TensorVariable
                            and len(v.get_parents()) == 0])
        task_spec['extra_in'] = ex_in

        # build loss function
        loss, inps, updts = mc_pilco.get_loss(
            pol, dyn, immediate_cost,
            n_samples=n_samples,
            noisy_cost_input=noisy_cost_input,
            noisy_policy_input=noisy_policy_input,
            split_H=split_H,
            truncate_gradient=(H/split_H)-truncate_gradient,
            crn=100,
            **ex_in)
        inps += ex_in.values()

        # add loss function as objective for optimizer
        optimizer.set_objective(
            loss, pol.get_params(symbolic=True), inps, updts,
            clip=gradient_clip, learning_rate=learning_rate)

    # train policy # TODO block if learning a multitask policy
    # task_state[task_name] = 'update_polopt'
    # build inputs to optimizer
    p0 = plant_params['state0_dist']
    gamma = task_spec['discount']
    polopt_args = [p0.mean, p0.cov, H, gamma]
    extra_in = task_spec.get('extra_in', OrderedDict)
    if len(extra_in) > 0:
        polopt_args += [task_spec['cost']['params'][k] for k in extra_in]

    # update dyn and pol (resampling)
    def callback(*args, **kwargs):
        if hasattr(dyn, 'update'):
            dyn.update(n_samples)
        if hasattr(pol, 'update'):
            pol.update(n_samples)
    # call minimize
    callback()
    optimizer.minimize(
        *polopt_args, return_best=task_spec['return_best'])
    # task_state[task_name] = 'ready'

    # check if task is done
    # n_polopt_iters = len([p for p in exp.policy_parameters if len(p) > 0])
    # if n_polopt_iters >= task_spec['n_opt']:
    #     task_state[task_name] = 'done'
    return pol.get_params(symbolic=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/get_task_init_status/<string:task_id>", methods=['GET'])
def get_task_init_status(task_id):

    sys.stderr.write("GET REQUEST: get_task_init_status/%s" % task_id+"\n")

    response = "NOT FOUND"
    if task_id in task_spec_dict:
        response = "INITIALISED"

    return "get_task_init_status/%s: %s" % (task_id, response)


@app.route("/init_task/<string:task_id>", methods=['POST'])
def init_task(task_id):

    sys.stderr.write("POST REQUEST: init_task/%s" % task_id+"\n")

    response = "FAILED"

    if 'tspec_file' not in request.files:
        response = "tspec_file missing"
        sys.stderr.write(response + "\n")

    else:
        f_tspec = request.files['tspec_file']
        if f_tspec.filename == '':
            response = "tspec_file not selected"
            sys.stderr.write(response + "\n")

        elif f_tspec and allowed_file(f_tspec.filename):
            tspec_filename = secure_filename(f_tspec.filename)
            task_spec_dict[task_id] = pickle.loads(f_tspec.read())

            sys.stderr.write("Received file:\t" + tspec_filename + "\t")

            response = "DONE"

    return "init_task/%s: %s" % (task_id, response)


@app.route("/optimize/<task_id>", methods=['POST'])
def optimize(task_id):
    utils.set_logfile("%s.log" % task_id, base_path="/localdata")
    sys.stderr.write("POST REQUEST: optimize/%s" % task_id+"\n")

    response = "FAILED"

    if task_id not in task_spec_dict:
        response = "TASK NOT INITIALIZED"

    elif 'exp_file' not in request.files:
        response = "exp_file missing"
        sys.stderr.write(response + "\n")

    elif 'pol_params_file' not in request.files:
        response = "pol_params_file missing"
        sys.stderr.write(response + "\n")

    else:
        f_exp = request.files['exp_file']
        f_pol_params = request.files['pol_params_file']

        if f_exp and allowed_file(f_exp.filename) and \
           f_pol_params and allowed_file(f_pol_params.filename):

            exp_filename = secure_filename(f_exp.filename)
            pol_params_filename = secure_filename(f_pol_params.filename)
            sys.stderr.write("Received files:\t" + exp_filename + "\t"
                                                 + pol_params_filename + "\n")

            task_spec_dict[task_id]['experience'] = pickle.loads(f_exp.read())
            task_spec_dict[task_id]['policy'].set_params(
                pickle.loads(f_pol_params.read()))
            f_exp.close(), f_pol_params.close()

            pol_params = mc_pilco_polopt(task_id, task_spec_dict[task_id])

            response = pickle.dumps(pol_params)

    return response  # "optimize/%s: %s" % (task_id, response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008)
