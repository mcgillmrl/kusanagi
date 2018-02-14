#!/usr/bin/env python
import os
import sys
import socket
import pickle
from collections import OrderedDict
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

from kusanagi.ghost.algorithms import mc_pilco
from kusanagi import utils
from kusanagi.base import (apply_controller, train_dynamics,
                           preprocess_angles, ExperienceDataset)

UPLOAD_FOLDER = '/home/automation/nikhil/workspace/roughwork_ws/robot_learning_server/uploads' #TODO: Check whether the folder exists
ALLOWED_EXTENSIONS = set(['zip', 'pkl'])
DEBUG = True

task_spec_dict = {}

def mc_pilco_polopt(task_name, task_spec):
    '''
    executes one iteration of mc_pilco (model updating and policy optimization)
    '''
    # get task specific variables
    n_samples = task_spec['n_samples']
    dyn = task_spec['transition_model']
    exp = task_spec['experience']
    pol = task_spec['policy']
    plant_params = task_spec['plant']
    immediate_cost = task_spec['cost']['graph']

    if state != 'init':
        # train dynamics model. TODO block if training multiple tasks with
        # the same model
        train_dynamics(dyn, exp, pol.angle_dims,
                    wrap_angles=task_spec['wrap_angles'])

        # init policy optimizer if needed
        optimizer = task_spec['optimizer']
        if optimizer.loss_fn is None:
            task_state[task_name] = 'compile_polopt'
            import theano.tensor as tt
            ex_in = OrderedDict([(k, v) for k, v in immediate_cost.keywords.items()
                                if type(v) is tt.TensorVariable
                                and len(v.get_parents()) == 0])
            task_spec['extra_in'] = ex_in
            loss, inps, updts = mc_pilco.get_loss(
                pol, dyn, immediate_cost, n_samples=n_samples, lr=1e-3,
                noisy_cost_input=False, noisy_policy_input=True,
                 **ex_in)
            inps += ex_in.values()
            optimizer.set_objective(
                loss, pol.get_params(symbolic=True), inps, updts, clip=1.0)

        # train policy # TODO block if learning a multitask policy
        task_state[task_name] = 'update_polopt'
        # build inputs to optimizer
        p0 = plant_params['state0_dist']
        H = int(np.ceil(task_spec['horizon_secs']/plant_params['dt']))
        gamma = task_spec['discount']
        polopt_args = [p0.mean, p0.cov, H, gamma]
        extra_in = task_spec.get('extra_in', OrderedDict)
        if len(extra_in) > 0:
            polopt_args += [task_spec['cost']['params'][k] for k in extra_in]
        # update dyn and pol (resampling)
        def callback(*args,**kwargs):
            if hasattr(dyn, 'update'):
                dyn.update(n_samples)
            if hasattr(pol, 'update'):
                pol.update(n_samples)
        # call minimize
        callback()
        optimizer.minimize(*polopt_args,
                        return_best=task_spec['return_best'])
        task_state[task_name] = 'ready'

    return pol.get_params(symbolic=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/check/<task_id>", methods=['GET'])
def check(task_id):
    if request.method == "GET":
        # return status of current task_id
        ret = str( task_spec_dict.get(task_id, "NOT FOUND") )
    else:
        ret = ""
    return ret

@app.route("/init/<task_id>", methods=['POST'])
def init(task_id):
    sys.stderr.write(str(request.files)+"\n")
    if 'tspec_file' not in request.files:
        sys.stderr.write('tspec_file missing\n')

    f_tspec = request.files['tspec_file']
    if f_tspec.filename == '':
        sys.stderr.write('tspec_file not selected\n')


    if f_tspec and allowed_file(f_tspec.filename):
        tspec_filename = secure_filename(f_tspec.filename)
        sys.stderr.write("Received file:\t" + tspec_filename + "\t")
        task_spec_dict[task_id] = pickle.load(f_tspec)

    return "DONE"

@app.route("/optimize/<task_id>", methods=['GET','POST'])
def optimize(task_id):
    '''
    Serves a http [GET, POST] request
    '''
    if request.method == "POST":
        ret = "POST"+str(task_id) #TODO: Use the task_id more meaningfully

        if task_id not in task_spec_dict:
            return "NOT FOUND"

        if DEBUG: sys.stderr.write(str(request.files)+"\n")

        # check if the post request has the file part
        print(request.files)
        if 'dyn_file' not in request.files:
            sys.stderr.write('dyn_file missing\n')
            return redirect(request.url)
        if 'exp_file' not in request.files:
            sys.stderr.write('exp_file missing\n')
            return redirect(request.url)
        if 'pol_file' not in request.files:
            sys.stderr.write('pol_file missing\n')
            return redirect(request.url)

        f_dyn = request.files['dyn_file']
        f_exp = request.files['exp_file']
        f_pol = request.files['pol_file']

        if f_dyn.filename == '':
            sys.stderr.write('dyn_file not selected\n')
            return redirect(request.url)
        if f_exp.filename == '':
            sys.stderr.write('exp_file not selected\n')
            return redirect(request.url)
        if f_pol.filename == '':
            sys.stderr.write('pol_file not selected\n')
            return redirect(request.url)

        if f_dyn and allowed_file(f_dyn.filename) and\
           f_exp and allowed_file(f_exp.filename) and\
           f_pol and allowed_file(f_pol.filename):

            dyn_filename = secure_filename(f_dyn.filename)
            exp_filename = secure_filename(f_exp.filename)
            pol_filename = secure_filename(f_pol.filename)

            sys.stderr.write("Recieved files:\t" + dyn_filename + "\t"
                                                 + exp_filename + "\t"
                                                 + pol_filename + "\n"
                            )

            task_spec = task_spec_dict[task_id]

            dyn = task_spec['transition_model']
            exp = task_spec['experience']
            pol = task_spec['policy']

            f_dyn.save(os.path.join(app.config['UPLOAD_FOLDER'], dyn_filename))
            f_exp.save(os.path.join(app.config['UPLOAD_FOLDER'], exp_filename))
            f_pol.save(os.path.join(app.config['UPLOAD_FOLDER'], pol_filename))

            exp.load(os.path.join(app.config['UPLOAD_FOLDER'], exp_filename))
            dyn.load(os.path.join(app.config['UPLOAD_FOLDER'], dyn_filename))
            pol.load(os.path.join(app.config['UPLOAD_FOLDER'], pol_filename))

            pol_params = mc_pilco_polopt(task_id, task_spec)

            return pol_params

            # return redirect(url_for('optimize', task_id=task_id))

        sys.stderr.write("ERROR: Parse error")


    return ret


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8008)
