#!/usr/bin/env python
from flask import Flask, request
import socket
from kusanagi.ghost.algorithms import mc_pilco


def mc_pilco_polopt():
    '''
    executes one iteration of mc_pilco (model updating and policy optimization)
    '''
    # get task specific variables
    n_samples = task_spec['n_samples']
    dyn = task_spec['transition_model']
    exp = task_spec.get('experience_data', ExperienceDataset())
    pol = task_spec['policy']
    plant_params = task_spec['plant']
    immediate_cost = task_spec['cost']['graph']

    # append new experrience to dataset
    task_state[task_name] = 'update_dyn'
    states, actions, costs, infos = experience
    ts = [info.get('t', None) for info in infos]
    exp.append_episode(states, actions, costs, infos,
                       pol.get_params(symbolic=False), ts)
    task_spec['experience_data'] = exp

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
            pol, dyn, immediate_cost, n_samples=n_samples, **ex_in)
        inps += ex_in.values()
        optimizer.set_objective(
            loss, pol.get_params(symbolic=True), inps, updts)

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
    if hasattr(dyn, 'update'):
        dyn.update(n_samples)
    if hasattr(pol, 'update'):
        pol.update(n_samples)
    # call minimize
    optimizer.minimize(*polopt_args,
                       return_best=task_spec['return_best'])

    # put task in the queue for execution
    task_state[task_name] = 'ready'
    task_queue.put((task_name, task_spec))

    return


app = Flask(__name__)

@app.route("/optimize/<task_id>", methods=['GET','POST'])
def optimize(task_id):
    if request.method == "GET":
        # return status of current task _id
    elif request.method == "POST":
        ret = "POST"+str(task_id)
    
    html = "<h3>kusanagi web {task_id}</h3>"\
           "<b>host: </b> {host} <br>"\
           "<b>request: </b> {ret} <br>"
    return html.format(task_id=task_id, host=socket.gethostname(), ret=ret)

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8008)