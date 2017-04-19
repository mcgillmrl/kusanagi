# pylint: disable=C0103
import os
import sys
import stat
from datetime import datetime

import math
import time
import zipfile
import traceback

import random
import numpy as np
import csv
import theano
from theano import tensor as tt, ifelse
from theano.gof import Variable
from theano.sandbox.linalg import psd, matrix_inverse
import matplotlib as mpl
# This line is necessary for plot_and_save to work on server side without a GUI.
# Needs to be set before plt is imported.
# mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

def maha(X1, X2=None, M=None, all_pairs=True):
    ''' Returns the squared Mahalanobis distance'''
    D = []
    deltaM = []
    if X2 is None:
        X2 = X1
    if all_pairs:
        if M is None:
            D = tt.sum(tt.square(X1), 1).dimshuffle(0, 'x')\
                + tt.sum(tt.square(X2), 1)\
                - 2*X1.dot(X2.T)
        else:
            X1M = X1.dot(M)
            D = tt.sum(X1M*X1, 1).dimshuffle(0, 'x') + tt.sum(X2.dot(M)*X2, 1) - 2*X1M.dot(X2.T)
    else:
        # computes the distance  x1i - x2i for each row i
        if X1 is X2:
            # in this case, we don't need to compute anything
            D = tt.zeros((X1.shape[0],))
            return D
        delta = X1-X2
        if M is None:
            deltaM = delta
        else:
            deltaM = delta.dot(M)
        D = tt.sum(deltaM*delta, 1)
    return D

def fast_jacobian(expr, wrt, chunk_size=16, func=None):
    '''
    Computes the jacobian by tiling the inputs
    Copied from https://gist.github.com/aam-at/2b2bc5c35850b553d4ec
    '''
    assert isinstance(expr, Variable), \
        "tensor.jacobian expects a Variable as `expr`"
    assert expr.ndim < 2, \
        ("tensor.jacobian expects a 1 dimensional variable as "
         "`expr`. If not use flatten to make it a vector")

    num_chunks = tt.ceil(1.0 * expr.shape[0] / chunk_size)
    num_chunks = tt.cast(num_chunks, 'int32')
    steps = tt.arange(num_chunks)
    remainder = expr.shape[0] % chunk_size

    def chunk_grad(i):
        ''' operates on a subset of the gradient variables '''
        wrt_rep = tt.tile(wrt, (chunk_size, 1))
        if func is not None:
            expr_rep = func(wrt_rep)
        else:
            expr_rep, _ = theano.scan(
                fn=lambda wrt_: theano.clone(expr, {wrt: wrt_}),
                sequences=wrt_rep)
        chunk_expr_grad = tt.roll(
            tt.identity_like(expr_rep),
            i * chunk_size,
            axis=1)
        return tt.grad(cost=None,
                       wrt=wrt_rep,
                       known_grads={
                           expr_rep: chunk_expr_grad
                       })

    grads, _ = theano.scan(chunk_grad, sequences=steps)
    grads = grads.reshape((chunk_size * grads.shape[0], wrt.shape[0]))
    jac = ifelse.ifelse(tt.eq(remainder, 0), grads, grads[:expr.shape[0], :])
    return jac

def print_with_stamp(message, name=None, same_line=False, use_log=True):
    '''
    Helper function to print with a current time stamp.
    '''
    out_str = ''
    if name is None:
        out_str = '[%s] %s'%(str(datetime.now()), message)
    else:
        out_str = '[%s] %s > %s'%(str(datetime.now()), name, message)

    logfile = get_logfile()
    # this will only log to a file if 1) use_log is True and 2) $KUSANAGI_LOGFILE
    # is set ( can be set with utils.set_logfile(new_path) )
    if not use_log or not logfile:
        if same_line:
            sys.stdout.write('\r'+out_str)
        else:
            sys.stdout.write(out_str)
            print ''
        sys.stdout.flush()
    else:
        write_mode = 'a+'
        with open(logfile, write_mode) as f:
            if same_line:
                f.seek(0, os.SEEK_END)
                pos = f.tell() - 1
                c = f.read(1)
                while pos > 0 and c != '\r' and c != os.linesep:
                    pos = pos - 1
                    f.seek(pos, os.SEEK_SET)
                    c = f.read(1)
                if pos > 0 and c == '\r':
                    f.seek(pos, os.SEEK_SET)
                    f.truncate()
                f.write('\r')
            f.write(out_str+os.linesep)
        os.system('chmod 666 %s'%(logfile))

def kmeanspp(X, k):
    '''
    Initializer for kmeans
    '''
    N = X.shape[0]
    c = [X[random.randint(0, N)]]
    d = np.full((k, N), np.inf)
    while len(c) < k:
        i = len(c) - 1
        # get distance from dataset to latest center
        d[i] = np.sum((X-c[i])**2, 1)
        # get minimum distances
        min_dists = d[:i+1].min(0)
        # select next center with probability proportional to the inverse minimum distance
        selection_prob = np.cumsum(min_dists/min_dists.sum())
        r = random.uniform(0, 1)
        j = np.searchsorted(selection_prob, r)
        c.append(X[j])

    return np.array(c)

def gTrig(x, angi, D):
    '''
    Replaces angle dimensions with their complex representation.
    i.e. if x[i] is an angle ( i in angi ), then x[i] will be replaced
    with cos[x[i]] and sin[x[i]]
    '''
    Da = 2*len(angi)
    n = x.shape[0]
    xang = tt.zeros((n, Da))
    xi = x[:, angi]
    xang = tt.set_subtensor(xang[:, ::2], tt.sin(xi))
    xang = tt.set_subtensor(xang[:, 1::2], tt.cos(xi))

    non_angle_dims = list(set(range(D)).difference(angi))
    if non_angle_dims:
        xnang = x[:, non_angle_dims]
        m = tt.concatenate([xnang, xang], axis=1)
    else:
        m = xang
    return m

def gTrig2(m, v, angi, D):
    '''
    Replaces angle dimensions with their complex representation.
    i.e. if x[i] is an angle ( i in angi ), then x[i] will be replaced
    with cos[x[i]] and sin[x[i]].
    Since the input is a gaussian distribution, the output mean and covariance
    are computed via moment matching
    '''
    if len(angi) < 1:
        return m, v, None

    non_angle_dims = list(set(range(D)).difference(angi))
    Da = 2*len(angi)
    Dna = len(non_angle_dims)
    Ma = tt.zeros((Da,))
    Va = tt.zeros((Da, Da))
    Ca = tt.zeros((D, Da))

    # compute the mean
    mi = m[angi]
    vi = v[angi, :][:, angi]
    vii = v[angi, angi]
    exp_vii_h = tt.exp(-vii/2)

    Ma = tt.set_subtensor(Ma[::2], exp_vii_h*tt.sin(mi))
    Ma = tt.set_subtensor(Ma[1::2], exp_vii_h*tt.cos(mi))

    # compute the entries in the augmented covariance matrix
    vii_c = vii.dimshuffle(0, 'x')
    vii_r = vii.dimshuffle('x', 0)
    lq = -0.5*(vii_c+vii_r); q = tt.exp(lq)
    exp_lq_p_vi = tt.exp(lq+vi)
    exp_lq_m_vi = tt.exp(lq-vi)
    mi_c = mi.dimshuffle(0, 'x')
    mi_r = mi.dimshuffle('x', 0)
    U1 = (exp_lq_p_vi - q)*(tt.sin(mi_c-mi_r))
    U2 = (exp_lq_m_vi - q)*(tt.sin(mi_c+mi_r))
    U3 = (exp_lq_p_vi - q)*(tt.cos(mi_c-mi_r))
    U4 = (exp_lq_m_vi - q)*(tt.cos(mi_c+mi_r))

    Va = tt.set_subtensor(Va[::2, ::2], U3-U4)
    Va = tt.set_subtensor(Va[1::2, 1::2], U3+U4)
    U12 = U1+U2
    Va = tt.set_subtensor(Va[::2, 1::2], U12)
    Va = tt.set_subtensor(Va[1::2, ::2], U12.T)
    Va = 0.5*Va

    # inv times input output covariance
    Is = 2*np.arange(len(angi)); Ic = Is +1
    Ca = tt.set_subtensor(Ca[angi, Is], Ma[1::2])
    Ca = tt.set_subtensor(Ca[angi, Ic], -Ma[::2])

    # construct mean vectors ( non angle dimensions come first, then angle dimensions)
    Mna = m[non_angle_dims]
    M = tt.concatenate([Mna, Ma])

    # construct the corresponding covariance matrices
    # just the blocks for the non angle dimensions and the angle dimensions separately
    V = tt.zeros((Dna+Da, Dna+Da))
    Vna = v[non_angle_dims, :][:, non_angle_dims]
    V = tt.set_subtensor(V[:Dna, :Dna], Vna)
    V = tt.set_subtensor(V[Dna:, Dna:], Va)

    # fill in the cross covariances
    q = v.dot(Ca)[non_angle_dims, :]
    V = tt.set_subtensor(V[:Dna, Dna:], q)
    V = tt.set_subtensor(V[Dna:, :Dna], q.T)
    #V = tt.concatenate([tt.concatenate([Vna,q],axis=1),tt.concatenate([q.T,Va],axis=1)], axis=0)

    return [M, V, Ca]

def gTrig_np(x, angi):
    if type(x) is list:
        x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]
    D = x.shape[1]
    Da = 2*len(angi)
    n = x.shape[0]
    xang = np.zeros((n, Da))
    xi = x[:, angi]
    xang[:, ::2] = np.sin(xi)
    xang[:, 1::2] = np.cos(xi)

    non_angle_dims = list(set(range(D)).difference(angi))
    xnang = x[:, non_angle_dims]
    m = np.concatenate([xnang, xang], axis=1)

    return m

def gTrig2_np(m, v, angi, D):
    non_angle_dims = list(set(range(D)).difference(angi))
    Da = 2*len(angi)
    Dna = len(non_angle_dims)
    n = m.shape[0]
    Ma = np.zeros((n, Da))
    Va = np.zeros((n, Da, Da))
    Ca = np.zeros((n, D, Da))
    Is = 2*np.arange(len(angi)); Ic = Is +1

    # compute the mean
    mi = m[:, angi]
    vi = (v[:, angi, :][:, :, angi])
    vii = (v[:, angi, angi])
    exp_vii_h = np.exp(-vii/2)

    Ma[:, ::2] = exp_vii_h*np.sin(mi)
    Ma[:, 1::2] = exp_vii_h*np.cos(mi)

    # compute the entries in the augmented covariance matrix
    lq = -0.5*(vii[:, :, None] + vii[:, None, :]); q = np.exp(lq)
    exp_lq_p_vi = np.exp(lq+vi)
    exp_lq_m_vi = np.exp(lq-vi)
    U1 = (exp_lq_p_vi - q)*(np.sin(mi[:, :, None]-mi[:, None, :]))
    U2 = (exp_lq_m_vi - q)*(np.sin(mi[:, :, None]+mi[:, None, :]))
    U3 = (exp_lq_p_vi - q)*(np.cos(mi[:, :, None]-mi[:, None, :]))
    U4 = (exp_lq_m_vi - q)*(np.cos(mi[:, :, None]+mi[:, None, :]))

    Va[:, ::2, ::2] = U3-U4
    Va[:, 1::2, 1::2] = U3+U4
    Va[:, ::2, 1::2] = U1+U2
    Va[:, 1::2, ::2] = Va[:, ::2, 1::2].transpose(0, 2, 1)
    Va = 0.5*Va

    # inv times input output covariance
    Ca[:, angi, Is] = Ma[:, 1::2]
    Ca[:, angi, Ic] = -Ma[:, ::2]

    # construct mean vectors ( non angle dimensions come first, then angle dimensions)
    Mna = m[:, non_angle_dims]
    M = np.concatenate([Mna, Ma], axis=1)

    # construct the corresponding covariance matrices
    # (ust the blocks for the non angle dimensions and the angle dimensions separately
    V = np.zeros((n, Dna+Da, Dna+Da))
    Vna = v[:, non_angle_dims, :][:, :, non_angle_dims]
    V[:, :Dna, :Dna] = Vna
    V[:, Dna:, Dna:] = Va

    # fill in the cross covariances
    V[:, :Dna, Dna:] = (v[:, :, :, None]*Ca[:, :, None, :]).sum(1)[:, non_angle_dims, :]
    V[:, Dna:, :Dna] = V[:, :Dna, Dna:].transpose(0, 2, 1)

    return [M, V]

def get_compiled_gTrig(angi, D, derivs=True):
    m = tt.dvector('x')      # n_samples x idims
    v = tt.dmatrix('x_cov')  # n_samples x idims x idims

    gt = gTrig2(m, v, angi, D, derivs=derivs)
    return theano.function([m, v], gt)

def wrap_params(p_list):
    # flatten out and concatenate the parameters
    if type(p_list) is not list:
        p_list = [p_list]
    P = []
    for pi in p_list:
        # to deal with other types that do not implement the numpy array API.
        # TODO this will cause a GPU memory transfer!
        pi = np.array(pi.__array__())
        P.append(pi.flatten())
    P = np.concatenate(P)
    return P

def unwrap_params(P, parameter_shapes):
    # get the correct sizes for the parameters
    p = []
    i = 0
    for pshape in parameter_shapes:
        # get the number of elemebt for current parameter
        npi = reduce(lambda x, y: x*y, pshape) if len(pshape) > 0 else 1
        # select corresponding elements and reshape into appropriate shape
        p.append(P[i:i+npi].reshape(pshape))
        # set index to the beginning  of next parameter
        i += npi
    if len(p) == 1:
        p = p[0]
    return p

class MemoizeJac(object):
    def __init__(self, fun, args=()):
        self.fun = fun
        self.value, self.jac = None, None
        self.x = None
        self.args = tuple(args)

    def _compute(self, x, *args):
        self.x = np.asarray(x).copy()
        args += self.args
        self.value, self.jac = self.fun(x, *args)

    def __call__(self, x, *args):
        if self.value is not None and np.alltrue(x == self.x):
            return self.value
        else:
            self._compute(x, *args)
            return self.value

    def derivative(self, x, *args):
        if self.jac is not None and np.alltrue(x == self.x):
            return self.jac
        else:
            self._compute(x, *args)
            return self.jac

def integer_generator(i=0):
    while True:
        yield i
        i += 1

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def update_errorbar(errobj, x, y, y_error):
    # from http://stackoverflow.com/questions/25210723/matplotlib-set-data-for-errorbar-plot
    ln, (erry_top, erry_bot), (barsy,) = errobj
    ln.set_xdata(x)
    ln.set_ydata(y)
    x_base = x
    y_base = y

    yerr_top = y_base + y_error
    yerr_bot = y_base - y_error

    erry_top.set_xdata(x_base)
    erry_bot.set_xdata(x_base)
    erry_top.set_ydata(yerr_top)
    erry_bot.set_ydata(yerr_bot)

    new_segments_y = [np.array([[x, yt], [x,yb]]) for x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
    barsy.set_segments(new_segments_y)

def plot_results(learner, H=None):
    '''
    Plots learning results. One plot for cost vs learning iteration,
    one for cost vs timestep for last episode, and plots for each state dimension vs
    time step for the last episode.
    '''
    dt = learner.plant.dt
    x0 = np.array(learner.plant.x0)
    S0 = np.array(learner.plant.S0)
    if H is None:
        H = learner.H
    H_steps = int(np.ceil(H/dt))
    T_range = np.arange(0, H+dt, dt)
    cost = np.array(learner.experience.immediate_cost[-1])
    rollout_ = learner.rollout(x0, S0, H_steps, 1)
    states = np.array(learner.experience.states[-1])

    if hasattr(learner, 'trajectory_samples'):
        predicted_costs_means = np.array(rollout_[0])
        predicted_costs_vars = np.array(rollout_[1])
        predicted_trajectories = np.array(rollout_[2])
        # plot  cost of trajectories
        plt.figure('Cost of last run and Predicted cost')
        plt.gca().clear()
        plt.errorbar(T_range, predicted_costs_means,
                     color='b', yerr=2*np.sqrt(predicted_costs_vars))
        plt.plot(T_range, cost, color='g', linewidth=2)

        # plot all trajectories
        for d in xrange(x0.size):
            plt.figure('Last run vs Predicted rollout for state dimension %d'%(d))
            plt.gca().clear()
            for tr_d in predicted_trajectories:
                plt.plot(T_range, tr_d[:, d], color='b', alpha=0.3)
            plt.plot(T_range, states[:, d], color='g', linewidth=2)
    else:
        # plot last run cost vs predicted cost
        plt.figure('Cost of last run and Predicted cost')
        plt.gca().clear()

        # plot predictive distributions
        plt.errorbar(T_range, rollout_[0], yerr=2*np.sqrt(rollout_[1]))
        plt.plot(T_range, cost)
        print_with_stamp('Predicted value: [%f]'%(np.array(rollout_[0]).sum()), 'plot_results')
        predicted_means = np.array(rollout_[2])
        predicted_vars = np.array(rollout_[3])

        for d in xrange(x0.size):
            plt.figure('Last run vs Predicted rollout for state dimension %d'%(d))
            plt.gca().clear()
            plt.errorbar(T_range, predicted_means[:, d], yerr=2*np.sqrt(predicted_vars[:, d, d]))
            plt.plot(T_range, states[:, d])

    plt.figure('Total cost per learning iteration')
    plt.gca().clear()
    iters = []
    cost_sums = []
    n_random = 0
    for i in xrange(learner.experience.n_episodes()):
        if not learner.experience.policy_parameters[i]:
            iters.append(0)
            n_random += 1
        else:
            iters.append(i-n_random)
        cost_sums.append(np.array(learner.experience.immediate_cost[i]).sum())
    plt.plot(np.array(iters)+1, np.array(cost_sums))

    plt.show(False)
    plt.waitforbuttonpress(0.05)

def plot_and_save(learner, filename, H=None, target=None, output_folder=None):
    output_file = None
    output_folder = get_output_dir() if output_folder is None else output_folder
    output_file = os.path.abspath(os.path.join(output_folder, filename))
    with open(output_file[:-4] + ".csv", 'w') as f:
        writer = csv.writer(f)

        with PdfPages(output_file) as pdf:
            dt = learner.plant.dt
            x0 = np.array(learner.plant.x0)
            S0 = np.array(learner.plant.S0)
            if H is None:
                H = learner.H
            H_steps = int(np.ceil(H/dt))
            # plot last run cost vs predicted cost
            plt.figure('Cost of last run and Predicted cost')
            plt.gca().clear()
            T_range = np.arange(0, H+dt, dt)
            cost = np.array(learner.experience.immediate_cost[-1])
            rollout_ =  learner.rollout(x0, S0, H_steps, 1)
            plt.errorbar(T_range, rollout_[0], yerr=2*np.sqrt(rollout_[1]))
            plt.plot(T_range, cost)
            plt.title('Cost of last run and Predicted cost')

            header = ['T_range, actual cost, expected cost mean, expected cost error']
            writer.writerow(header)
            writer.writerow(T_range)
            writer.writerow(cost)
            writer.writerow(rollout_[0])
            writer.writerow(2*np.sqrt(rollout_[1]))
            endline = ['-------------------------------------------------------------']
            writer.writerow(endline)
            pdf.savefig()
            plt.close()

            states = np.array(learner.experience.states[-1])
            predicted_means = np.array(rollout_[2])
            predicted_vars = np.array(rollout_[3])

            for d in xrange(x0.size):
                plt.figure('Last run vs Predicted rollout for state dimension %d'%(d))
                plt.gca().clear()
                plt.errorbar(T_range, predicted_means[:, d], 
                             yerr=2*np.sqrt(predicted_vars[:, d, d]))
                plt.plot(T_range, states[:, d])
                if target:
                    plt.plot(T_range, [target[d]]*len(T_range))
                plt.title('Last run vs Predicted rollout for state dimension %d'%(d))
                header = ['T_range, actual state, expected state, expected state error. \
                (State dimension %d)'%(d)]
                writer.writerow(header)
                writer.writerow(T_range)
                writer.writerow(states[:, d])
                writer.writerow(predicted_means[:, d])
                writer.writerow(2*np.sqrt(predicted_vars[:, d, d]))
                endline = ['-------------------------------------------------------------']
                writer.writerow(endline)
                pdf.savefig()
                plt.close
            ep_nums = []
            ep_sums = []
            n_random = 0
            for i in xrange(learner.experience.n_episodes()):
                if learner.experience.policy_parameters[i]:
                    ep_nums.append(i-n_random)
                    total_c = 0.0
                    for c in learner.experience.immediate_cost[i]:
                        total_c += c[0]
                    ep_sums.append(total_c)
                else:
                    n_random += 1
            plt.figure('Total episode cost vs Iteration number')
            plt.gca().clear()
            plt.plot(np.array(ep_nums), np.array(ep_sums))
            plt.axis([0, ep_nums[-1], 0, max(ep_sums)])
            plt.title('Total episode cost vs Iteration number')
            header = ['Iteration number, Total episode Cost']
            writer.writerow(header)
            writer.writerow(ep_nums)
            writer.writerow(ep_sums)
            endline = ['-------------------------------------------------------------']
            writer.writerow(endline)
            pdf.savefig()
            plt.close()
    return output_file

def plot_learning_results(plant, H=None):
    ''' 
    Generates plots of the current learning run. Uses bokeh to dislpay the plots
    One plot for cost vs learning iteration,
    one for cost vs timestep for last episode, and plots for each state dimension vs
    time step for the last episode.
    '''
    dt = learner.plant.dt
    x0 = np.array(learner.plant.x0)
    S0 = np.array(learner.plant.S0)
    if H is None:
        H = learner.H
    H_steps = int(np.ceil(H/dt))
    T_range = np.arange(0, H+dt, dt)
    cost = np.array(learner.experience.immediate_cost[-1])
    rollout_ = learner.rollout(x0, S0, H_steps, 1)
    states = np.array(learner.experience.states[-1])

    if hasattr(learner, 'trajectory_samples'):
        predicted_costs_means = np.array(rollout_[0])
        predicted_costs_vars = np.array(rollout_[1])
        predicted_trajectories = np.array(rollout_[2])
        # plot  cost of trajectories
        plt.figure('Cost of last run and Predicted cost')
        plt.gca().clear()
        plt.errorbar(T_range, predicted_costs_means,
                     color='b', yerr=2*np.sqrt(predicted_costs_vars))
        plt.plot(T_range, cost, color='g', linewidth=2)

        # plot all trajectories
        for d in xrange(x0.size):
            plt.figure('Last run vs Predicted rollout for state dimension %d'%(d))
            plt.gca().clear()
            for tr_d in predicted_trajectories:
                plt.plot(T_range, tr_d[:, d], color='b', alpha=0.3)
            plt.plot(T_range, states[:, d], color='g', linewidth=2)
    else:
        # plot last run cost vs predicted cost
        plt.figure('Cost of last run and Predicted cost')
        plt.gca().clear()

        # plot predictive distributions
        plt.errorbar(T_range, rollout_[0], yerr=2*np.sqrt(rollout_[1]))
        plt.plot(T_range, cost)
        print_with_stamp('Predicted value: [%f]'%(np.array(rollout_[0]).sum()), 'plot_results')
        predicted_means = np.array(rollout_[2])
        predicted_vars = np.array(rollout_[3])

        for d in xrange(x0.size):
            plt.figure('Last run vs Predicted rollout for state dimension %d'%(d))
            plt.gca().clear()
            plt.errorbar(T_range, predicted_means[:, d], yerr=2*np.sqrt(predicted_vars[:, d, d]))
            plt.plot(T_range, states[:, d])

    plt.figure('Total cost per learning iteration')
    plt.gca().clear()
    iters = []
    cost_sums = []
    n_random = 0
    for i in xrange(learner.experience.n_episodes()):
        if not learner.experience.policy_parameters[i]:
            iters.append(0)
            n_random += 1
        else:
            iters.append(i-n_random)
        cost_sums.append(np.array(learner.experience.immediate_cost[i]).sum())
    plt.plot(np.array(iters)+1, np.array(cost_sums))

    plt.show(False)
    plt.waitforbuttonpress(0.05)

def get_logfile():
    ''' Returns the path of the file where the output of print_with_stamp wil be redirected. This can be set 
    via the $KUSANAGI_LOGFILE environment variable. If not set, it will return an empty string.'''

    if 'KUSANAGI_LOGFILE' in os.environ:
        return os.environ['KUSANAGI_LOGFILE']
    else:
        return ''

def get_run_output_dir():
    ''' Returns the current output folder for the last run results. This can be set via the $KUSANAGI_RUN_OUTPUT environment 
    variable. If not set, it will default to $HOME/.kusanagi/output/last_run. The directory will be created 
    by this method, if it does not exist.'''
    if not 'KUSANAGI_RUN_OUTPUT' in os.environ:
        os.environ['KUSANAGI_RUN_OUTPUT'] = os.path.join(get_output_dir(),"last_run")
    try: 
        os.makedirs(os.environ['KUSANAGI_RUN_OUTPUT'])
        chmod_cmd = 'chmod a+rwx -R ' + os.path.abspath(os.environ['KUSANAGI_RUN_OUTPUT'])
        os.system(chmod_cmd)
    except OSError:
        if not os.path.isdir(os.environ['KUSANAGI_RUN_OUTPUT']):
            raise
    return os.environ['KUSANAGI_RUN_OUTPUT']

def get_output_dir():
    ''' Returns the current output folder. This can be set via the $KUSANAGI_OUTPUT environment 
    variable. If not set, it will default to $HOME/.kusanagi/output. The directory will be created 
    by this method, if it does not exist.'''
    if not 'KUSANAGI_OUTPUT' in os.environ:
        homefolder = os.environ['HOME'] if 'HOME' in os.environ else os.environ['USERPROFILE']
        os.environ['KUSANAGI_OUTPUT'] = os.path.join(os.path.join(homefolder, ".kusanagi"), "output")
    try: 
        os.makedirs(os.environ['KUSANAGI_OUTPUT'])
        chmod_cmd = 'chmod a+rwx -R ' + os.path.abspath(os.environ['KUSANAGI_OUTPUT'])
        os.system(chmod_cmd)
    except OSError:
        if not os.path.isdir(os.environ['KUSANAGI_OUTPUT']):
            raise
    return os.environ['KUSANAGI_OUTPUT']

def set_logfile(new_path, base_path=None):
    ''' Sets the path of the log file. Assumes that new_path is well formed'''
    if base_path is None:
        os.environ['KUSANAGI_LOGFILE'] = new_path
    else:
        os.environ['KUSANAGI_LOGFILE'] = os.path.join(base_path,new_path)

def set_run_output_dir(new_path):
    ''' Sets the output directory for the files related to the current run. Assumes that new_path is well formed'''
    os.environ['KUSANAGI_RUN_OUTPUT'] = new_path

def set_output_dir(new_path):
    ''' Sets the output directory temporary files. Assumes that new_path is well formed'''
    os.environ['KUSANAGI_OUTPUT'] = new_path

def sync_output_filename(output_filename, obj_filename, suffix):
  if output_filename is None:
    output_filename = obj_filename+suffix
  else:
    obj_filename = output_filename
    # try removing suffix
    suffix_idx = obj_filename.find(suffix)
    if suffix_idx >= 0:
      obj_filename = obj_filename[:suffix_idx]
  return output_filename, obj_filename

def unzip_snapshot(zip_filepath, extract_path = ''):
  if not zip_filepath.lower().endswith('.zip'):
    zip_filepath += '.zip'
    
  with zipfile.ZipFile(zip_filepath, 'r') as myzip:
    myzip.extractall(extract_path)
    print_with_stamp('Extracted %s to %s'%(zip_filepath, os.path.abspath(extract_path)), 'Utils')

# creates zip of files: <snapshot_header>_<YYMMDD_HHMMSS.mmm>.zip
# if filename clash, will append _#
#
# Sample usage:
#   save_snapshot_zip('test', ['PILCO_GP_UI_Cartpole_RBFGP_sat.zip', 'PILCO_GP_UI_Cartpole_RBFGP_sat_dataset.zip', 'RBFGP_sat_5_1_cpu_float64.zip'])
def save_snapshot_zip(snapshot_header='snapshot', archived_files=[], with_timestamp=False):
  # Construct filename
  snapshot_filename = snapshot_header
  if with_timestamp:
    now = time.time()
    ms = now - math.floor(now)
    ms = math.floor(ms*1000)
    time_str = time.strftime('%y%m%d_%H%M%S')
    snapshot_filename='%s_%s.%03d' % (snapshot_header, time_str, int(ms))

  # Crash if snapshot file already exists
  repeat_counter = None
  if os.path.isfile(snapshot_filename+'.zip'):
    raise IOError('snapshot file %s already exists' % (snapshot_filename+'.zip'))

  # Save files
  with zipfile.ZipFile(snapshot_filename+'.zip', 'w') as myzip:
    for archived_filepath in archived_files:
      if os.path.isfile(archived_filepath):
        arcname = archived_filepath
        sep_idx = arcname.rfind(os.sep)
        if sep_idx >= 0:
          arcname = arcname[sep_idx+1:]
        myzip.write(archived_filepath, arcname)
      else:
        print_with_stamp('Snapshot cannot find %s'%(archived_filepath), 'Utils')
    myzip.close()
    print_with_stamp('Saved snapshot to %s.zip'%(snapshot_filename), 'Utils')
