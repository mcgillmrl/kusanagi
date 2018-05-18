import argparse
import inspect
import numpy as np
import theano
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
from time import time
from theano import d3viz

from kusanagi.ghost import regression
from kusanagi import utils

np.set_printoptions(linewidth=500, precision=17, suppress=True)


def test_func1(X, ftype=0):
    if ftype == 0:
        ret = 100*np.exp(-0.05*(np.sum((X**2), 1)))*np.sin(2.5*X.sum(1))
    elif ftype == 1:
        ret = np.zeros((X.shape[0]))
        ret[X.max(1) > 0] = 1
        ret -= 0.5
        ret = 10*(ret + 0.1*np.sin(2.5*X.sum(1)))
    elif ftype == 2:
        a, b, c, d, e, f = 0.6, -1.8, -0.5, -0.5, 1.7, 0
        ret = a*np.sin(b*X.sum(1)+c) + d*np.sin(e*X.sum(1)+f)
    elif ftype == 3:
        ret = np.sin(X.sum(1)) + np.cos(X.sum(1))
    else:
        ret = X[:, 0]
    return ret


def build_dataset(idims=9, odims=6, angi=[], f=test_func1, n_train=500,
                  n_test=50, input_noise=0.01, output_noise=0.01, f_type=0,
                  rand_seed=None):
    if rand_seed is not None:
        np.random.seed(rand_seed)
    #  ================== train dataset ==================
    # sample training points
    x_train = 5*(np.random.rand(int(n_train/2), idims) - 1.25)
    x_train2 = 5*(np.random.rand(int(n_train/2), idims) + 1.25)
    x_train = np.concatenate([x_train, x_train2], axis=0)
    # generate the output at the training points
    y_train = np.empty((n_train, odims))
    for i in range(odims):
        y_train[:, i] = (i+1)*f(x_train, f_type)
        y_train[:, i] += output_noise*(np.random.randn(n_train))
    x_train = utils.gTrig_np(x_train, angi)

    #  ================== test  dataset ==================
    # generate testing points
    # kk = input_noise*convolve2d(
    #      np.array([[1,2,3,2,1]]),np.array([[1,2,3,2,1]]).T)/9.0;
    # s_test = convolve2d(np.eye(idims),kk,'same')
    s_test = input_noise*np.eye(idims)
    s_test = np.tile(s_test, (n_test, 1)).reshape(n_test, idims, idims)
    x_test = 75*(np.random.rand(n_test, idims) - 0.5)
    # generate the output at the test points
    y_test = np.empty((n_test, odims))
    for i in range(odims):
        y_test[:, i] = (i+1)*f(x_test, f_type)
    if len(angi) > 0:
        x_test, s_test = utils.gTrig2_np(x_test, s_test, angi, idims)

    return (x_train, y_train), (x_test, y_test, s_test)


def build_GP(idims=9, odims=6, reg_class='GP', profile=theano.config.profile):
    reg_classes = dict(inspect.getmembers(regression, inspect.isclass))
    gp = reg_classes[reg_class](idims=idims, odims=odims, profile=profile)
    return gp


def write_profile_files(gp):
    d3viz.d3viz(gp.dnlml, 'dnlml.html')
    d3viz.d3viz(gp.predict_fn, 'predict.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reg_class', nargs='?',
        help='the name of the regressor class (from kusanagi.ghost.regression). Default: GP_UI.',
        default='GP_UI')
    parser.add_argument(
        '--n_train', nargs='?', type=int,
        help='Number of training samples. Default: 500.', default=500)
    parser.add_argument(
        '--n_test', nargs='?', type=int,
        help='Number of testing samples. Default: 200', default=200)
    parser.add_argument(
        '--idims', nargs='?', type=int, help='Input dimensions. Default: 4',
        default=4)
    parser.add_argument(
        '--odims', nargs='?', type=int, help='Output dimensions. Default: 2',
        default=2)
    parser.add_argument(
        '--noise1', nargs='?', type=float,
        help='Measurement noise of training targets. Default: 0.01',
        default=0.01)
    parser.add_argument(
        '--noise2', nargs='?', type=float,
        help='Noise on test inputs. Default: 0.01', default=0.01)
    parser.add_argument(
        '--func', nargs='?', type=int, help='Test function to use (default 0)',
        default=0)
    args = parser.parse_args()

    idims = args.idims
    odims = args.odims
    n_train = args.n_train
    n_test = args.n_test
    utils.print_with_stamp("Building test dataset", 'main')
    train_dataset, test_dataset = build_dataset(
        idims=idims, odims=odims, n_train=n_train, n_test=n_test,
        output_noise=args.noise1, input_noise=args.noise2, f_type=args.func,
        rand_seed=31337)
    utils.print_with_stamp("Building regressor", 'main')
    gp = build_GP(
        idims, odims, reg_class=args.reg_class, profile=theano.config.profile)
    gp.load()
    kk = args.noise2*convolve2d(
        np.array([[1, 2, 3, 2, 1]]), np.array([[1, 2, 3, 2, 1]]).T)/9.0
    s_train = convolve2d(np.eye(idims), kk, 'same')
    # s_train = args.noise2*np.eye(idims)
    # s_train = np.tile(s_train, (n_train, 1)).reshape(n_train, idims, idims)
    # s_train = s_train*(1-np.exp(-0.25*train_dataset[0][:, 0]**2))

    # gp.set_dataset(train_dataset[0], train_dataset[1], s_train)
    gp.set_dataset(train_dataset[0], train_dataset[1])

    gp.train()
    gp.save()

    utils.print_with_stamp("Testing regressor", 'main')
    test_mX = test_dataset[0]
    test_sX = test_dataset[2]
    test_Y = test_dataset[1]
    errors = []
    probs = []
    preds = []
    times = []
    for i in range(n_test):
        start_time = time()
        ret = gp(test_mX[i], test_sX[i])
        cov = ret[1]
        idx = range(cov.shape[0])
        cov[idx, idx] -= np.minimum(cov[idx, idx].min()-1e-6, 1e-6)
        ret[1] = cov
        times.append(time()-start_time)
        preds.append(ret)
        print('============%04d============' % (i))
        print('Test Point:\n%s' % (test_mX[i]))
        print('Ground Truth:\n%s' % (test_Y[i]))
        print('Mean Prediction:\n%s' % (ret[0]))
        print('Prediction Covariance:\n%s' % (ret[1]))
        print('Input/Output Covariance:\n%s' % (ret[2]))
        errors.append(np.sqrt(((ret[0]-test_Y[i])**2).sum()))
        print('Error:\t%f' % (errors[-1]))
        probs.append(np.log(multivariate_normal.pdf(
            test_Y[i], mean=ret[0], cov=ret[1])))
        print('Log Probability of Ground Truth:\t%f' % (probs[-1]))

    errors = np.array(errors)
    probs = np.array(probs)
    times = np.array(times)
    print('=============================')
    print('Min/Max/Mean Prediction Error:\t %f / %f / %f' % (
        errors.min(), errors.max(), errors.mean()))
    print('Min/Max/Mean Log Probablity:\t %f / %f / %f' % (
        probs.min(), probs.max(), probs.mean()))
    print('Min/Max/Mean Time per eval:\t %f / %f / %f' % (
        times.min(), times.max(), times.mean()))

    if idims == 1 and odims == 1:
        # plot regression result
        idx = np.argsort(train_dataset[0][:, 0])
        Xtr = train_dataset[0][idx]
        Ytr = train_dataset[1][idx]

        idx = np.argsort(test_dataset[0][:, 0])
        Xts = test_dataset[0][idx]
        SXts = test_dataset[2][idx]
        Yts = test_dataset[1][idx]

        plt.figure()
        plt.scatter(Xtr, Ytr)
        plt.plot(Xts, Yts)

        Ypred, Yvar, Yio = list(zip(*preds))
        Ypred = np.concatenate(Ypred, axis=0)[idx]
        Yvar = np.concatenate(Yvar, axis=0)[idx]
        alpha = 0.5
        for i in range(4):
            alpha = alpha/2.0
            lower_bound = Ypred - i*np.sqrt(Yvar).squeeze()
            upper_bound = Ypred + i*np.sqrt(Yvar).squeeze()
            plt.fill_between(
                Xts.squeeze(), lower_bound, upper_bound, alpha=alpha)

        plt.plot(Xts, Ypred)

        # plot samples from NN
        if isinstance(gp, regression.BNN):
            import theano
            x = theano.tensor.matrix()
            y = gp.predict(
                x, return_samples=True,
                iid_per_eval=False, deterministic=False)
            fpred = theano.function([x], y, allow_input_downcast=True)
            if hasattr(gp, 'update'):
                gp.update(n_samples=10)
            rets = []
            for xt in Xts:
                ret = fpred(np.tile(xt, (10, 1)))
                rets.append(ret[0])
            rets = np.array(rets)
            plt.plot(Xts.squeeze(), rets.squeeze(), alpha=0.5)

        plt.show()
