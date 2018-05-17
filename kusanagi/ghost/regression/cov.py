import theano.tensor as tt
from kusanagi import utils


def SEard(hyp, X1, X2=None, all_pairs=True):
    ''' Squard exponential kernel with diagonal scaling matrix
        (one lengthscale per dimension)'''
    n = 1
    idims = 1
    if X1.ndim == 2:
        n, idims = X1.shape
    elif X2.ndim == 2:
        n, idims = X2.shape
    else:
        idims = X1.shape[0]

    sf2 = hyp[idims]**2
    if (not all_pairs) and (X1 is X2 or X2 is None):
        # all the distances are going to be zero
        K = tt.tile(sf2, (n,))
        return K

    ls2 = hyp[:idims]**2
    D = utils.maha(X1, X2, tt.diag(1.0/ls2),
                   all_pairs=all_pairs)
    K = sf2*tt.exp(-0.5*D)
    return K


def Noise(hyp, X1, X2=None, all_pairs=True):
    ''' Noise kernel. Takes as an input a distance matrix D
    and creates a new matrix as Kij = sn2 if Dij == 0 else 0'''
    if X2 is None:
        X2 = X1

    sn2 = hyp**2
    if all_pairs and X1 is X2:
        # D = (X1[:,None,:] - X2[None,:,:]).sum(2)
        K = tt.eye(X1.shape[0])*sn2
        return K
    else:
        # D = (X1 - X2).sum(1)
        if X1 is X2:
            K = tt.ones((X1.shape[0],))*sn2
        else:
            K = 0
        return K

    # K = tt.eq(D,0)*sn2
    # return K


def Sum(hyp_l, cov_l, X1, X2=None, all_pairs=True):
    ''' Returns the sum of multiple covariance functions'''
    K = sum([cov_l[i](hyp_l[i], X1, X2, all_pairs=all_pairs)
             for i in range(len(cov_l))])
    return K
