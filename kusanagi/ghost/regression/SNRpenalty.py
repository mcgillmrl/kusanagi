import numpy as np

def SEard( loghyp, params={'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': np.log(1), 'p': 30}):
    p = np.asarray(params['p'],loghyp.dtype)
    log_snr = np.asarray(params['log_snr'],loghyp.dtype)
    log_std = np.asarray(params['log_std'],loghyp.dtype)
    log_ls = np.asarray(params['log_ls'],loghyp.dtype)

    log_sf = loghyp[:,-2]
    log_sn = loghyp[:,-1]
    log_ll = loghyp[:,:-2]
    penalty = ((log_sf - log_sn)/log_snr)**p
    penalty += (((log_ll - log_std)/log_ls)**p).sum(1)
    return penalty
