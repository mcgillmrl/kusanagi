import numpy as np


def SEard(loghyp,
          log_snr=np.log(1000),
          log_ls=np.log(100),
          log_std=np.log(1),
          p=30):
    log_sf = loghyp[:, -2]
    log_sn = loghyp[:, -1]
    log_ll = loghyp[:, :-2]
    penalty = ((log_sf - log_sn)/log_snr)**p
    penalty += (((log_ll - log_std)/log_ls)**p).sum(1)
    return penalty
