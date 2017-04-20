#!/usr/bin/env python
import os,sys
import numpy as np
from ghost.learners.ExperienceDataset import ExperienceDataset

def print_exp(path):
  exp = ExperienceDataset(filename=path)
  K = len(exp.states)
  x0 = np.array([0,0,0,1,0,0,0])
  for k in range(K):
    diff = abs(x0 - exp.states[k][0])
    max_diff = max(diff)
    mean_diff = sum(diff)/len(x0)
    print('iter %02d: avg=%.4f max=%.4f' % (k, mean_diff, max_diff))
    # print '  data=', exp.states[k][0]

if __name__=='__main__':
  if not len(sys.argv) == 2:
    print('Usage: %s <PATH_TO_EXP_FILE>' % sys.argv[0])
  else:
    print_exp(sys.argv[1])
