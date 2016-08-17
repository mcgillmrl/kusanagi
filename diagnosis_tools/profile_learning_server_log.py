#!/usr/bin/env python

import sys
from datetime import datetime

# Print comma-separated timings: dt_all, dt_dyn, dt_pol, dt_end, t_ini, t_dyn, t_pol, t_end
def analyse_log(logfile):
  print '% ' + logfile
  print '%'
  print '% dt_all, dt_dyn, dt_pol, dt_end, t_ini, t_dyn, t_pol, t_end'
  nan = float('nan')
  epoch = datetime(1970,1,1)
  ini_t = nan
  dyn_t = nan
  pol_t = nan
  end_t = nan
  state = 'ini'
  with open(logfile, 'r') as f:
    for line in f.readlines():
      if len(line) < 29:
        continue
      if state == 'ini' and line.find('Experience > Initialising new experience dataset') >= 0:
        ini_t = (datetime.strptime(line[:28], '[%Y-%m-%d %H:%M:%S.%f]')-epoch).total_seconds()
        state = 'dyn'
      elif state == 'dyn' and line.find('MultiTaskLearnerServer > Done training dynamics model') >= 0:
        dyn_t = (datetime.strptime(line[:28], '[%Y-%m-%d %H:%M:%S.%f]')-epoch).total_seconds()
        state = 'pol'
      elif state == 'pol' and line.find('MultiTaskLearnerServer > Done training. New value') >= 0:
        pol_t = (datetime.strptime(line[:28], '[%Y-%m-%d %H:%M:%S.%f]')-epoch).total_seconds()
        state = 'end'
      elif state == 'end' and line.find('Utils > Saved snapshot to') >= 0:
        end_t = (datetime.strptime(line[:28], '[%Y-%m-%d %H:%M:%S.%f]')-epoch).total_seconds()
        state = 'ini'
        print '%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' % (end_t-ini_t, dyn_t-ini_t, pol_t-dyn_t, end_t-pol_t, ini_t, dyn_t, pol_t, end_t)
        ini_t = nan
        dyn_t = nan
        pol_t = nan
        end_t = nan

if __name__=='__main__':
  if len(sys.argv) != 2:
    print 'Usage: %s LOGFILE' % sys.argv[0]
    sys.exit(-1)
  else:
    analyse_log(sys.argv[1])
    sys.exit(0)
