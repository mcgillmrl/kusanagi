#!/usr/bin/env python

import
import sys
from datetime import datetime

# Print comma-separated timings: dt_all, dt_dyn, dt_pol, dt_end, dt_nxt, t_ini, t_dyn, t_pol, t_end
def analyse_log(logpath):
  filename = os.path.basename(logpath)
  dirname  = os.path.dirname(logpath)
  basename = filename
  if basename.endswith('.log'):
    basename = basename[:-4]
  csvname  = basename + '_timings.csv'
  pngname  = basename + '_timings.png'
  csvpath  = os.path.abspath(os.path.join(dirname, csvname))
  pngpath  = os.path.abspath(os.path.join(dirname, pngname))
  
  print '- Analyzing timings: %s' % (logpath)
  
  with open(csvpath, 'w') as csvfile:
    csvfile.write('% ' + logpath + '\n')
    csvfile.write('%\n')
    csvfile.write('% dt_all, dt_dyn, dt_pol, dt_end, dt_nxt, t_ini, t_dyn, t_pol, t_end\n')
    csvlines = 3
    nan = float('nan')
    epoch = datetime(1970,1,1)
    ini_t = nan
    dyn_t = nan
    pol_t = nan
    end_t = nan
    state = 'ini'
    with open(logpath, 'r') as logfile:
      for line in logfile.readlines():
        if len(line) < 29:
          continue
        if state == 'ini' and line.find('Experience > Initialising new experience dataset') >= 0:
          new_ini_t = (datetime.strptime(line[:28], '[%Y-%m-%d %H:%M:%S.%f]')-epoch).total_seconds()
          csvfile.write('%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n' % (end_t-ini_t, dyn_t-ini_t, pol_t-dyn_t, end_t-pol_t, new_ini_t-end_t, ini_t, dyn_t, pol_t, end_t))
          csvlines += 1
          ini_t = new_ini_t
          dyn_t = nan
          pol_t = nan
          end_t = nan
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
    if state != 'dyn': # didn't print last line yet
      new_ini_t = end_t
      csvfile.write('%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n' % (end_t-ini_t, dyn_t-ini_t, pol_t-dyn_t, end_t-pol_t, new_ini_t-end_t, ini_t, dyn_t, pol_t, end_t))
      csvlines += 1
  
  print '- Wrote %d lines to %s' % (csvlines, csvpath)
  
  print '- Generating plot: %s' % (pngpath)
  plot_cmd = 'start matlab -nosplash -nodesktop -nojvm -r "plot_learning_server_profile(%s, %s, 1, %s); exit"' % (basename, csvpath, pngpath)
  print '> ' + plot_cmd
  os.system(plot_cmd)
  if os.path.isfile(pngpath):
    print '- Wrote to %s' % plot_cmd
  else:
    print '! Plotting command unsuccessful; debug by hand'
  
  print '- ALL DONE'

if __name__=='__main__':
  if len(sys.argv) != 2:
    print 'Usage: %s LOGPATH' % sys.argv[0]
    sys.exit(-1)
  else:
    analyse_log(sys.argv[1])
    sys.exit(0)
