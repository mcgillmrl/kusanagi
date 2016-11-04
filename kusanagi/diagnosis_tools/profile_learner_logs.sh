#!/usr/bin/env bash

if (($# < 1 )); then
  echo 'Usage:' $0 'KUSANAGI_OUTPUT_DIR'
  exit 1
fi

for logfile in ${1}/*.log; do
  ./profile_learner_log.py $logfile
done
