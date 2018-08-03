#!//usr/bin/env bash

#OPTS="-r True -d 1 -o ${HOME}/.kusanagi/cp_results"
ODIR=${HOME}/.kusanagi/cp_results_final
OPTS="-o ${ODIR}"
EXPERIMENT_OPTS="-k n_opt 50 -k n_rnd 1 -k crn_dropout True -k crn True"
MCPILCO_OPTS="-k mc_samples 100 -k heteroscedastic_dyn True -k mm_state True -k mm_cost False -k noisy_policy_input True -k noisy_cost_input False"
OPTIMZER_OPTS="-k max_evals 1000 -k learning_rate 1e-3 -k polyak_averaging None -k clip_gradients 1.0"
python examples/PILCO/cartpole_learn.py -e 1 -n pilco_ssgp_rbfp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 3 -n mcpilco_dropoutd_rbfp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 4 -n mcpilco_dropoutd_mlpp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 5 -n mcpilco_lndropoutd_rbfp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 6 -n mcpilco_lndropoutd_mlpp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 7 -n mcpilco_dropoutd_dropoutp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 8 -n mcpilco_lndropoutd_dropoutp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}
python examples/PILCO/cartpole_learn.py -e 9 -n mcpilco_cdropoutd_dropoutp ${OPTS} ${EXPERIMENT_OPTS} ${MCPILCO_OPTS} ${OPTIMZER_OPTS}

EVAL_OPTS = '-e cartpole.Cartpole -c cartpole.cartpole_loss -k n_trials 20 -k last_iteration 50'
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/pilco_ssgp_rbfp_1 -p RBFPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_dropoutd_rbfp_3 -p RBFPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_dropoutd_mlpp_4 -p RBFPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_lndropoutd_rbfp_5 -p NNPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_lndropoutd_mlpp_6 -p NNPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_dropoutd_dropoutp_7 -p NNPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_lndropoutd_dropoutp_8 -p NNPolicy $EVAL_OPTS
python kusanagi/shell/evaluate_policy.py -d ${ODIR}/mcpilco_cdropoutd_dropoutp_9 -p NNPolicy $EVAL_OPTS
