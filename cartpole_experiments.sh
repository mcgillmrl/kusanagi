#!//usr/bin/env bash

OPTS="-r True"
EXTRA_OPTS="-k mc_samples 100 -k max_evals 1000 -k learning_rate 1e-3 -k polyak_averaging None -k clip_gradients 1.0 -k heteroscedastic_dyn True -k debug_plot 1 -k n_opt 50"

python examples/PILCO/cartpole_learn.py -e 1 -n pilco_ssgp_rbfp ${OPTS} ${EXTRA_OPTS}
python examples/PILCO/cartpole_learn.py -e 3 -n mcpilco_dropoutd_rbfp ${OPTS} ${EXTRA_OPTS}
python examples/PILCO/cartpole_learn.py -e 4 -n mcpilco_dropoutd_mlpp ${OPTS} ${EXTRA_OPTS}
python examples/PILCO/cartpole_learn.py -e 5 -n mcpilco_lndropoutd_rbfp ${OPTS} ${EXTRA_OPTS}
python examples/PILCO/cartpole_learn.py -e 6 -n mcpilco_lndropoutd_mlpp ${OPTS} ${EXTRA_OPTS}
python examples/PILCO/cartpole_learn.py -e 7 -n mcpilco_dropoutd_dropoutp ${OPTS} ${EXTRA_OPTS}
python examples/PILCO/cartpole_learn.py -e 8 -n mcpilco_dropoutd_lndropoutp ${OPTS} ${EXTRA_OPTS}
