# kusanagi

## Recommended way to install:

Install the Miniconda 3 distribution: https://conda.io/miniconda.html

    conda install numpy scipy mkl mkl-service jupyter
    conda install pygpu libgpuarray -c mila-udem
    pip install --upgrade git+https://github.com/Theano/Theano
    pip install --upgrade git+https://github.com/Lasagne/Lasagne
    pip install gym
    pip install mujoco-py
    cd <KUSANAGI_ROOT>
    pip install -e .



## Example to reproduce some of the results:

python examples/PILCO/cartpole_learn.py -e 8 -n mcpilco_lognormal_dropout -k mc_samples 100 -k max_evals 1000 -k learning_rate 1e-3 -k polyak_averaging None -r True -k clip_gradients 1.0
