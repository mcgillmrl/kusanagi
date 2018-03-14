# kusanagi

## Recommended way to install:

Install the Miniconda 3 distribution: https://conda.io/miniconda.html

    conda install numpy scipy mkl mkl-rt mkl-service jupyter
    conda install pygpu libgpuarray -c mila-udem
    pip install --upgrade git+https://github.com/Theano/Theano
    pip install --upgrade git+https://github.com/Lasagne/Lasagne
    pip install gym
    pip install mujoco-py
    cd <KUSANAGI_ROOT>
    pip install -e .

If you use the Miniconda 2 distribution (python 2.7.x), then you need to install gym 0.5.7
    pip install gym==0.5.7

## Example to reproduce some of the results:

    python examples/PILCO/cartpole_learn.py -e 8 -n mcpilco_lognormal_dropout -k mc_samples 100 -k max_evals 1000 -k learning_rate 1e-3 -k polyak_averaging None -r True -k clip_gradients 1.0

## Results on robot learning with ROS

See http://github.com/juancamilog/robot_learning for an example on how to integrate this library with ROS, and some sample results (from our IROS 2018 submission "[Synthesizing Neural Network Controllers with Probabilistic Model based Reinforcement Learning](http://www.cim.mcgill.ca/~gamboa/publications/iros_2018_model_based_rl.pdf)")
