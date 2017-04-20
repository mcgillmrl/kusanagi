import utils
from shell.cartpole import default_params
from shell.plant import SerialPlant
from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.control import RBFPolicy,AdjustedPolicy
from ghost.regression import GP
import numpy as np

def setup_cartpole():
    learner_params = default_params()
    learner_params['params']['use_empirical_x0'] = True
    learner_params['dynmodel_class'] = GP.SSGP_UI
    learner_params['params']['dynmodel']['n_inducing'] = 100
    return learner_params
 
def setup_serial_cartpole(serial_port='/dev/ttyACM0'):
    learner_params = setup_cartpole()
   
    learner_params['plant_class'] = SerialPlant
    learner_params['params']['plant']['maxU'] = np.array(learner_params['params']['policy']['maxU'])*0.75/0.4
    learner_params['params']['plant']['state_indices'] = [0,2,3,1]
    learner_params['params']['plant']['baud_rate'] = 4000000
    learner_params['params']['plant']['port'] = serial_port
    return learner_params

def setup_transfer(N=100, J=100, simulation= False,
                   source_dir='examples/learned_policies/cartpole_serial',
                   target_dir='examples/learned_policies/target_180g_run_1',
                   serial_port='/dev/ttyACM0'):
    # SOURCE DOMAIN 
    utils.set_output_dir(source_dir)
    # load source experience
    if simulation:
        source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_Cartpole_RBFPolicy_sat_dataset')
    else:
        source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_SerialPlant_RBFPolicy_sat_dataset')

    #load source policy
    source_policy = RBFPolicy(filename='RBFPolicy_sat_5_1_cpu_float64')

    # TARGET DOMAIN
    utils.set_output_dir(target_dir)
    target_params = setup_serial_cartpole(serial_port=serial_port)
    target_params['params']['max_evals'] = 125
    # policy
    target_params['dynmodel_class'] = GP.SSGP_UI
    target_params['invdynmodel_class'] = GP.GP_UI
    target_params['params']['invdynmodel'] = {}
    target_params['params']['invdynmodel']['max_evals'] = 1000
    target_params['policy_class'] = AdjustedPolicy
    target_params['params']['policy']['adjustment_model_class'] = GP.GP
    #target_params['params']['policy']['adjustment_model_class'] = control.RBFPolicy
    #target_params['params']['policy']['n_inducing'] = 20
    target_params['params']['policy']['sat_func'] = None # this is because we probably need bigger controls for heavier pendulums
    target_params['params']['policy']['max_evals'] = 5000
    target_params['params']['policy']['m0'] = np.zeros(source_policy.D+source_policy.E)
    target_params['params']['policy']['S0'] = 1e-2*np.eye(source_policy.E)

    # initialize target plant
    if not simulation:
        print(target_params['params']['policy']['maxU'])
        print(target_params['params']['plant']['maxU'])
    else:
        # TODO get these as command line arguments
        target_params['params']['plant']['params'] = {'l': 0.5, 'm': 1.5, 'M': 1.5, 'b': 0.1, 'g': 9.82}
        target_params['params']['cost']['pendulum_length'] = target_params['params']['plant']['params']['l']

    target_params['params']['source_policy'] = source_policy
    target_params['params']['source_experience'] = source_experience

    return target_params

def run_policy(learner,i,H=4.0):
    learner.policy.set_params(learner.experience.policy_parameters[i])
    learner.plant.reset_state()
    learner.apply_controller(H)


