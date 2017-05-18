import pybuller as pb
from shell import Plant

class BulletPlant(Plant):
    def __init__(self, params=None, x0=None, S0=None,
                 dt=0.01, noise=None, name='Plant',
                 angle_dims = []):
        '''
            Sets up a pybullet GUI client, loading a urdf, sdf or mjcf
            description for an articulated robot.
        '''
        super(BulletPlant, self).__init__(params, x0, S0, dt,
                                          noise, name, angle_dims)
        
        try:
            pb.connect(pb.GUI)
    
