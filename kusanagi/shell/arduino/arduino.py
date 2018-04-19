
import numpy as np
import serial
import struct
from time import time, sleep
from kusanagi.shell import plant
from kusanagi import utils


try:
   input = raw_input
except NameError:
   pass


class SerialPlant(plant.Plant):
    metadata = {
        'render.modes': ['human']
    }    
    cmds = ['RESET_STATE', 'GET_STATE', 'APPLY_CONTROL', 'CMD_OK', 'STATE']
    cmds = dict(list(zip(cmds, [str(i) for i in range(len(cmds))])))

    def __init__(self, state_indices=None, maxU=None, loss_func=None,
                 baud_rate=4000000, port='/dev/ttyACM0',
                 name='SerialPlant', *args, **kwargs):
        self.__dict__.update(kwargs)
        super(SerialPlant, self).__init__(name=name, *args, **kwargs)
        self.loss_func = loss_func
        self.port = port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(self.port, self.baud_rate)
        self.state_indices = state_indices
        self.U_scaling = 1.0/np.array(maxU)
        self.t = -1
        self._reset(wait_for_user=False)

    def apply_control(self, u):
        if not self.serial.isOpen():
            self.serial.open()
        self.u = np.array(u, dtype=np.float64)
        if len(self.u.shape) < 2:
            self.u = self.u[:, None]
        if self.U_scaling is not None:
            self.u *= self.U_scaling
        if self.t < 0:
            self.state, self.t = self.state_from_serial()

        u_array = self.u.flatten().tolist()
        u_array.append(self.t+self.dt)
        u_string = ','.join([str(ui) for ui in u_array])  # TODO pack as binary
        self.serial.flushInput()
        self.serial.flushOutput()
        cmd = self.cmds['APPLY_CONTROL']+','+u_string+";"
        self.serial.write(cmd.encode())

    def _step(self, action):
        self.apply_control(action)
        if not self.serial.isOpen():
            self.serial.open()
        dt = self.dt
        t1 = self.t + dt
        while self.t < t1:
            self.state, self.t = self.state_from_serial()
        state, t = self.get_state()
        if self.loss_func is not None:
            cost = self.loss_func(np.array(self.state)[None, :])
        else:
            cost = 0
        return state, cost, False, dict(t=t)

    def state_from_serial(self):
        self.serial.flushInput()
        self.serial.write((self.cmds['GET_STATE']+";").encode())
        c = self.serial.read()
        buf = [c]
        tmp = (self.cmds['STATE']+',').encode()
        while buf != tmp:  # TODO timeout this loop
            c = self.serial.read()
            buf = buf[-1]+c
        buf = []
        res = []
        escaped = False
        while True:  # TODO timeout this loop
            c = self.serial.read()
            if not escaped:
                if c == b'/':
                    escaped = True
                    continue
                elif c == b',':
                    res.append(b''.join(buf))
                    buf = []
                    continue
                elif c == b';':
                    res.append(b''.join(buf))
                    buf = []
                    break
            buf.append(c)
            escaped = False
        res = np.array([struct.unpack('<d', ri) for ri in res]).flatten()
        if self.state_indices is not None:
            return res[self.state_indices], res[-1]
        else:
            return res[:-1], res[-1]

    def _reset(self, wait_for_user=True):
        if wait_for_user:
            msg = 'Please reset your plant to its initial state and hit Enter'
            utils.print_with_stamp(msg, self.name)
            input()
        if not self.serial.isOpen():
            self.serial.open()
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write((self.cmds['RESET_STATE']+";").encode())
        sleep(self.dt)
        self.state, self.t = self.state_from_serial()
        self.t = -1
        return self.state

    def stop(self):
        super(SerialPlant, self).stop()
        self.serial.close()