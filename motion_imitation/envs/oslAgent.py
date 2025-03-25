import os
import sys
sys.path.append(os.getcwd())
from gym import spaces
from osl.control.myoosl_control import MyoOSLController
import numpy as np
class OSLAgent:
    def __init__(self, cfg, ctrl_joints, freeze_joints):
        self.cfg = cfg
        self.osl_param_set = 4
        self.setup_osl_controller(cfg.M)
        self.overwrite = False 
        self.ctrl_joints = ctrl_joints # joints that are controlled by OSL
        self.freeze_joints = freeze_joints # joints/dofs that are frozen by OSL  
        
    def setup_osl_controller(self,mass,init_state = 'l_swing'):
        # Initialize the OSL controller
        self.OSL_CTRL = MyoOSLController(mass, init_state=init_state, n_sets=self.osl_param_set)
        self.OSL_CTRL.start()

        # Define OSL-controlled joints
        self.osl_joints = ['knee', 'ankle']
        

    def update_osl_control(self, osl_sens_data):
        self.OSL_CTRL.update(osl_sens_data)
        osl_torques = self.OSL_CTRL.get_osl_torque()
        self.osl_info = {"osl_ctrl": osl_torques, 
                        "phase": self.OSL_CTRL.STATE_MACHINE.get_current_state.get_name(), 
                        "osl_sense_data": osl_sens_data}
        return osl_torques

    def change_osl_mode(self, mode=0):
        """
        Accessor function to activte a set of state machine variables
        """
        assert mode < 4
        self.OSL_CTRL.change_osl_mode(mode)

    def upload_osl_param(self, dict_of_dict):
        """
        Accessor function to upload full set of paramters to OSL leg
        """   
        assert len(dict_of_dict.keys()) <= 4   
        for idx in dict_of_dict.keys():
            self.OSL_CTRL.set_osl_param_batch(dict_of_dict[idx], mode=idx)

