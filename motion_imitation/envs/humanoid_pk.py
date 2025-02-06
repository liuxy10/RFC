import os
import sys
sys.path.append(os.getcwd())

from khrylib.rl.envs.common import mujoco_env
from gym import spaces
from khrylib.utils import *
from khrylib.utils.transformation import quaternion_from_euler
from motion_imitation.utils.tools import get_expert
import mujoco_py as mujoco
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
from motion_imitation.envs.humanoid_im import HumanoidEnv
from osl.control.myoosl_control import MyoOSLController


class HumanoidEnvProthesis (HumanoidEnv):
    def __init__(self, cfg_p, cfg):
        self.cfg = cfg
        self.cfg_p = cfg_p  
        self.normalize_action = False
        
        super().__init__(cfg_p)
        # OSL specific init
        self.osl_param_set = 4
        self.setup_osl_controller()
                
        
    def load_expert(self):
        expert_qpos, expert_meta = pickle.load(open(self.cfg.expert_traj_file, "rb"))
        body_qposaddr = get_body_qposaddr(self.model)
        body_qposaddr_list_start_index = [idxs[0] for idxs in list(body_qposaddr.values())]
        self.knee_qposaddr = body_qposaddr_list_start_index[2] 
        expert_qpos_p = np.hstack([expert_qpos[:,:self.knee_qposaddr+1], expert_qpos[:,self.knee_qposaddr+3:] ]) 

        expert_meta_p = expert_meta.copy()
        self.expert = get_expert(expert_qpos_p , expert_meta_p, self)
     
    def get_full_obs(self):  # definition of observation for knee

        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # print(f"[freez]dim of qpos, qvel = {   qpos.shape, qvel.shape}")
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cfg.obs_heading: # set to False
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading: # set to True
            qpos[3:7] = de_heading(qpos[3:7])
        qpos = np.hstack([qpos[:self.knee_qposaddr+1], np.zeros(2), qpos[self.knee_qposaddr+1:]]) # add ankle back as zeros
        qvel = np.hstack([qvel[:self.knee_qveladdr+1], np.zeros(2), qvel[self.knee_qveladdr+1:]]) # add ankle back as zeros
        obs.append(qpos[2:])
        # vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full': # set to True
            obs.append(qvel)

        if self.cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))
        obs = np.concatenate(obs)
        return obs

    def compute_torque(self, ctrl):
        cfg = self.cfg_p
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        k_p[6:] = cfg.jkp
        k_d[6:] = cfg.jkd
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]
        return torque

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg_p
        for i in range(n_frames):
            ctrl = action.copy()
            ctrl = np.hstack([ctrl[:self.knee_qveladdr+1], ctrl[self.knee_qveladdr + 3:]]) # remove the knee and ankle back as zeros
            if cfg.action_type == 'position': # used action type is position
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == 'torque':
                torque = ctrl * cfg.a_scale
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            
            osl_sens_data = self.get_osl_sens()
            self.OSL_CTRL.update(osl_sens_data)
            osl_torques = self.OSL_CTRL.get_osl_torque()
            self.osl_info = {"osl_ctrl": osl_torques, 
                        "phase": self.OSL_CTRL.STATE_MACHINE.get_current_state.get_name(), 
                        "osl_sense_data": osl_sens_data}
            overwrite = True
            if overwrite:
                torque_overwrite= self._overwrite_osl_actions(torque, osl_torques)
                self.data.ctrl[:] = torque_overwrite
            else:
                self.data.ctrl[:] = torque
        
            # print("test: disable the knee and ankle")
            # self.data.ctrl[:] = self._overwrite_osl_actions(torque, {"knee":0,"ankle":0})
            
            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy() # vfs are at the last of action
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                    # print("test: disable residual force")
                    # self.data.qfrc_applied[:] = 0.
                else:
                    self.rfc_explicit(vf)

            self.sim.step()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def reset_model(self): 
        cfg = self.cfg_p
        if self.expert is not None:
            ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.expert['len'])
            self.start_ind = ind
            init_pose = self.expert['qpos'][ind, :].copy()
            init_vel = self.expert['qvel'][ind, :].copy()
            self.init_qpos_p = init_pose.copy()
            self.init_qvel_p = init_vel.copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)# 
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.update_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()   

    
    """ OSL specific functions
    """
     
    def setup_osl_controller(self, init_state = 'l_swing'):
        # Initialize the OSL controller
        self.OSL_CTRL = MyoOSLController(np.sum(self.model.body_mass), init_state=init_state, n_sets=self.osl_param_set)
        self.OSL_CTRL.start()

        # Define OSL-controlled joints
        self.osl_joints = ['knee', 'ankle']
        
        # Adjust action space
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
  
    def _overwrite_osl_actions(self, humanoid_torques, osl_torques):
        
        torque_ow= humanoid_torques.copy()

        
        torque_ow[self.knee_qposaddr - 6] = osl_torques["knee"]
        torque_ow[self.knee_qposaddr+1 - 6] = osl_torques["ankle"]
        
       
        return torque_ow
    
    def upload_osl_param(self, dict_of_dict):
        """
        Accessor function to upload full set of paramters to OSL leg
        """   
        assert len(dict_of_dict.keys()) <= 4   
        for idx in dict_of_dict.keys():
            self.OSL_CTRL.set_osl_param_batch(dict_of_dict[idx], mode=idx)

    
    def get_osl_sens(self):

        osl_sens_data = {}
        
        osl_sens_data['knee_angle'] = self.sim.data.qpos[self.knee_qposaddr].copy()
        osl_sens_data['knee_vel'] = self.sim.data.qvel[self.knee_qveladdr].copy()
        osl_sens_data['ankle_angle'] = self.sim.data.qpos[self.knee_qposaddr +1].copy()
        osl_sens_data['ankle_vel'] = self.sim.data.qvel[self.knee_qveladdr +1].copy()
        # print(self.sim.data.get_sensor('lload'))
        grf, _, grf_mag = self.get_ground_reaction_force() # a choice here, whether use the magnitude or the z component
        osl_sens_data['load'] =  grf[2] #np.maximum(- self.get_sensor('lforce', 3).copy() [2], 0.0)  # magnitude
        # osl_sens_data['touch'] = np.sign(self.get_sensor('ltouch', 1).copy() [0] )# magnitude
   
        return osl_sens_data

    def change_osl_mode(self, mode=0):
        """
        Accessor function to activte a set of state machine variables
        """
        assert mode < 4
        self.OSL_CTRL.change_osl_mode(mode)

    def get_sensor(self, name, dimensions):
        i = self.sim.model.sensor_name2id(name)
        return self.sim.data.sensordata[i:i+dimensions]       
     
     
     
        
   
       