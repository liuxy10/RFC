import os
import sys
sys.path.append(os.getcwd())

from khrylib.rl.envs.common import mujoco_env

from motion_imitation.envs.humanoid_im import HumanoidEnv
from gym import spaces
from khrylib.utils import *
from khrylib.utils.transformation import quaternion_from_euler
from motion_imitation.utils.tools import get_expert
import mujoco_py as mujoco
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor

class HumanoidImpAwareEnv(HumanoidEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        

    def set_spaces(self):
        cfg = self.cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6 
            else:
                if cfg.residual_force_bodies == 'all':
                    self.vf_bodies = self.model.body_names[1:]
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)
        self.action_dim = self.ndof + self.vf_dim + (self.lower_index[1] - self.lower_index[0])
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def get_full_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # print(f"dim of qpos, qvel = {qpos.shape, qvel.shape}")
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()# transform velocity to root frame by default, or heading frame if specified
        obs = []
        # pos
        if self.cfg.obs_heading: # set to False
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading: # set to True
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:]) # why? because x,y position is not useful, so start from z(2)
        #  vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full': # set to True
            obs.append(qvel)
        # phase # todo: change it to gait phase
        if self.cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))
        # joint impedance
        obs.append(np.concatenate((self.jkp[self.lower_index[0]: self.lower_index[1]], 
                                   self.jkd[self.lower_index[0]: self.lower_index[1]])))
        obs = np.concatenate(obs)
        return obs

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        target_pos = self.get_target_pose(ctrl)
        
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        k_p[6:] = self.jkp.copy()
        k_d[6:] = self.jkd.copy()
        
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = - k_p[6:] * qpos_err[6:] - k_d[6:] * qvel_err[6:]
        return torque
    
    
    def update_impedance(self, ctrl):
        self.jkp[self.lower_index[0]: self.lower_index[1]] += ctrl[-(self.lower_index[1] - self.lower_index[0]):] * 0.1 # learning rate can be changed as well
        # print(self.jkp[self.lower_index[0]: self.lower_index[1]])
        self.jkp[self.lower_index[0]: self.lower_index[1]] = np.clip(self.jkp[self.lower_index[0]: self.lower_index[1]], 0, 500) 

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
            self.update_impedance(ctrl.copy())
            if cfg.action_type == 'position': # used action type is position
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == 'torque':
                torque = ctrl * cfg.a_scale
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy() # vfs are at the last of action
                if cfg.residual_force_mode == 'implicit': # True
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)
            self.sim.step()
    

   
if __name__ == "__main__":
    from motion_imitation.utils.config import Config
    cfg = Config('0202', False, create_dirs=False)
    cfg_f = Config('0202_freeze', False, create_dirs=False)
    cfg_p = Config('0202_prothesis', False, create_dirs=False)
    cfg.env_start_first = True
    cfg_f.env_start_first = True
    cfg_p.env_start_first = True
    env = HumanoidEnv(cfg)
    print("HumanoidEnv",env.observation_space, env.action_space)


    



