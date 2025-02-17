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
        self.jkp, self.jkd = cfg.jkp, cfg.jkd  # inital value of joint stiffness, damping, reference position, scale
    
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
        # phase
        if self.cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))
            
        obs = np.concatenate(obs)
        return obs


    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(self.model.nv, self.model.nv)
        C = self.data.qfrc_bias.copy() 
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        # solve for acceleration based on LQR control
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), 
                            overwrite_b=True, 
                            check_finite=False) 
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        
        # TODO" k_p, k_d should be set to the updated joint stiffness and damping
        
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]
        return torque

    

    def reset_model(self):
        cfg = self.cfg
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


    



