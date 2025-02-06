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
from osl.control.myoosl_control import MyoOSLController

class HumanoidEnvFreezeKnee(HumanoidEnv):
    """ take out left knee and ankle degree of freedom, 
    but compatible with the normally trained policy """

    def __init__(self, cfg_f, cfg):
        self.cfg = cfg
        self.cfg_f = cfg_f  
        
        super().__init__(cfg_f)
        

        
    def load_expert(self):
        expert_qpos, expert_meta = pickle.load(open(self.cfg.expert_traj_file, "rb"))
        body_qposaddr = get_body_qposaddr(self.model)
        body_qposaddr_list_start_index = [idxs[0] for idxs in list(body_qposaddr.values())]
        self.knee_id = body_qposaddr_list_start_index[2] 
        expert_qpos_f = np.hstack([expert_qpos[:,:self.knee_id], expert_qpos[:,self.knee_id+4:] ]) 

        expert_meta_f = expert_meta.copy()
        self.expert = get_expert(expert_qpos_f , expert_meta_f, self)

    
    def get_full_obs(self):  # definition of observation for knee
        """
        Get the full observation for the humanoid.

        This function constructs the observation vector for the humanoid, which includes
        position, velocity, and optionally heading and phase information based on the configuration.

        Returns:
            np.ndarray: The observation vector containing:
                - qpos[2:] (np.ndarray): Position information excluding the first two elements.
                - qvel[:6] or qvel (np.ndarray): Velocity information, either root or full based on configuration.
        """
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
        qpos = np.hstack([qpos[:self.knee_id], np.zeros(4),qpos[self.knee_id:]]) # add the knee and ankle back as zeros
        qvel = np.hstack([qvel[:self.knee_id], np.zeros(4),qvel[self.knee_id:]]) # add the knee and ankle back as zeros
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
        cfg = self.cfg_f
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
        # return super().compute_torque(ctrl)
    
    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg_f
        for i in range(n_frames):
            ctrl = action.copy()
            ctrl = np.hstack([ctrl[:self.knee_id], ctrl[self.knee_id + 4:]]) # remove the knee and ankle back as zeros
            if cfg.action_type == 'position': # used action type is position
                torque = self.compute_torque(ctrl) 
            elif cfg.action_type == 'torque': 
                torque = ctrl * cfg.a_scale
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy() # vfs are at the last of action
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)

            self.sim.step()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0
            
    
    def get_ee_pos(self, transform): 
        # get end effector position in world frame
        data = self.data
        ee_name = ['lfemur', 'rfoot', 'lwrist', 'rwrist', 'head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)
    
    def reset_model(self): 
        cfg = self.cfg_f
        if self.expert is not None:
            ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.expert['len'])
            self.start_ind = ind
            init_pose = self.expert['qpos'][ind, :].copy()
            init_vel = self.expert['qvel'][ind, :].copy()
            self.init_qpos_f = init_pose.copy()
            self.init_qvel_f = init_vel.copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)# 
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.update_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()


     
        
   
       