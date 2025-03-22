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
sys.path.append("/home/xliu227/Github/human-model-generator/code/")
from write_xml import *

class HumanoidEnv(mujoco_env.MujocoEnv):

    def __init__(self, cfg):
        if cfg.H >0 and cfg.M > 0:
            assert cfg.mujoco_model_file in ['mocap_v2.xml', 'mocap_v3.xml', 'mocap_v2_prothesis.xml'] , f"The standard model file should be mocap_v2, instead it is {cfg.mujoco_model_file}"
            if not path.exists(cfg.mujoco_model_file):
                # try the default assets path
                fullpath = path.join(Path(__file__).parent.parent.parent, 'khrylib/assets/mujoco_models', path.basename(cfg.mujoco_model_file))
                if not path.exists(fullpath):
                    raise IOError("File %s does not exist" % fullpath)
            input_xml = fullpath
            output_height_xml = fullpath.replace('.xml', '_height_scaled.xml')
            output_xml = fullpath.replace('.xml', '_all_scacled.xml')
            scale_humanoid_model(input_xml, output_height_xml, cfg.H)
            assign_mass_inertia(cfg.M, linkMass, output_height_xml, output_xml)
            print("Scaled model is saved at ", output_xml)
            print(f"scaled height: {calculate_humanoid_height(output_xml)}")
            cfg.mujoco_model_file = output_xml
            scale_inertia = cfg.M /(30.9534) * (cfg.H / 1.4954 )**2 # scale inertia proportional to mass and the square of the height 
            cfg = scale_torque_related_params(cfg, scale_inertia)
            # exit(0)
        mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file, frame_skip = 15) # 15 is the frame skip 
        self.cfg = cfg
        self.set_cam_first = set()
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.body_qposaddr_list_start_index = [idxs[0] for idxs in list(self.body_qposaddr.values())]
        self.knee_qposaddr = self.body_qposaddr_list_start_index[2]
        try:
            self.knee_qveladdr = self.sim.model.get_joint_qvel_addr("ltibia_x")
        except:
            self.knee_qveladdr = self.sim.model.get_joint_qvel_addr("knee")
        self.jkp, self.jkd = cfg.jkp, cfg.jkd  # inital value of joint stiffness, damping, reference position, scale
        self.lower_index = [0, 14] # lower body index from yml config file
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.set_model_params()
        self.expert = None
        self.load_expert()
        self.set_spaces()
        self.body_tree = build_body_tree(cfg.mujoco_model_file)
        print(self.body_tree)
        self.lower_limb_names = ['root','lfemur', 'ltibia', 'lfoot', 'rfemur', 'rtibia', 'rfoot']
        self.max_vf = 30.0 # N
        self.grf_normalized = get_ideal_grf(total_idx = 100, rhs_index = [3,33,63], offset_period = 15, stance_period = 18) 
        self.mass = self.model.body_subtreemass[0]
    
    def load_expert(self):
        expert_qpos, expert_meta = pickle.load(open(self.cfg.expert_traj_file, "rb"))
        # print(expert_meta)
        # print(expert_qpos.shape)
        try:
            self.expert = get_expert(expert_qpos, expert_meta, self)
        except ValueError:
            self.expert = None
        
    def set_model_params(self):
        if self.cfg.action_type == 'torque' and hasattr(self.cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cfg.j_stiff
            self.model.dof_damping[6:] = self.cfg.j_damp

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
        self.action_dim = self.ndof + self.vf_dim
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self):
        if self.cfg.obs_type == 'full':
            obs = self.get_full_obs()
        return obs
    
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

    def get_ee_pos(self, transform): 
        # get end effector position in world frame
        data = self.data
        ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
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

    def get_body_quat(self):
        # get body quaternion
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.model.body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self):
        # get center of mass
        return self.data.subtree_com[0, :].copy()

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        """
        Compute desired acceleration given position and velocity errors
        :param qpos_err: position error
        :param qvel_err: velocity error
        :param k_p: position gain
        :param k_d: velocity gain
        """
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(self.model.nv, self.model.nv)
        C = self.data.qfrc_bias.copy() 
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        # solve for desired acceleration of all the joints (M qdd + C =τ)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), 
                            overwrite_b=True, 
                            check_finite=False) 
        return q_accel.squeeze()
    
    # lower level controller: impdance based control 
    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        target_pos = self.get_target_pose(ctrl)
        
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

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


    """ RFC-Explicit """
    def rfc_explicit(self, vf):
        qfrc = np.zeros_like(self.data.qfrc_applied)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            contact_point = vf[i*self.body_vf_dim: i*self.body_vf_dim + 3]
            force = vf[i*self.body_vf_dim + 3: i*self.body_vf_dim + 6] * self.cfg.residual_force_scale
            torque = vf[i*self.body_vf_dim + 6: i*self.body_vf_dim + 9] * self.cfg.residual_force_scale if self.cfg.residual_force_torque else np.zeros(3)
            contact_point = self.pos_body2world(body, contact_point) # 
            force = self.vec_body2world(body, force)
            torque = self.vec_body2world(body, torque)
            mjf.mj_applyFT(self.model, self.data, force, torque, contact_point, body_id, qfrc) # apply a Cartesian force and torque to a point on a body, and add the result to the vector mjData.qfrc_applied of all applied forces. Note that the function requires a pointer to this vector, because sometimes we want to add the result to a different vector.
        self.data.qfrc_applied[:] = qfrc

    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.data.qpos[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        self.vf = vf
        self.data.qfrc_applied[:vf.shape[0]] = vf # qfrc_applied is the residual force applied to the body
        
    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
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
            # print("ctrl torques vs vf", torque.tolist(), vf.tolist())
            # contact = self.get_contact()
            # print("contact = ", contact)
            self.sim.step()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0
    
    def step(self, a):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()
        # do simulation
        self.do_simulation(a, self.frame_skip) 
        self.cur_t += 1
        self.bquat = self.get_body_quat() 
        self.update_expert()
        # get obs
        head_pos = self.get_body_frame_position('head')
        reward = 1.0
        if cfg.env_term_body == 'head':
            fail = self.expert is not None and head_pos[2] < self.expert['head_height_lb'] - 0.1
        else:
            fail = self.expert is not None and self.data.qpos[2] < self.expert['height_lb'] - 0.5
        cyclic = self.expert['meta']['cyclic']
        end =  (cyclic and self.cur_t >= cfg.env_episode_len) or (not cyclic and self.cur_t + self.start_ind >= self.expert['len'] + cfg.env_expert_trail_steps)
        done = fail or end
        obs = self.get_obs()
        return obs, reward, done, {'fail': fail, 'end': end}

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

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.data.qpos[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def update_expert(self):
        expert = self.expert
        if expert['meta']['cyclic']:
            if self.cur_t == 0:
                expert['cycle_relheading'] = np.array([1, 0, 0, 0])
                expert['cycle_pos'] = expert['init_pos'].copy()
            elif self.get_expert_index(self.cur_t) == 0:
                expert['cycle_relheading'] = quaternion_multiply(get_heading_q(self.data.qpos[3:7]),
                                                              quaternion_inverse(expert['init_heading']))
                expert['cycle_pos'] = np.concatenate((self.data.qpos[:2], expert['init_pos'][[2]]))


    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        return ind / self.expert['len']

    def get_expert_index(self, t):
        return (self.start_ind + t) % self.expert['len'] \
                if self.expert['meta']['cyclic'] else min(self.start_ind + t, self.expert['len'] - 1)

    def get_expert_offset(self, t):
        if self.expert['meta']['cyclic']:
            n = (self.start_ind + t) // self.expert['len']
            offset = self.expert['meta']['cycle_offset'] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]
    
##############################################################################################################
### xinyi's code
##############################################################################################################

    def forward_kinematics(self, qpos):
        assert type(qpos) == np.ndarray, "Joint angles must be a numpy array"
        assert qpos.shape[0] == self.nq, "Joint angles must have the same length as the number of degrees of freedom"
        # Set the joint angles
        self.sim.data.qpos = qpos

        # Forward kinematics
        self.sim.forward()

        # Get the positions of all bodies in the body tree
        body_positions = {}
        for body_name in self.body_tree:
            try:
                body_pos = self.get_body_frame_position(body_name)
                body_positions[body_name] = body_pos
            except KeyError:
                pass

        return body_positions
    
    def get_body_com(self, body_name):
        body_id = self.sim.model.body_name2id(body_name)
        return self.sim.data.xipos[body_id] # data.body_xpos is the position of the body FRAME in world frame, which is joint position
    
    def get_body_mass(self, body_name):
        body_id = self.sim.model.body_name2id(body_name)
        return self.sim.model.body_mass[body_id] 
    def get_joint_qvel_addr(self, joint_name):
        return self.sim.model.get_joint_qvel_addr(joint_name)

    def get_contact_force(self):
        forces = []
        poss = []
        ids = []
        for contact_id in range(self.data.ncon): 
            contact = self.data.contact[contact_id]
            if contact.efc_address >= 0 and contact.dim > 1 and contact.pos[2] <= 0.1:
                assert contact.dim == 3, "contact force dimension should be 3"
                ids.append(self.model.body_id2name(self.model.geom_bodyid[contact.geom2]))
                # print(ids)
                mat = np.transpose(contact.frame.reshape(3, 3))
                force_local = np.zeros(6)
                mjf.mj_contactForce(self.sim.model, self.data, contact_id, force_local)
                force_global = (mat @ force_local[:3, None]).ravel().copy()
                    
                forces.append(force_global)
                poss.append(contact.pos.copy())
        return np.array(forces), np.array(poss), np.array(ids)

    def get_ground_reaction_force(self):
        forces, poss, ids = self.get_contact_force()
        # print(ids)
        force_sum, pos_sum, force_sum_magnitude = get_sum_force(forces, poss)
        return force_sum, pos_sum, force_sum_magnitude
    
    def get_grf_rl(self):
        forces, poss, ids = self.get_contact_force()
        # print( ids, ids == 'rfoot', ids == 'lfoot')
        (force_sum_r, pos_sum_r, force_sum_magnitude_r) = get_sum_force(forces[ids == 'rfoot'], poss[ids == 'rfoot']) if len(ids) > 0 else (np.zeros(3), None, 0)
        (force_sum_l, pos_sum_l, force_sum_magnitude_l) = get_sum_force(forces[ids == 'lfoot'], poss[ids == 'lfoot']) if len(ids) > 0 else (np.zeros(3), None, 0)
        # print(force_sum_l[2], force_sum_r[2])
        assert force_sum_r.shape == (3,) and force_sum_l.shape == (3,), "Force sum should be a 3D vector"
        return force_sum_r, pos_sum_r, force_sum_magnitude_r, force_sum_l, pos_sum_l, force_sum_magnitude_l
    
    def get_target_pose(self, ctrl):
        ctrl_joint = ctrl[:self.ndof] * self.cfg.a_scale
        base_pos = self.cfg.a_ref
        target_pos = base_pos + ctrl_joint
        return target_pos
    
    def get_applied_torque_generalized(self):
        return self.data.qfrc_applied[6:]
    
    def compute_global_force(self):
        # Get full constraint Jacobian (efc_J)
        J = self.data.efc_J.copy()  # Shape: (n_contacts, nv)
        
        # Compute pseudoinverse of J^T
        J_transpose = J.T  # Shape: (nv, n_contacts)
        J_transpose_pinv = np.linalg.pinv(J_transpose)
        
        # Solve for Cartesian forces
        F_cartesian = J_transpose_pinv @ self.data.qfrc_applied
        
        return F_cartesian
    
    
    def visualize_by_frame(self, show = False, label =  "normal", plot_lr = True):
        joint_pos = {n: self.get_body_frame_position(n) for n in self.model.body_names if n != "world"}
        body_com_pos = {n: [self.get_body_com(n), self.get_body_mass(n)] for n in self.model.body_names if n != "world"}
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 10))
        # forces, poss = self.get_contact_force()
        ax.view_init(elev=0, azim=180)  # Set the view to face the yz plane
        ax.set_title(label)
        # visualize residual force on the root

        f, cops, _ = self.get_ground_reaction_force()
        if len(f) > 0: 
            if plot_lr:
                fs_r, cop_r, fm_r, fs_l, cop_l, fm_l = self.get_grf_rl()
                visualize_3d_forces(fig, ax, fs_l, cop_l, sc = 500)
                visualize_3d_forces(fig, ax, fs_r, cop_r, sc = 500)
                
            else:
                visualize_3d_forces(fig, ax, f, cops)
        fig, ax = visualize_skeleton(fig, ax, joint_pos, self.body_tree, body_com_pos)
        

        if show:
            plt.show()
        return fig, ax
        
   
if __name__ == "__main__":
    from motion_imitation.utils.config import Config
    cfg = Config('0202', False, create_dirs=False)
    cfg_f = Config('0202_freeze', False, create_dirs=False)
    cfg_p = Config('0202_prothesis', False, create_dirs=False)
    cfg.env_start_first = True
    cfg_f.env_start_first = True
    cfg_p.env_start_first = True
    env = HumanoidEnv(cfg)
    # env_f = HumanoidEnvFreezeKnee(cfg_f, cfg)
    # env_p = HumanoidEnvProthesis(cfg_p, cfg)
    print("HumanoidEnv",env.observation_space, env.action_space)
    # print("HumanoidEnvFreezeKnee",env_f.observation_space, env_f.action_space)
    # print("HumanoidEnvProthesis", env_p.observation_space, env_p.action_space)
    
    # print(env.reset().shape)
    
    # print(env_p.reset().shape)
    # print(env_p.get_osl_sens())
    # print(env_p.model.sensor_names)
    
    # print(env_p.sim.data.get_sensor('ltouch'), env_p.sim.data.get_sensor('lforce'))
    # print(env_p.get_osl_sens())
    # print(env_p.action_space.high, env_p.action_space.low)
    
    # Access the contact forces

    



