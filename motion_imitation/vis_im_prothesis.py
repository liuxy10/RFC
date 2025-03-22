import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
sys.path.append(os.getcwd())

from khrylib.utils import *
from khrylib.rl.utils.visualizer import Visualizer
from khrylib.rl.core.policy_gaussian import PolicyGaussian
from khrylib.rl.core.critic import Value
from khrylib.models.mlp import MLP
from motion_imitation.envs.humanoid_im import HumanoidEnv
from motion_imitation.envs.humanoid_pk import HumanoidEnvProthesis
from motion_imitation.utils.config import Config 
import matplotlib.pyplot as plt
import glfw

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='0202')
parser.add_argument('--vis_model_file', default='mocap_v2_vis')
parser.add_argument('--iter', type=int, default=410)
parser.add_argument('--focus', action='store_true', default=True)
parser.add_argument('--hide_expert', action='store_true', default=False)
parser.add_argument('--preview', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--record_expert', action='store_true', default=False)
parser.add_argument('--azimuth', type=float, default=45)
parser.add_argument('--video_dir', default='out/videos/prothesis_hw')
args = parser.parse_args()

cfg = Config('0202', False, create_dirs=False)
cfg_p = Config('0202_prothesis', False, create_dirs=False)
cfg.env_start_first = True
cfg_p.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
env = HumanoidEnv(cfg)
env.seed(cfg.seed)

env_p = HumanoidEnvProthesis(cfg_p, cfg)
env_p.seed(cfg_p.seed)

actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actuators_f = env_p.model.actuator_names
state_dim_f = env_p.observation_space.shape[0]
action_dim_f = env_p.action_space.shape[0]

"""load learner policy"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
logger.info('loading model from checkpoint: %s' % cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
value_net.load_state_dict(model_cp['value_dict'])
running_state = model_cp['running_state']
# change based on osl default controller
phase_list = ['e_stance', 'l_stance', 'e_swing', 'l_swing']
joint_list = ['knee', 'ankle', 'threshold']
n_sets = env_p.OSL_CTRL.n_sets

# for phase_name in phase_list: 
#     env_p.OSL_CTRL.set_osl_param( phase_name, 'gain',  'ankle_stiffness', 0 , mode=0)


class MyVisulizerPK(Visualizer):

    def __init__(self, vis_file):
        super().__init__(vis_file)
        ngeom = len(env.model.geom_rgba) - 1
        self.env_vis.model.geom_rgba[ngeom + 21: ngeom * 2 + 21] = np.array([0.7, 0.0, 0.0, 1])
        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = 45
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.T = 12

    def data_generator(self):
        poses = {'pred': [], 'gt': []}
        osl_infos = { 'phase': [], 'osl_sense_data': {'knee_angle': [], 'knee_vel': [], 'ankle_angle': [], 'ankle_vel': [], 'load': []}}
        forces = []
        grfs = []
        state = env_p.reset()
        assert env_p.init_qpos_p[1] == env.expert['qpos'][0,1], "init_qpos[1] != expert_qpos[0,1]"
        assert np.allclose(env_p.init_qpos_p, env_p.data.qpos), "init_qpos[1] != qpos[0,1]"
        if running_state is not None:
            state = running_state(state, update=False)
        for t in range(1000): 
            print(t)
            epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
            if env.expert['meta']['cyclic']:
                init_pos = env.expert['init_pos']
                cycle_h = env.expert['cycle_relheading']
                cycle_pos = env.expert['cycle_pos']
                epos[:3] = quat_mul_vec(cycle_h, epos[:3] - init_pos) + cycle_pos 
                epos[3:7] = quaternion_multiply(cycle_h, epos[3:7])
            poses['gt'].append(epos) 
            qpos = env_p.data.qpos.copy()
            qpos = np.concatenate([qpos[:env.knee_qposaddr+1], np.zeros(2), qpos[env.knee_qposaddr+1:]])
            poses['pred'].append(qpos)
            
            save_by_frame = True
            if save_by_frame:
                lab = f"Prothesis (red) at {osl_infos['phase'][-1]}, enabled = {env_p.overwrite}" if len(osl_infos['phase']) > 0 else "Prothesis (red)" 
                fig, ax = env_p.visualize_by_frame(show = False, label = lab ) 
                vfs = env.data.qfrc_applied[:3].copy()
                com_root = env.data.subtree_com[0,:].copy() 
                visualize_3d_forces(fig, ax, vfs, com_root, sc = 20)
                # print osl_infos on the figure
                
                fig.canvas.draw()
                frame_dir = f'{args.video_dir}/frame_skeleton/'
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
                save_image_hwc(data,  f'{frame_dir}/%04d.png' % t) 
                plt.close(fig)
            # attain action from policy
            state_var = tensor(state, dtype=dtype).unsqueeze(0)
            action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
            next_state, reward, done, fail = env_p.step(action) # env_p also include the osl update
            # logging osl info
            osl_info = env_p.osl_info
            osl_infos['phase'].append(osl_info['phase'])
            # print(osl_infos['phase'])
            for key in osl_infos['osl_sense_data']:
                osl_infos['osl_sense_data'][key].append(osl_info['osl_sense_data'][key])
            # logging virtual force and grf info
            forces.append(env_p.data.actuator_force.copy())# env_p.data.qfrc_actuator.copy()[6:]) #np.hstack([env.data.qfrc_applied[:6].copy(), env.data.qfrc_actuator[6:].copy()]))   
            f, cop, f_m = env_p.get_ground_reaction_force()
            grfs.append(f)
            # logging osl sense data
            
            # filter state
            if running_state is not None:
                next_state = running_state(next_state, update=False)
            if done:
                break
            state = next_state
            # print("np.std(state) vs ",np.std(state), np.std(epos))

        poses['gt'] = np.vstack(poses['gt'])
        poses['pred'] = np.vstack(poses['pred'])
        plot_pose = False
        if plot_pose:
            fig, axs = plt.subplots(nrows=poses['gt'].shape[1]//4+1, ncols=4, figsize=(6, 12))
            fig, axs = visualize_poses(fig, axs, poses, env.body_qposaddr)
            plt.show() 
            
        plot_sensor = True
        # import matplotlib.pyplot as plt
        if plot_sensor:
            fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(7, 14))
            fig, axs = visualize_phases(fig, axs, osl_infos)
            plt.show()
        plot_torque = True
        if plot_torque:
            forces = np.vstack(forces)
            # print("virtual force dim = ", forces.shape)
            # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
            # fig, axs = visualize_torques(fig, axs, forces)
            visualize_force(forces, env_p.model.actuator_names)
            plt.show()
        self.num_fr = poses['pred'].shape[0]
        plot_grfs = False
        if plot_grfs:
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            fig, axs = visualize_grfs(fig, axs, grfs)
            plt.show()
        contact_video = True
        if contact_video:
            out_name = f'{args.cfg}_{"expert" if args.record_expert else args.iter}_skeleton.mp4'
            frames_to_video(f'{args.video_dir}/frame_skeleton', args.video_dir, 5, out_name)   
        yield poses

            

    def update_pose(self):
        self.env_vis.data.qpos[:env.model.nq] = self.data['pred'][self.fr]
        self.env_vis.data.qpos[env.model.nq:] = self.data['gt'][self.fr] 
        self.env_vis.data.qpos[env.model.nq] += 1.0 # time update
        # if args.record_expert:
        # self.env_vis.data.qpos[:env.model.nq] = self.data['gt'][self.fr]
        if True: #args.hide_expert:
            self.env_vis.data.qpos[env.model.nq + 2] = 100.0
        # if args.focus:
        #     self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
        self.env_vis.sim_forward()

    def record_video(self, preview = False):
        frame_dir = f'out/videos/prothesis/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            # print("hi")
            self.fr = fr
            self.update_pose()
            for _ in range(200): 
                self.render()
            if not preview:
                t0 = time.time()
                # save_screen_shots(self.env_vis.viewer.window,f'{frame_dir}/%04d.png' % fr,  autogui = True)
                
                width, height = glfw.get_window_size(self.env_vis.viewer.window)
                data = self.env_vis._get_viewer("human").read_pixels(width, height, depth=False)
                save_image_hwc(data[::-1, :, [0,1,2]],  f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))
         
        if preview:
            out_name = f'out/videos/prothesis/0202_800.mp4'
            cmd = ['ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)
            
            
vis = MyVisulizerPK(f'mocap_v2_vis.xml')

# vis.record_video(preview = False)
torch.cuda.empty_cache()
