import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from khrylib.utils import *
from khrylib.rl.utils.visualizer import Visualizer
from khrylib.rl.core.policy_gaussian import PolicyGaussian
from khrylib.rl.core.critic import Value
from khrylib.models.mlp import MLP
from motion_imitation.envs.humanoid_im import HumanoidEnv
from motion_imitation.utils.config import Config
import glfw
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='0202')
parser.add_argument('--vis_model_file', default='mocap_v2_vis')
parser.add_argument('--iter', type=int, default=800)
parser.add_argument('--focus', action='store_true', default=True)
parser.add_argument('--hide_expert', action='store_true', default=False)
parser.add_argument('--preview', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--record_expert', action='store_true', default=False)
parser.add_argument('--azimuth', type=float, default=45)
parser.add_argument('--video_dir', default='out/videos/motion_im')
args = parser.parse_args()
cfg = Config(args.cfg, False, create_dirs=False)
cfg.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
env = HumanoidEnv(cfg)
env.seed(cfg.seed)

actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""load learner policy"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
logger.info('loading model from checkpoint: %s' % cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
value_net.load_state_dict(model_cp['value_dict'])
running_state = model_cp['running_state']


class MyVisulizer(Visualizer):

    def __init__(self, vis_file):
        super().__init__(vis_file)
        ngeom = len(env.model.geom_rgba) - 1
        self.env_vis.model.geom_rgba[ngeom + 21: ngeom * 2 + 21] = np.array([0.7, 0.0, 0.0, 1])
        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = args.azimuth
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.T = 12

    def data_generator(self):
        while True:
            poses = {'pred': [], 'gt': []}
            vfs = []
            state = env.reset()
            if running_state is not None:
                state = running_state(state, update=False)
            
            
            for t in range(1000): 
                
                epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
                # print(epos.shape)
                if env.expert['meta']['cyclic']:
                    init_pos = env.expert['init_pos']
                    cycle_h = env.expert['cycle_relheading']
                    cycle_pos = env.expert['cycle_pos']
                    epos[:3] = quat_mul_vec(cycle_h, epos[:3] - init_pos) + cycle_pos
                    epos[3:7] = quaternion_multiply(cycle_h, epos[3:7])
                poses['gt'].append(epos) 
                poses['pred'].append(env.data.qpos.copy())
                print(t, 
                      env.get_contact_force()
                    #   env.get_end_effector_position("rfoot"),
                    #   env.get_ground_reaction_force()
                    )
                
                
                save_by_frame = False
                if save_by_frame:
                    fig, ax = env.visualize_by_frame(show = False) 
                    frame_dir = f'{args.video_dir}/frame_skeleton'
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    save_image_hwc(data,  f'{frame_dir}/%04d.png' % t) 
                    plt.close(fig)
                    
                print("*"*20)
                state_var = tensor(state, dtype=dtype).unsqueeze(0)
                action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                
                next_state, reward, done,info = env.step(action)
                vfs.append(np.hstack([env.data.qfrc_applied[:6].copy(), env.data.qfrc_actuator[6:].copy()])) 
                if running_state is not None:
                    next_state = running_state(next_state, update=False)
                if done:
                    print(f"fail: {info['fail']}")
                    break
                state = next_state

            poses['gt'] = np.vstack(poses['gt'])
            poses['pred'] = np.vstack(poses['pred'])
            plot_pose = False
            if plot_pose:
                fig, axs = plt.subplots(nrows=poses['gt'].shape[1]//4+1, ncols=4, figsize=(10, 12))
                fig, axs = visualize_poses(fig, axs, poses, env.body_qposaddr)
                plt.show()
            plot_torque = False
            if plot_torque:
                vfs = np.vstack(vfs)
                print("virtual force dim = ", vfs.shape)
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
                fig, axs = visualize_torques(fig, axs, vfs)
                plt.show()
            self.num_fr = poses['pred'].shape[0]
            
            contact_video = True
            if contact_video:
                out_name = f'{args.cfg}_{"expert" if args.record_expert else args.iter}_skeleton.mp4'
                frames_to_video(f'{args.video_dir}/frame_skeleton', args.video_dir, 5, out_name)
                
            yield poses

    def update_pose(self):
        self.env_vis.data.qpos[:env.model.nq] = self.data['pred'][self.fr]
        self.env_vis.data.qpos[env.model.nq:] = self.data['gt'][self.fr] 
        self.env_vis.data.qpos[env.model.nq] += 1.0 # time update
        if args.record_expert:
            self.env_vis.data.qpos[:env.model.nq] = self.data['gt'][self.fr]
        if args.hide_expert:
            self.env_vis.data.qpos[env.model.nq + 2] = 100.0
        if args.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
        self.env_vis.sim_forward()

    def record_video(self):
        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            for _ in range(200): 
                self.render()
            if not args.preview:
                t0 = time.time()
                # save_screen_shots(self.env_vis.viewer.window,f'{frame_dir}/%04d.png' % fr,  autogui = True)
                
                width, height = glfw.get_window_size(self.env_vis.viewer.window)
                data = self.env_vis._get_viewer("human").read_pixels(width, height, depth=False)
                save_image_hwc(data[::-1, :, [0,1,2]],  f'{frame_dir}/%04d.png' % fr)
                
                # img = self.env_vis.sim.render(width, height)
                # img = img.astype(np.uint8)
                # save_image_hwc(img,  f'{frame_dir}/{fr}.png')
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))

        if not args.preview:
            out_name = f'{args.video_dir}/{args.cfg}_{"expert" if args.record_expert else args.iter}.mp4'
            cmd = ['ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)
        
        

vis = MyVisulizer(f'{args.vis_model_file}.xml')

# if args.record:
#     vis.record_video()
# else:
#     vis.show_animation()
