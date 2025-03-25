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
from motion_imitation.envs.humanoid_ia import HumanoidImpAwareEnv
from motion_imitation.utils.config import Config
import glfw

from motion_imitation.reward_function import reward_func
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='0202_wo_phase')
parser.add_argument('--vis_model_file', default='mocap_v2_vis')
parser.add_argument('--iter', type=int, default=375)
parser.add_argument('--focus', action='store_true', default=True)
parser.add_argument('--hide_expert', action='store_true', default=False)
parser.add_argument('--preview', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--record_expert', action='store_true', default=False)
parser.add_argument('--azimuth', type=float, default=45)
parser.add_argument('--imp_aware', action='store_true', default=False)
parser.add_argument('--video_dir', default='out/videos/normal_hw') # need to be manually switched
args = parser.parse_args()
cfg = Config(args.cfg, False, create_dirs=False)
cfg.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))
if args.imp_aware:
    args.video_dir = 'out/videos/ia'
    change_config_path_via_args(cfg, args.cfg, '_ia')
"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
env = HumanoidImpAwareEnv(cfg) if args.imp_aware else HumanoidEnv(cfg)
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

# reward functions
custom_reward = reward_func[cfg.reward_id]

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
            poses = {'pred': [], 'gt': [], 'target': []}
            torques, torques_osl, phases, grfs, jkps= [], [],[], {'l':[], 'r':[]}, []
            state = env.reset()
            if running_state is not None:
                state = running_state(state, update=False)
            action = env.action_space.sample()
            t = 0
            while True: 
            # for t in range(env.expert['len']): 
                if t in range(env.expert['len']): 
                    epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
                    # epos_old = epos # print(epos.shape)
                    if env.expert['meta']['cyclic']:
                        init_pos = env.expert['init_pos']
                        cycle_h = env.expert['cycle_relheading']
                        cycle_pos = env.expert['cycle_pos']
                        epos[:3] = quat_mul_vec(cycle_h, epos[:3] - init_pos) + cycle_pos
                        # vel = (epos[:3] - epos_old[:3])/env.model.opt.timestep
                        # print("vel", vel, "dt", env.model.opt.timestep)
                        # epos_old = epos.copy()
                        epos[3:7] = quaternion_multiply(cycle_h, epos[3:7])
                    poses['gt'].append(epos[7:].copy()) 
                poses['pred'].append(env.data.qpos[7:].copy())
                poses['target'].append(env.get_target_pose(action))
                print('com vel abs',  np.linalg.norm(env.data.qvel[:3]) ) #"dt low level ctrl", env.model.opt.timestep, "high level ctrl frames", env.frame_skip)
                # print('com displacement',  np.linalg.norm(env.data.qvel[:3]) * env.model.opt.timestep * env.frame_skip)

                save_by_frame = True #False
                if save_by_frame:
                    fig, ax = env.visualize_by_frame(show = False)
                    vfs = env.data.qfrc_applied[:3].copy()
                    com_root = env.data.subtree_com[0,:].copy() 
                    visualize_3d_forces(fig, ax, vfs, com_root, sc = 20)
                    
                    fig.canvas.draw()
                    frame_dir = f'{args.video_dir}/frame_skeleton'
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    save_image_hwc(data,  f'{frame_dir}/%04d.png' % t) 
                    plt.close(fig)
                    
                # print("*"*20)
                state_var = tensor(state, dtype=dtype).unsqueeze(0)
                action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                
                next_state, reward, done,info = env.step(action, nonstop=True)
                torques.append(env.data.ctrl.copy()/env.mass) # env.data.qfrc_actuator.copy()[6:]) #np.hstack([env.data.qfrc_applied[:6].copy(), env.data.qfrc_actuator[6:].copy()])) 
                if env.cfg.osl: 
                    torques_osl.append([env.osl.osl_info['osl_ctrl']['knee'] /env.mass,
                                        -env.osl.osl_info['osl_ctrl']['ankle'] /env.mass])
                    phases.append(env.osl.osl_info['phase'])
                # print(env.data.actuator_force.copy())
                # f, cop, f_m = env.get_ground_reaction_force()
                
                if running_state is not None:
                    next_state = running_state(next_state, update=False)
                if done or t > 180:
                    print(f"fail: {info['fail']}")
                    break
                state = next_state
                
                grf_r, _, _, grf_l, _, _ = env.get_grf_rl()
                grfs['l'].append(grf_l)
                grfs['r'].append(grf_r)
                jkps.append(env.jkp[env.lower_index[0]: env.lower_index[1]].copy())
                
                print(t, 
                    "grf_desired",env.grf_normalized[t], "|", 
                    "grf_current", np.array([grf_r[2],grf_l[2], grf_r[1],grf_l[1]]) /9.81 / env.mass,"|"
                    "rew", custom_reward(env, state, action, info)[1][-1], "|",  # reward in real time
                    env.expert['height_lb'] - env.data.qpos[2], # height difference 
                    )
                
                t += 1

            poses['gt'],  poses['pred'], poses['target'] = np.vstack(poses['gt']), np.vstack(poses['pred']), np.vstack(poses['target'])
            torques, torques_osl = np.vstack(torques), np.vstack(torques_osl)
            np.save("grf_r.npy", np.array(grfs['r']))
            np.save("grf_l.npy", np.array(grfs['l']))
            self.visualize_traj(poses, torques, torques_osl, phases, grfs, jkps)
                    
            contact_video = True
            if contact_video:
                out_name = f'{args.cfg}_{"expert" if args.record_expert else args.iter}_skeleton.mp4'
                frames_to_video(f'{args.video_dir}/frame_skeleton', args.video_dir, 5, out_name)
        
            yield poses

    def visualize_traj(self, poses, torques, torques_osl, phases, grfs, jkps):
        plot_pose, plot_impedance, plot_torque, plot_grfs = True, False, True, False

        if plot_pose:
            fig, axs = visualize_poses( poses, env.model.actuator_names, phases = phases if env.cfg.osl else None) 
            plt.show()
        if plot_impedance:
            jkps = np.vstack(jkps)
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            fig, axs = visualize_impedance(fig, axs, jkps)
            plt.show()
        if plot_torque:
            
            print("virtual force dim = ", torques.shape)
            if env.cfg.osl:
                visualize_force(torques, env.model.actuator_names, torques_osl, ['ltibia_x', 'lfoot_x'], phases)
            else:    
                visualize_force(torques, env.model.actuator_names)
        self.num_fr = poses['pred'].shape[0]
            
            # summarize grf stats
        grfs['l'] = np.vstack(grfs['l'])/env.mass/9.81 # normalize by mass
        grfs['r'] = np.vstack(grfs['r'])/env.mass/9.81 # normalize by mass
        mse_v = np.sqrt((np.mean((grfs['l'][:,2] - env.grf_normalized[:grfs['l'].shape[0],1])**2) + np.mean((grfs['r'][:,2] - env.grf_normalized[:grfs['l'].shape[0],0])**2))/2)
        mse_ap = np.sqrt((np.mean((grfs['l'][:,1] - env.grf_normalized[:grfs['l'].shape[0],3])**2) + np.mean((grfs['r'][:,1] - env.grf_normalized[:grfs['l'].shape[0],2])**2))/2)
            # print states
        print("mse_v", mse_v, "mse_ap", mse_ap)
            
        if plot_grfs:
                fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                axs[2].plot(env.grf_normalized[:grfs['l'].shape[0],0],'r:', label = 'right ideal') # vert right
                axs[2].plot(env.grf_normalized[:grfs['l'].shape[0],1],'b:',label = 'left ideal') # vert left
                axs[1].plot(env.grf_normalized[:grfs['l'].shape[0],2],'r:', label = 'right ideal') # ap right
                axs[1].plot(env.grf_normalized[:grfs['l'].shape[0],3],'b:',label = 'left ideal') # ap left

                    # print(np.vstack(grfs['l']).shape, grfs['l'][0].shape)
                fig, axs = visualize_grfs(fig, axs, grfs['l'],lab='left', color = 'b')
                fig, axs = visualize_grfs(fig, axs, grfs['r'],'right', color = 'r')
                plt.show()

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

    def record_video(self, skeleton = False):
        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            if skeleton: 
                fig, ax = self.env_vis.visualize_by_frame(show = False, label =  "expert")
                fig.canvas.draw()
                frame_dir = f'{args.video_dir}/frame_skeleton_exp'
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
                save_image_hwc(data,  f'{frame_dir}/%04d.png' % fr) 
            else:
                for _ in range(200): 
                    self.render()
                if not args.preview:
                    t0 = time.time()
                    # save_screen_shots(self.env_vis.viewer.window,f'{frame_dir}/%04d.png' % fr,  autogui = True)
                    
                    width, height = glfw.get_window_size(self.env_vis.viewer.window)
                    data = self.env_vis._get_viewer("human").read_pixels(width, height, depth=False)
                    save_image_hwc(data[::-1, :, [0,1,2]],  f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))

        if not args.preview:
            out_name = f'{args.video_dir}/{args.cfg}_{"expert" if args.record_expert else args.iter}.mp4'
            cmd = ['ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)
        
        

vis = MyVisulizer(f'{args.vis_model_file}.xml')
torch.cuda.empty_cache()
# vis.record_video(skeleton = True) # record ground truth skeleton
# if args.record:
#     vis.record_video()
# else:
#     vis.show_animation()
