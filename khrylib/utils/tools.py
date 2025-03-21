import numpy as np
import os
import shutil
import datetime
import subprocess
from os import path
from PIL import Image
from khrylib.utils.math import *
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import re


from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def generate_interpolated_grf(t):
    # Define key timing points and force values for vertical and AP forces
    time_points = [0, 15, 50, 85, 100]  # Percentage of stance phase
    force_vert_points = [0, 1.0, 0.8, 1.0, 0]  # Normalized to body weight
    force_ap_points = [0, -0.2, 0, 0.2, 0]  # Example AP force values

    # Create cubic spline interpolation for vertical and AP forces
    cs_vert = CubicSpline(time_points, force_vert_points, bc_type='natural')
    cs_ap = CubicSpline(time_points, force_ap_points, bc_type='natural')
    
    return cs_vert(t), cs_ap(t)

# hard coded for 0202
def get_ideal_grf(total_idx = 76, rhs_index = [0,30,60], offset_period = 15, stance_period = 18):
    grf = np.zeros((total_idx, 2))
    grf_ap = np.zeros((total_idx, 2))

    # for 1 row (right), 0 row (left)
    for i in rhs_index: 
        if i + stance_period > total_idx:
            grf[i:, 1], grf_ap[i:, 1] = generate_interpolated_grf(np.linspace(0, 100, total_idx-i))[:total_idx-i]
            continue
    
        if i + stance_period + offset_period > total_idx:
            grf[i:, 0], grf_ap[i:, 1] = generate_interpolated_grf(np.linspace(0, 100, total_idx-i - offset_period))[:total_idx-i-offset_period]
            continue
        
        grf [i:i+stance_period,1], grf_ap[i:i+stance_period,1] = generate_interpolated_grf(np.linspace(0, 100, stance_period))  # High-resolution time point
        grf [i+offset_period:i+stance_period+offset_period, 0], grf_ap[i+offset_period:i+stance_period+offset_period, 0] = generate_interpolated_grf(np.linspace(0, 100, stance_period))
        
    return np.hstack([grf, -grf_ap])

def change_config_path_via_args(cfg, cfg_num, postdix = ''):
    cfg.cfg_dir = '%s/motion_im%s/%s' % (cfg.base_dir, postdix, cfg_num)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.video_dir = '%s/videos' % cfg.cfg_dir
    cfg.tb_dir = '%s/motion_im%s/%s/tb' % (cfg.base_dir, postdix, cfg_num)

def get_sum_force(forces, poss):
    forces = np.array(forces) if len(forces) > 0 else np.zeros((1,3))
    poss = np.array(poss) if len(poss) > 0 else np.zeros((1,3))
    total_force = np.sum(forces, axis=0) 
    total_force_magnitude = np.linalg.norm(total_force)
    
    if total_force_magnitude == 0:
        cop = np.zeros(3)
        total_force = np.zeros(3)
    else:
        cop = np.sum(poss * np.linalg.norm(forces, axis=1)[:, np.newaxis], axis=0) / total_force_magnitude # TODO: CHANGE THIS TO BE INNER PRODUCT OF POS AND FORCE
    assert total_force.shape == (3,), f"Total force shape is {total_force.shape}"
    assert cop.shape == (3,), f"COP shape is {cop.shape}"
    assert total_force_magnitude.shape == (), f"Total force magnitude shape is {total_force_magnitude.shape}"
    return total_force, cop, total_force_magnitude

   
def visualize_grfs(fig, axs, grfs, lab = '', color = 'r'):
    if type(grfs) is list:
        grfs = np.array(grfs)
    label = 'xyz'
    for i in range(3):
        axs[i].plot(grfs[:,i], color+"-",label = f'GRF {label[i]} {lab}')
        axs[i].set_ylabel(f'GRF_{label[i]}/BW')
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    return fig, axs

def visualize_qpos(qpos, body_qposaddr_list_start_index, body_qposaddr):
    fig, axs = plt.subplots(nrows=qpos.shape[1]//4+1, ncols=4, figsize=(10, 12))
    for i in range(qpos.shape[1]//4+1):
        for j in range(4):
            idx = i*4 + j
            if idx >= qpos.shape[1]:
                break
            gt = qpos[:, idx]
            axs[i, j].plot(gt, 'r', label='gt')
            axs[i, j].set_ylim([-np.pi, np.pi])
            if idx in [idxs[0] for idxs in list(body_qposaddr.values())]:
                body_name = [name for name, addr in body_qposaddr.items() if addr[0] == idx][0]
                axs[i, j].set_title(f"idx = {idx}, {body_name}", fontsize=12)
            if i == 0 and j == 0:
                axs[i, j].legend()
    plt.tight_layout()
    plt.show()
    
def visualize_qvel(qvel, body_qveladdr_list_start_index, body_qveladdr):
    fig, axs = plt.subplots(nrows=qvel.shape[1]//4+1, ncols=4, figsize=(10, 12))
    for i in range(qvel.shape[1]//4+1):
        for j in range(4):
            idx = i*4 + j
            if idx >= qvel.shape[1]:
                break
            gt = qvel[:, idx]
            axs[i, j].plot(gt, 'r', label='gt')
            axs[i, j].set_ylim([-np.pi, np.pi])
            if idx in [idxs[0] for idxs in list(body_qveladdr.values())]:
                body_name = [name for name, addr in body_qveladdr.items() if addr[0] == idx][0]
                axs[i, j].set_title(f"idx = {idx}, {body_name}", fontsize=12)
            if i == 0 and j == 0:
                axs[i, j].legend()
    plt.tight_layout()
    plt.show()
    
def visualize_phases(fig, axs, osl_infos):
    phases = ['e_stance', 'l_stance', 'e_swing', 'l_swing']
    phase_colors = {'e_stance': 'r', 'l_stance': 'g', 'e_swing': 'b', 'l_swing': 'y'}
    phase_data = osl_infos['phase']
    
    phase_values = {'e_stance': 1, 'l_stance': 2, 'e_swing': 3, 'l_swing': 4}
    phase_line = [phase_values[phase] for phase in phase_data]
    axs[0].plot(phase_line, label='phase change', color='black',  marker='o')
    axs[0].legend()
    
    sensor_names = ['knee_angle', 'knee_vel', 'ankle_angle', 'ankle_vel', 'load']
    for i, name in enumerate(sensor_names):
        if 'ankle' in name or 'knee' in name:
            axs[i+1].plot(np.rad2deg(osl_infos['osl_sense_data'][name]), marker='o', label=name)
        else:
            axs[i+1].plot(osl_infos['osl_sense_data'][name], marker='o', label=name)
        axs[i+1].set_title(name)
        axs[i+1].set_ylabel('deg' if 'angle' in name else 'deg/s' if 'vel' in name else 'N')
        axs[i+1].legend() 
    plt.tight_layout()
    return fig, axs

def visualize_poses(poses, joint_names): 
    
    num_poses = poses['gt'].shape[1]
    num_cols = 4
    num_rows = num_poses // num_cols + (num_poses % num_cols > 0)
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 3))
    axs = axs.flatten()
    for i, (force, name) in enumerate(zip(poses['pred'].T, joint_names)):
        axs[i].plot(force, 'r',label=name + ' pred')
    for i, (force, name) in enumerate(zip(poses['gt'].T, joint_names)):
        axs[i].plot(force, 'b', label=name + ' gt')
        mse = np.mean((force - poses['pred'][:,i]) ** 2)
    # for i, (force, name) in enumerate(zip(poses['target'].T, joint_names)):
    #     axs[i].plot(force, 'r:', label=name + ' target')
        
        axs[i].set_title(name + f', MSE: {mse:.4f}')
        axs[i].set_ylabel('qpos (rad)')
        axs[i].legend()
        axs[i].grid()
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout()
    plt.show()
    return fig, axs


def visualize_impedance(fig, axs, jkps):
    for i in range(jkps[0].shape[0]):
        axs.plot(jkps[:,i] - jkps[0,i], label= f'jkp {i}, init = {jkps[0,i]:.2f}')
        axs.grid()
    axs.legend()
    axs.set_title('Joint stiffness changes')
    
    plt.tight_layout()
    return fig, axs


def visualize_force( actuator_forces, actuator_names):
    num_actuators = len(actuator_names)
    num_cols = 4
    num_rows = num_actuators // num_cols + (num_actuators % num_cols > 0)
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 3))
    axs = axs.flatten()
    
    for i, (force, name) in enumerate(zip(actuator_forces.T, actuator_names)):
        axs[i].plot(force, label=name)
        axs[i].set_title(name)
        axs[i].set_ylabel('torque (Nm/kg)')
        axs[i].legend()
        axs[i].grid()
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout()
    plt.show()
    return fig, axs


def visualize_torques(fig, axs, vfs):
    for i in range(vfs[0].shape[0]):
        if i < 6: 
            axs[0].plot(vfs[:,i], label= f'rf {i}')
            
        else: 
            axs[1].plot(vfs[:,i])
    axs[0].legend()
    plt.tight_layout()
    return fig, axs

def visualize_3d_forces(fig, ax, forces, positions, sc = 500):
    if type(forces) is list:
        # Plot the positions
        positions = np.array(positions)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
        print(forces)
        # Plot the forces
        for pos, force in zip(positions, forces):
            ax.quiver(pos[0], pos[1], pos[2], force[0], force[1], force[2], length=np.linalg.norm(force)/sc, normalize=True)
            ax.text(pos[0] + force[0]/sc * 1.1, pos[1] + force[1]/sc * 1.1, pos[2] + force[2] /sc* 1.1, f'{np.linalg.norm(force):.2f}', color='blue', fontsize=8)
            
    else:
        if forces is not None and positions is not None:
            ax.quiver(positions[0], positions[1], positions[2], forces[0], forces[1], forces[2], length=np.linalg.norm(forces)/sc, normalize=True)
            ax.scatter(positions[0], positions[1], positions[2], c='r', marker='o')
            ax.text(positions[0] + forces[0]/sc * 1.1, positions[1] + forces[1]/sc * 1.1, positions[2] + forces[2] /sc* 1.1, f'{np.linalg.norm(forces):.2f}', color='blue', fontsize=8)

    return fig, ax


def visualize_skeleton(fig, ax, joints, tree, coms = None):
    """
    visualzie the skeletons specifed by the joints and tree
    """
    # Plot the joints
    for name, com in joints.items():
        ax.scatter(com[0], com[1], com[2], c='r', marker='o')

    # Convert dictionary values to a numpy array
    joints_array = np.array(list(joints.values()))
    joints_array = joints_array[~np.isnan(joints_array).any(axis=1)]

    # Set equal scaling
    max_range = np.array([joints_array[:, 0].max() - joints_array[:, 0].min(), 
                          joints_array[:, 1].max() - joints_array[:, 1].min(), 
                          joints_array[:, 2].max() - joints_array[:, 2].min()]).max() / 2.0 + 0.3
    
    # link the component from parent to child
   
    for parent, children in tree.items():
        for child in children:
            try: 
                parent_com = joints[parent]
                child_com = joints[child]
                ax.plot([parent_com[0], child_com[0]], 
                        [parent_com[1], child_com[1]], 
                        [parent_com[2], child_com[2]], 'k-')
                if child == 'lfoot' : # seems like prothetic side
                    ax.plot([parent_com[0], child_com[0]], 
                        [parent_com[1], child_com[1]], 
                        [parent_com[2], child_com[2]], 'r-')
                    
            except KeyError:
                pass
    
    if coms is not None: 
        for com in coms.values():
            ax.scatter(com[0][0], com[0][1], com[0][2], c='b', marker='o') 
    mid_x = (joints_array[:, 0].max() + joints_array[:, 0].min()) * 0.5
    mid_y = (joints_array[:, 1].max() + joints_array[:, 1].min()) * 0.5
    mid_z = (joints_array[:, 2].max() + joints_array[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=0)
    ax.set_aspect('equal')
    
    # Add a plane at z=0
    xx, yy = np.meshgrid(np.linspace(mid_x - max_range, mid_x + max_range, 10),
                         np.linspace(mid_y - max_range, mid_y + max_range, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    return fig, ax

def frames_to_video(frame_dir, out_dir, fps=30, filename='output.mp4'):

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Define numerical sort for frames
    numbers = re.compile(r'(\d+)')
    def numerical_sort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    # Get all frames from the directory
    frames = sorted(glob.glob(os.path.join(frame_dir, '*.png')), key=numerical_sort)
    
    if not frames:
        raise ValueError(f"No frames found in directory: {frame_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    height, width, layers = first_frame.shape
    
    # Create video writer
    output_path = os.path.join(out_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    # Release the video writer
    out.release()
    


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))


def out_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../out'))


def log_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../logs'))


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    
def load_img(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        I = Image.open(f)
        img = I.resize((224, 224), Image.ANTIALIAS).convert('RGB')
        return img


def save_image_hwc(data, file_path):
    """
    Save a NumPy array as an image file.
    
    Args:
        data (np.ndarray): Image array in HWC format (Height x Width x Channels)
        file_path (str): Path where the image will be saved
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert the array to uint8 if not already
    if data.dtype != np.uint8:
        data = (data * 255).astype(np.uint8)
    
    # Create and save the image
    image = Image.fromarray(data)
    image.save(file_path)

## TODO: fix this: WSL is not supported
def save_screen_shots(window, file_name, transparent=False, autogui=False):
    import glfw
    xpos, ypos = glfw.get_window_pos(window) 
    width, height = glfw.get_window_size(window)
    
    if autogui:
        import pyautogui
        image = pyautogui.screenshot(region=(xpos*2, ypos*2, width*2, height*2))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA if transparent else cv2.COLOR_RGB2BGR)
        if transparent:
            image[np.all(image >= [240, 240, 240, 240], axis=2)] = [255, 255, 255, 0]
        cv2.imwrite(file_name, image)
    else:
        print(width*2, height*2)
        subprocess.call(['screencapture', '-x', '-m', f'-R {xpos},{ypos},{width},{height}', file_name])


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count