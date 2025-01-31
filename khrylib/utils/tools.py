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
def plot_qpos(qpos, body_qposaddr_list_start_index, body_qposaddr):
    fig, axs = plt.subplots(nrows=qpos.shape[1]//4+1, ncols=4, figsize=(10, 12))
    for i in range(qpos.shape[1]//4+1):
        for j in range(4):
            idx = i*4 + j
            if idx >= qpos.shape[1]:
                break
            gt = qpos[:, idx]
            axs[i, j].plot(gt, 'r', label='gt')
            axs[i, j].set_ylim([-np.pi, np.pi])
            if idx in body_qposaddr_list_start_index:
                body_name = [name for name, addr in body_qposaddr.items() if addr[0] == idx][0]
                axs[i, j].set_title(f"idx = {idx}, {body_name}", fontsize=12)
            if i == 0 and j == 0:
                axs[i, j].legend()
    plt.tight_layout()
    plt.show()
    
def visualize_poses(fig, axs, poses, body_qposaddr): 
    for i in range(poses['gt'].shape[1]//4+1):
        for j in range(4):
            idx = i*4 + j
            if idx < poses['gt'].shape[1]: 
                gt = poses['gt'][:, idx ]
                pred = poses['pred'][:, idx ]
                mse = np.mean((gt - pred) ** 2)
                axs[i, j].plot(gt, 'r', label='gt')
                axs[i, j].plot(pred, 'b', label='pred')
                axs[i, j].set_ylim([-np.pi, np.pi])
                # body_name = [name for name, addr in body_qposaddr.items() if addr[0] == idx][0]
                # axs[i, j].set_title(f"idx = {idx}, {body_name} MSE: {mse:.4f}", fontsize=4)
                axs[i, j].set_title(f'MSE: {mse:.4f}')
            if i == 0 and j == 0:
                axs[i, j].legend()
    plt.tight_layout()
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

def visualize_contact_forces(fig, ax, forces, positions):
    # Plot the positions
    positions = np.array(positions)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    
    # Plot the forces
    for pos, force in zip(positions, forces):
        ax.quiver(pos[0], pos[1], pos[2], force[0], force[1], force[2], length=np.linalg.norm(force)/500.0, normalize=True)
        ax.text(pos[0] + force[0] * 1.1, pos[1] + force[1] * 1.1, pos[2] + force[2] * 1.1, f'{np.linalg.norm(force):.2f}', color='blue', fontsize=8)
    
    return fig, ax

def visualize_lower_limb_com(fig, ax, coms, tree):
    # Plot the COMs
    for name, com in coms.items():
        ax.scatter(com[0], com[1], com[2], c='r', marker='o')

    # Convert dictionary values to a numpy array
    coms_array = np.array(list(coms.values()))

    # Set equal scaling
    max_range = np.array([coms_array[:, 0].max() - coms_array[:, 0].min(), 
                          coms_array[:, 1].max() - coms_array[:, 1].min(), 
                          coms_array[:, 2].max() - coms_array[:, 2].min()]).max() / 2.0 + 0.3
    
    # link the component from parent to child
   
    for parent, children in tree.items():
        for child in children:
            parent_com = coms[parent]
            child_com = coms[child]
            ax.plot([parent_com[0], child_com[0]], 
                    [parent_com[1], child_com[1]], 
                    [parent_com[2], child_com[2]], 'k-')

    mid_x = (coms_array[:, 0].max() + coms_array[:, 0].min()) * 0.5
    mid_y = (coms_array[:, 1].max() + coms_array[:, 1].min()) * 0.5
    mid_z = (coms_array[:, 2].max() + coms_array[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lower Limb COMs')
    
    # Add a plane at z=0
    xx, yy = np.meshgrid(np.linspace(mid_x - max_range, mid_x + max_range, 10),
                         np.linspace(mid_y - max_range, mid_y + max_range, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    return fig, ax


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