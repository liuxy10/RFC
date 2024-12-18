import numpy as np
import os
import shutil
import datetime
import subprocess
from os import path
from PIL import Image
from khrylib.utils.math import *
import cv2


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
def save_screen_shots(window, file_name, transparent=False, autogui=False, wsl = True):
    import glfw
    xpos, ypos = glfw.get_window_pos(window) 
    width, height = glfw.get_window_size(window)
    if False: #wsl: # for wsl2 compatibility
        # Headless FFmpeg screen capture for WSL2
        subprocess.call([
        'ffmpeg', 
        # '-video_size', f'{width - 20}x{height-20}', 
        '-framerate', '25', 
        # '-f', 'x11grab', 
        # '-i', '$DISPLAY', #f':0.0+{xpos},{ypos}', 
        file_name])
    elif autogui:
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