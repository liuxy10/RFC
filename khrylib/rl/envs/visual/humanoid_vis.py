import numpy as np

import os
import sys
sys.path.append(os.getcwd())
from khrylib.rl.envs.common import mujoco_env
from khrylib.utils import * 
import matplotlib.pyplot as plt 

class HumanoidVisEnv(mujoco_env.MujocoEnv):
    def __init__(self, vis_model_file, nframes=6, focus=True):
        mujoco_env.MujocoEnv.__init__(self, vis_model_file, nframes)
        self.set_cam_first = set()
        self.focus = focus
        self.body_tree = build_body_tree(vis_model_file)

    def step(self, a):
        return np.zeros((10, 1)), 0, False, dict()

    def reset_model(self):
        c = 0
        self.set_state(
            self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return None

    def sim_forward(self):
        self.sim.forward()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        if self.focus:
            self.viewer.cam.lookat[:2] = self.data.qpos[:2]
            self.viewer.cam.lookat[2] = 0.8
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 30
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.5
            self.viewer.cam.elevation = -10
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)
        
    def visualize_by_frame(self, show = False, label =  "normal"):
        # print(self.model.body_names)
        # exit()
        body_pos_pred = {n: self.get_body_position(n) for n in self.model.body_names if n != "world" and "1_" not in n}
        body_pos_gt = {n: self.get_body_position(n) for n in self.model.body_names if n != "world" and "1_" in n}
        
        print(body_pos_pred, body_pos_gt)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 10))
        # forces, poss = self.get_contact_force()
        # f, cops, _ = self.get_ground_reaction_force()
        ax.view_init(elev=0, azim=180)  # Set the view to face the yz plane
        ax.set_title(label)
        fig, ax = visualize_skeleton(fig, ax, body_pos_pred, self.body_tree)
        
        if show:
            plt.show()
        return fig, ax
    
    def get_body_position(self, body_name):
        sim = self.sim
        body_id = sim.model.body_name2id(body_name)
        return sim.data.body_xpos[body_id]
        