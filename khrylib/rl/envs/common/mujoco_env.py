from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from pathlib import Path
import mujoco_py
from khrylib.rl.envs.common.mjviewer import MjViewer
import mujoco
DEFAULT_SIZE = 500


class MujocoEnv:
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, fullpath, frame_skip):
        if not path.exists(fullpath):
            # try the default assets path
            fullpath = path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models', path.basename(fullpath))
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.model_dm = mujoco.MjModel.from_xml_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model) 
        self.data = self.sim.data # changing self.data will change self.sim.data
        self.data_dm = mujoco.MjData(self.model_dm)
        self.viewer = None
        self._viewers = {}
        self.obs_dim = None
        self.action_space = None
        self.observation_space = None
        self.np_random = None
        self.cur_t = 0  # number of steps taken
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.prev_qpos = None
        self.prev_qvel = None
        self.seed()

    def set_spaces(self):
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        self.obs_dim = observation.size
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def step(self, action):
        """
        Step the environment forward.
        """
        raise NotImplementedError

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self, mode):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        self.cur_t = 0
        ob = self.reset_model()
        old_viewer = self.viewer
        for mode, v in self._viewers.items():
            self.viewer = v
            self.viewer_setup(mode)
        self.viewer = old_viewer
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state() 
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state) # udd = user defined
        self.sim.set_state(new_state) # set_state() sets the internal data of the simulation to the given state
        self.sim.forward() # forward() updates the internal data of the simulation 

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'image':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it, and the image format is BGR for OpenCV
            return data[::-1, :, [2, 1, 0]]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = MjViewer(self.sim)
            elif mode == 'image':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
            self._viewers[mode] = self.viewer
        self.viewer_setup(mode)
        return self.viewer

    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func

    def get_body_frame_position(self, body_name):
        return self.data.get_body_xpos(body_name) # get_body_xpos() returns the position of the body in world coordinates

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def vec_body2world(self, body_name, vec):
        body_xmat = self.data.get_body_xmat(body_name)
        vec_world = (body_xmat @ vec[:, None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world

