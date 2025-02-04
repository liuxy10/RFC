# Residual Force Control (RFC)

This repo contains the official implementation of our paper:
  
Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis  
Ye Yuan, Kris Kitani  
**NeurIPS 2020**  
[[website](https://www.ye-yuan.com/rfc)] [[paper](https://arxiv.org/pdf/2006.07364.pdf)] [[video](https://youtu.be/XuzH1u78o1Y)]

# Installation 
### Dataset and Pretrained Models
* The CMU mocap data we use is already included in the [data/cmu_mocap](data/cmu_mocap) folder. The pretrained models are in [result/motion_im](result/motion_im) where each subfolder (e.g., [0506](result/motion_im/0506)) corresponds to a training config file (e.g., [0506.yml](motion_imitation/cfg/0506.yml)) in [motion_imitation/cfg](motion_imitation/cfg).
* We have provided the following configs (label): 
0506 (ballet1), 0507 (ballet2), 0513 (ballet3), 8801 (backflip), 9002 (cartwheel), 9005 (jump
kick), 9008 (side flip), 9011 (handspring).
### Environment
* **Tested OS:** MacOS, Linux
* Python >= 3.6
### How to install
1. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Install MuJoCo following the steps [here](https://github.com/openai/mujoco-py#install-mujoco). Note that [mujoco-py](https://github.com/openai/mujoco-py) (MuJoCo's python binding) is already installed in step 1. This step is to install the actual MuJoCo library.  
   **Note: [MuJoCo](https://mujoco.org/) is now free thanks to DeepMind!** 🎉🎉🎉
3. Set the following environment variable to improve multi-threaded sampling performance:    
    ```
    export OMP_NUM_THREADS=1
    ```

# Quick Demo
### Visualize Pretrained Policy
```
python motion_imitation/vis_im.py --cfg 0506 --iter 1000
```
Here [0506](motion_imitation/cfg/0506.yml) corresponds to the config [motion_imitation/cfg/0506.yml](motion_imitation/cfg/0506.yml) which uses the motion_clip of subject 05 and trial 06 from the [CMU mocap database](http://mocap.cs.cmu.edu/). Check out [motion_imitation/cfg](motion_imitation/cfg) for other configs.

**Keyboard shortcuts** for the GUI:  
| Key           | Functionality          | Key           | Functionality          |
| ------------- | ---------------------- | ------------- | ---------------------- |
| space         | pause/resume animation | w/e           | go to first/last frame |
| left/right    | previous/next frame    | d/f           | slow down/speed up     |
| g             | toggle loop            | s             | toggle reverse         |

# Training
### Prepare training data
**Note**: for motion clips used in the paper, the data has already been processed.

To train policies with other motion clips from the [CMU mocap database](http://mocap.cs.cmu.edu/), you will need to first download the `.amc` mocap file and place it inside [data/cmu_mocap/amc](data/cmu_mocap/amc), and run the following script:
```
python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 05_06 --out_id 05_06_test --render
```
Here we take `05_06.amc` as an example and this command will show a GUI that visualizes the motion clip. The GUI has a few keyboard shortcuts that allow you to adjust the clip and save the training pickle file (`05_06_test.p`) inside [data/cmu_mocap/motion](data/cmu_mocap/motion):
| Key           | Functionality                    | Key           | Functionality           |
| ------------- | -------------------------------- | ------------- | ----------------------- |
| space         | pause/resume animation           | w/e           | go to first/last frame  |
| left/right    | previous/next frame              | d/f           | slow down/speed up      |
| up/down       | move humanoid up/down            | z/x           | set start/end of motion |
| c             | cut motion with selected range   | r             | save motion as pickle   |

If you don't need to adjust the clip, you can just run the script without `--render`, so the motion pickle is saved without GUI.


### Train RFC policy
You can create your own config file using [0506.yml](motion_imitation/cfg/0506.yml) as a template. Be sure to set [motion_id](motion_imitation/cfg/0506.yml#L2) to the motion you have processed. To train an RFC policy, just run:
```
python motion_imitation/normal.py --cfg 0506 --num_threads <max_num_CPU_threads_you_have>
```
This will save models and logs into [result/motion_im/0506](result/motion_im/0506).

# Citation
If you find our work useful in your research, please cite our paper [RFC](https://www.ye-yuan.com/rfc):
```bibtex
@inproceedings{yuan2020residual,
    title={Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis},
    author={Yuan, Ye and Kitani, Kris},
    booktitle={Advances in Neural Information Processing Systems},
    year={2020}
}
```

# License
The software in this repo is freely available for free non-commercial use. Please see the [license](LICENSE) for further details.

# XINYI:

Successfully installed filelock-3.13.1 fsspec-2024.2.0 jinja2-3.1.3 mpmath-1.3.0 networkx-3.0 nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-cupti-cu11-11.8.87 nvidia-cuda-nvrtc-cu11-11.8.89 nvidia-cuda-runtime-cu11-11.8.89 nvidia-cudnn-cu11-9.1.0.70 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.3.0.86 nvidia-cusolver-cu11-11.4.1.48 nvidia-cusparse-cu11-11.7.5.86 nvidia-nccl-cu11-2.20.5 nvidia-nvtx-cu11-11.8.86 sympy-1.13.1 torch-2.4.1+cu118 torchaudio-2.4.1+cu118 torchvision-0.19.1+cu118 triton-3.0.0 typing-extensions-4.9.0


added new WALKING amc data ([from HERE](http://mocap.cs.cmu.edu/search.php))
``` 
python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 02_01 --out_id 02_01_test --render
```

```
python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 02_02 --out_id 02_02_test --render
```

2/4/2025 birthday!
    body_parts = {
        'Left_Front_Head': [2, 3, 4], 'Right_Front_Head': [5, 6, 7], 'Left_Back_Head': [8, 9, 10], 'Right_Back_Head': [11, 12, 13],
        'C7': [14, 15, 16], 'T10': [17, 18, 19], 'Clavicle': [20, 21, 22], 'Sternum': [23, 24, 25],
        'Right_Back': [26, 27, 28], 'Left_Shoulder': [29, 30, 31], 'Left_Upper_Arm': [32, 33, 34], 'Left_Elbow': [35, 36, 37],
        'Left_Forearm': [38, 39, 40], 'Left_Wrist_A': [41, 42, 43], 'Left_Wrist_B': [44, 45, 46], 'Left_Finger': [47, 48, 49],
        'Right_Shoulder': [50, 51, 52], 'Right_Upper_Arm': [53, 54, 55], 'Right_Elbow': [56, 57, 58], 'Right_Forearm': [59, 60, 61],
        'Right_Wrist_A': [62, 63, 64], 'Right_Wrist_B': [65, 66, 67], 'Right_Finger': [68, 69, 70], 'Left_ASIS': [71, 72, 73],
        'Right_ASIS': [74, 75, 76], 'Left_PSIS': [77, 78, 79], 'Right_PSIS': [80, 81, 82], 'Left_Thigh': [83, 84, 85],
        'Left_Knee': [86, 87, 88], 'Left_Tibia': [89, 90, 91], 'Left_Ankle': [92, 93, 94], 'Left_Heel': [95, 96, 97],
        'Left_Toe': [98, 99, 100], 'Right_Thigh': [101, 102, 103], 'Right_Knee': [104, 105, 106], 'Right_Tibia': [107, 108, 109],
        'Right_Ankle': [110, 111, 112], 'Right_Heel': [113, 114, 115], 'Right_Toe': [116, 117, 118]
    }
    body_tree = {
        'Left_Front_Head': ['Right_Front_Head', 'Left_Back_Head'],
        'Right_Front_Head': ['Right_Back_Head'],
        'Left_Back_Head': ['Right_Back_Head'],
        'Right_Back_Head': ['C7'],
        'C7': ['T10', 'Clavicle'],
        'T10': ['Left_ASIS', 'Right_ASIS'],
        'Clavicle': ['Sternum', 'Left_Shoulder', 'Right_Shoulder'],
        'Sternum': [],
        'Right_Back': [],
        'Left_Shoulder': ['Left_Upper_Arm'],
        'Left_Upper_Arm': ['Left_Elbow'],
        'Left_Elbow': ['Left_Forearm'],
        'Left_Forearm': ['Left_Wrist_A', 'Left_Wrist_B'],
        'Left_Wrist_A': ['Left_Finger'],
        'Left_Wrist_B': ['Left_Finger'],
        'Left_Finger': [],
        'Right_Shoulder': ['Right_Upper_Arm'],
        'Right_Upper_Arm': ['Right_Elbow'],
        'Right_Elbow': ['Right_Forearm'],
        'Right_Forearm': ['Right_Wrist_A', 'Right_Wrist_B'],
        'Right_Wrist_A': ['Right_Finger'],
        'Right_Wrist_B': ['Right_Finger'],
        'Right_Finger': [],
        'Left_ASIS': ['Left_PSIS', 'Left_Thigh'],
        'Right_ASIS': ['Right_PSIS', 'Right_Thigh'],
        'Left_PSIS': [],
        'Right_PSIS': [],
        'Left_Thigh': ['Left_Knee'],
        'Left_Knee': ['Left_Tibia'],
        'Left_Tibia': ['Left_Ankle'],
        'Left_Ankle': ['Left_Heel', 'Left_Toe'],
        'Left_Heel': [],
        'Left_Toe': [],
        'Right_Thigh': ['Right_Knee'],
        'Right_Knee': ['Right_Tibia'],
        'Right_Tibia': ['Right_Ankle'],
        'Right_Ankle': ['Right_Heel', 'Right_Toe'],
        'Right_Heel': [],
        'Right_Toe': []
    }