import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/root/Github/RFC/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from khrylib.utils.tools import *
import cv2

def process_force_data(csv_file, dt = 0.01):
    df = pd.read_csv(csv_file)
    # Find index of 'Trajectories'
    trajectories_index = df[df.iloc[:, 0] == 'Trajectories'].index[0]
    treadmil_index = df[df.iloc[:, 0] == 'Devices'].index[0]
    # Read CSV file
    raw = df.iloc[treadmil_index+5:trajectories_index, :41].astype(float)

    # Define column names
    columns = [
        'Frame', 'Sub Frame', # 2
        'RFx', 'RFy', 'RFz', 'RMx', 'RMy', 'RMz', 'RCx', 'RCy', 'RCz', 'RPin1', 'RPin2', 'RPin3', 'RPin4', 'RPin5', 'RPin6', # 2 + 15
        'LFx', 'LFy', 'LFz', 'LMx', 'LMy', 'LMz', 'LCx', 'LCy', 'LCz', 'LPin8', 'LPin9', 'LPin10', 'LPin11', 'LPin12', 'LPin13', # 2 + 15 + 15
        'CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz', 'CCx', 'CCy', 'CCz' # 2 + 15 + 15 + 9
    ]
    raw.columns = columns[:41]  # Assign column names to the raw data
    ts = raw.iloc[:, 0] * dt 
    # Extract force data
    force_data = raw.iloc[:, [2, 3, 4, 17, 18, 19, 32, 33, 34]].astype(float)

    # Extract moment data
    moment_data = raw.iloc[:, [5, 6, 7, 20, 21, 22, 35, 36, 37]].astype(float)

    # Extract CoP data
    cop_data = raw.iloc[:, [8, 9, 10, 23, 24, 25, 38, 39, 40]].astype(float)

    return ts, force_data, moment_data, cop_data

def process_motion_data(csv_file, dt = 0.01):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Find index of 'Trajectories'
    trajectories_index = df[df.iloc[:, 0] == 'Trajectories'].index[0]
    # Extract raw data
    raw = df.iloc[trajectories_index+5:, 2:119].astype(float)

    # Calculate normalization from Clavicle
    Clav = raw.iloc[:, 18:21]
    Norm = [np.nanmean(Clav.iloc[:, 0]), np.nanmean(Clav.iloc[:, 1])]
    # Extract treadmill speed from the CSV file
    treadmill_speed = float(df.columns[0].split(':')[1].split()[0])
    treadmill_speed = 0.0
    total_time = len(raw) * dt
    print(f"Total time: {total_time} seconds")
    print(f"Time step (dt): {dt} seconds")
    print(f"Walking speed: {treadmill_speed} m/s")
    Norm[1] += treadmill_speed * total_time * np.linspace(0, 1, len(raw)) * 1000 # Add treadmill speed to y-axis
    # Calculate body part coordinatespip install pandas

    # body parts
    body_parts = {  # with joint name commented here
        
        'lowerneck': ([12, 13, 14], [0, 0, None]), # 'Neck'
        'lclavicle': ([27, 28, 29], [0, 0, None]), # 'Left Should'
        'lhumerus': ([33, 34, 35], [0, 0, None]), # 'Left Elbow'
        'lradius': ([39, 42, 40, 43, 41, 44], [0, 0, None]), # 'Left wrist'
        'rclavicle': ([48, 49, 50], [0, 0, None]), # 'Right Should'
        'rhumerus': ([54, 55, 56], [0, 0, None]), # 'Right Elbow'
        'rradius': ([60, 63, 61, 64, 62, 65], [0, 0, None]), # 'Rwrist'
        'lowerback': ([75, 78, 79, 73, 77, 80], [0, 30, None]), # 'Pelv'
        'lfemur': ([69, 70, 71], [-30, 0, -70]), # 'Lhip'
        'rfemur': ([72, 73, 74], [30, 0, -70]), # 'Rhip'
        'ltibia': ([84, 85, 86], [0, 0, None]), # 'Lknee'
        'lfoot': ([90, 91, 92], [0, 0, None]), # 'Lank'
        'lheel': ([93, 94, 95], [0, 0, None]), # 'Lheel'
        'ltoes': ([96, 97, 98], [0, 0, None]), # 'Ltoe'
        'rtibia': ([102, 103, 104], [0, 0, None]), # 'Rknee'
        'rfoot': ([108, 109, 110], [0, 0, None]), # 'Rank'
        'rtoes': ([114, 115, 116], [0, 0, None]), # 'Rtoe'
        'rheel': ([111, 112, 113], [0, 0, None]) # 'Rheel'
    }

    body_tree = {
        'root': ['lhipjoint', 'rhipjoint', 'lowerback'],
        'lhipjoint': ['lfemur'],
        'lfemur': ['ltibia'],
        'ltibia': ['lfoot'],
        'lfoot': ['ltoes', 'lheel'],
        'ltoes': ['lheel'],
        'rhipjoint': ['rfemur'],
        'rfemur': ['rtibia'],
        'rtibia': ['rfoot'],
        'rfoot': ['rtoes', 'rheel'],
        'rtoes': ['rheel'],
        'lowerback': ['upperback'],
        'upperback': ['thorax'],
        'thorax': ['lowerneck'],
        'lowerneck': ['upperneck', 'lclavicle', 'rclavicle'],
        'upperneck': ['head'],
        'head': [],
        'lclavicle': ['lhumerus'],
        'lhumerus': ['lradius'],
        'lradius': ['lwrist'],
        'lwrist': [],
        'rclavicle': ['rhumerus'],
        'rhumerus': ['rradius'],
        'rradius': ['rwrist'],
        'rwrist': [],
    }


    coordinates = {}
    for part, (cols, offsets) in body_parts.items():
        if len(cols) == 3:
            coordinates[part] = np.column_stack([
                raw.iloc[:, cols[0]] - Norm[0] + (offsets[0] or 0),
                raw.iloc[:, cols[1]] - Norm[1] + (offsets[1] or 0),
                raw.iloc[:, cols[2]] + (offsets[2] or 0)
            ])
        else:  # For averaged coordinates (wrist, pelvis)
            coordinates[part] = np.column_stack([
                (raw.iloc[:, cols[0]] + raw.iloc[:, cols[1]])/2 - Norm[0] + (offsets[0] or 0),
                (raw.iloc[:, cols[2]] + raw.iloc[:, cols[3]])/2 - Norm[1] + (offsets[1] or 0),
                (raw.iloc[:, cols[4]] + raw.iloc[:, cols[5]])/2 + (offsets[2] or 0)
            ])
    # Head coordinates (average of 4 markers)
    coordinates["head"] = np.column_stack([
        (raw.iloc[:, [0,3,6,9]].mean(axis=1) - Norm[0]),
        (raw.iloc[:, [1,4,7,10]].mean(axis=1) - Norm[1]),
        (raw.iloc[:, [2,5,8,11]].mean(axis=1))
    ]) # not used in this project
    coordinates['upperback'] = np.column_stack([
        (raw.iloc[:, [12, 15]].mean(axis=1) - Norm[0]), # C7 & T10
        (raw.iloc[:, [13, 16]].mean(axis=1) - Norm[1]), 
        (raw.iloc[:, [14, 17]].mean(axis=1))
    ])
    coordinates['lowerback'] = np.column_stack([
        (raw.iloc[:, [15, 15, 75, 78]].mean(axis=1) - Norm[0]), # T10 + (LPSI, RPSI)
        (raw.iloc[:, [16, 16, 76, 79]].mean(axis=1) - Norm[1]), 
        (raw.iloc[:, [17, 17, 77, 80]].mean(axis=1))
    ])
    coordinates['thorax'] = np.column_stack([
        (raw.iloc[:, [12, 18]].mean(axis=1) - Norm[0]), # C7 & CLAV
        (raw.iloc[:, [13, 19]].mean(axis=1) - Norm[1]), 
        (raw.iloc[:, [14, 20]].mean(axis=1))
    ])
    coordinates['lhipjoint'] = np.column_stack([
        (raw.iloc[:, [75, 69]].mean(axis=1) - Norm[0]), # LASI & LPSI
        (raw.iloc[:, [76, 70]].mean(axis=1) - Norm[1]),
        (raw.iloc[:, [77, 71]].mean(axis=1))
    ])
    coordinates['rhipjoint'] = np.column_stack([
        (raw.iloc[:, [72, 78]].mean(axis=1) - Norm[0]), # RASI & RPSI
        (raw.iloc[:, [73, 79]].mean(axis=1) - Norm[1]),
        (raw.iloc[:, [74, 80]].mean(axis=1))
    ])
    # Find foot strike indices
    # foot_strikes = df[(df.iloc[:, 1] == 'Right') & (df.iloc[:, 2] == 'Foot Strike')].index
    # steps = foot_strikes# [-22:-1] # including all right foot strikes
    step_times = {}
    step_times["r_s"] = np.float32(df.iloc[ df[(df.iloc[:, 1] == 'Right') & (df.iloc[:, 2] == 'Foot Strike')].index , 3].astype(float))
    step_times["r_e"] = np.float32(df.iloc[ df[(df.iloc[:, 1] == 'Right') & (df.iloc[:, 2] == 'Foot Off')].index , 3].astype(float))
    step_times["l_s"] = np.float32(df.iloc[ df[(df.iloc[:, 1] == 'Left') & (df.iloc[:, 2] == 'Foot Strike')].index , 3].astype(float))
    step_times["l_e"] = np.float32(df.iloc[ df[(df.iloc[:, 1] == 'Left') & (df.iloc[:, 2] == 'Foot Off')].index , 3].astype(float))
    return coordinates, step_times, body_tree, treadmill_speed

def plot_force_data(force_data, ts = None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    if ts is None:
        ts = force_data.index
    # Plot Fx
    axs[0].plot(ts, force_data['RFx'], label='Right Fx')
    axs[0].plot(ts, force_data['LFx'], label='Left Fx')
    axs[0].set_ylabel('Fx')
    axs[0].legend()
    # Plot Fy
    axs[1].plot(ts, force_data['RFy'], label='Right Fy')
    axs[1].plot(ts, force_data['LFy'], label='Left Fy')
    axs[1].set_ylabel('Fy')
    axs[1].legend()
    # Plot Fz
    axs[2].plot(ts, force_data['RFz'], label='Right Fz')
    axs[2].plot(ts, force_data['LFz'], label='Left Fz')
    axs[2].set_ylabel('Fz')
    axs[2].legend()
    plt.xlabel('Time')
    plt.show()
def print_data(coordinates, step_times, force_data, moment_data, cop_data):
    print("Coordinates:")
    for part, coords in coordinates.items():
        print(f"{part}: {coords.shape}")

    print("\nStep times:")
    print(step_times)

    print("\nForce data:")
    print(force_data.head())

    print("\nMoment data:")
    print(moment_data.head())

    print("\nCoP data:")
    print(cop_data.head())  
def plot_motion(coordinates):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for part, coords in coordinates.items():
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=part)
    plt.legend()
    plt.show()

def save_skeleton_frame(frame_dir, coordinates,  body_tree, force_data, cop_data, freq = 10):
    
    for i in range(0, coordinates['head'].shape[0], freq):
        
        print(f"Processing frame {i}/{coordinates['head'].shape[0]}")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        chosen_parts_coms = {part: coords[i] for part, coords in coordinates.items()}
        if force_data is not None:
            visualize_3d_forces(fig, ax, np.array([force_data[k].to_numpy()[i] for k in ('RFx', 'RFy', 'RFz')]), 
                                np.array([cop_data[k].to_numpy()[i] for k in ('RCx', 'RCy', 'RCz')]), sc = 500)
            visualize_3d_forces(fig, ax, np.array([force_data[k].to_numpy()[i] for k in ('LFx', 'LFy', 'LFz')]), 
                                np.array([cop_data[k].to_numpy()[i] for k in ('LCx', 'LCy', 'LCz')]), sc = 500)
        visualize_skeleton(fig, ax, chosen_parts_coms, body_tree)
        ax.view_init(elev=0, azim=0)
        plt.show()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        save_image_hwc(data,  f'{frame_dir}/%04d.png' % i) 
        plt.close(fig)
        # except ValueError:
        #     print(f"Error processing frame {i}")
            

if __name__ == "__main__":
    # Process motion data

    dt = 0.01
    fname = 'Walking_passive01_K4'
    # fname = 'no-rail-W1_High K2'  
    fdir = f'data/nrel/{fname}'
    frame_dir = f'{fdir}/frame_skeleton'
    fpath = os.path.join(fdir, f'{fname}.csv')
    coordinates, step_times, body_tree, treadmill_speed = process_motion_data(fpath, dt)
    ts, force_data, moment_data, cop_data = process_force_data(fpath)
    force_data["LFz"] *= -1
    force_data["RFz"] *= -1
    cop_data["LCx"] -= 550
    cop_data["RCx"] -= 550
    cop_data["LCy"] -= 1200
    cop_data["RCy"] -= 1200
    
    
    print(ts.shape, force_data.shape, moment_data.shape, cop_data.shape)     
    
    # exit()
    # visualize the motion using matplotlib
    plot_motion = False
    if plot_motion: plot_motion(coordinates)

    # visualize the force data using matplotlib
    plot_force = False
    if plot_force: 
        plot_force_data(force_data)
        # Print coordinates and step times
        
        print_data(coordinates, step_times, force_data, moment_data, cop_data)
    
    
    save_skeleton_frame(frame_dir, coordinates, body_tree,  force_data, cop_data)
    out_name = f'{fname}_skeleton.mp4'
    frames_to_video(frame_dir, fdir, 30, out_name)