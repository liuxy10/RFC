import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from khrylib.utils.tools import *
import cv2
    
def process_force_data(csv_file):
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
    
    # Extract force data
    force_data = raw.iloc[:, [2, 3, 4, 17, 18, 19, 32, 33, 34]].astype(float)
    
    # Extract moment data
    moment_data = raw.iloc[:, [5, 6, 7, 20, 21, 22, 35, 36, 37]].astype(float)
    
    # Extract CoP data
    cop_data = raw.iloc[:, [8, 9, 10, 23, 24, 25, 38, 39, 40]].astype(float)
    
    return force_data, moment_data, cop_data

def process_motion_data(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Find index of 'Trajectories'
    trajectories_index = df[df.iloc[:, 0] == 'Trajectories'].index[0]
    # Extract raw data
    raw = df.iloc[trajectories_index+5:, 2:119].astype(float)
    
    # Calculate normalization from Clavicle
    Clav = raw.iloc[:, 18:21]
    Norm = [np.nanmean(Clav.iloc[:, 0]), np.nanmean(Clav.iloc[:, 1])]
    
    # Calculate body part coordinatespip install pandas
    # Head coordinates (average of 4 markers)
    Head = np.column_stack([
        (raw.iloc[:, [0,3,6,9]].mean(axis=1) - Norm[0]),
        (raw.iloc[:, [1,4,7,10]].mean(axis=1) - Norm[1]),
        (raw.iloc[:, [2,5,8,11]].mean(axis=1))
    ]) # not used in this project
    
    # Other body parts
    body_parts = {  # with joint name commented here
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
        'rtibia': ([102, 103, 104], [0, 0, None]), # 'Rknee'
        'rfoot': ([108, 109, 110], [0, 0, None]) # 'Rank'
    }
    
    

    body_tree = {
        'root': ['lhipjoint', 'rhipjoint', 'lowerback'],
        'lhipjoint': ['lfemur'],
        'lfemur': ['ltibia'],
        'ltibia': ['lfoot'],
        'lfoot': [],
        'rhipjoint': ['rfemur'],
        'rfemur': ['rtibia'],
        'rtibia': ['rfoot'],
        'rfoot': [],
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
    Head = np.column_stack([
        (raw.iloc[:, [0, 3, 6, 9]].mean(axis=1) - Norm[0]),
        (raw.iloc[:, [1, 4, 7, 10]].mean(axis=1) - Norm[1]),
        (raw.iloc[:, [2, 5, 8, 11]].mean(axis=1))
    ])
    coordinates['head'] = Head
    # Find foot strike indices
    foot_strikes = df[(df.iloc[:, 1] == 'Right') & (df.iloc[:, 2] == 'Foot Strike')].index
    steps = foot_strikes[-22:-1]
    step_times = np.round(df.iloc[steps, 3].astype(float) * 100)
    
    return coordinates, step_times, body_tree

if __name__ == "__main__":
    # Process motion data
    # fpath = 'data/nrel/no-rail-W1_High K2/no-rail-W1_High K2.csv'
    dt = 0.01
    # fname = 'Walking_passive01_K4'  
    # fdir = 'data/nrel/Walking_passive01_K4'
    fdir = 'data/nrel/no-rail-W1_High K2'
    fname = 'no-rail-W1_High K2'
    frame_dir = f'{fdir}/frame_skeleton'
    fpath = os.path.join(fdir, f'{fname}.csv')
    coordinates, step_times, body_tree = process_motion_data(fpath)
    force_data, moment_data, cop_data = process_force_data(fpath)
    # Print coordinates and step times
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

    # visualize the motion using matplotlib
    plot_motion = False
    if plot_motion:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for part, coords in coordinates.items():
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=part)
        plt.legend()
        plt.show()
        
    # Loop through frames and save them
    for i in range(0, coordinates['head'].shape[0], 100):
        try: 
            print(f"Processing frame {i}/{coordinates['head'].shape[0]}")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            chosen_parts_coms = {part: coords[i] for part, coords in coordinates.items()}
            visualize_skeleton(fig, ax, chosen_parts_coms, body_tree)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            save_image_hwc(data,  f'{frame_dir}/%04d.png' % i) 
            plt.close(fig)
        except ValueError:
            print(f"Error processing frame {i}")
    
    # Release the video writer object
    out_name = f'{fname}_skeleton.mp4'
    frames_to_video(frame_dir, fdir, 10, out_name)
    # visualize the force data using matplotlib
    plot_force = False
    if plot_force: 
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(313)
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Plot Fx
        axs[0].plot(force_data.index, force_data['RFx'], label='Right Fx')
        axs[0].plot(force_data.index, force_data['LFx'], label='Left Fx')
        axs[0].set_ylabel('Fx')
        axs[0].legend()

        # Plot Fy
        axs[1].plot(force_data.index, force_data['RFy'], label='Right Fy')
        axs[1].plot(force_data.index, force_data['LFy'], label='Left Fy')
        axs[1].set_ylabel('Fy')
        axs[1].legend()

        # Plot Fz
        axs[2].plot(force_data.index, force_data['RFz'], label='Right Fz')
        axs[2].plot(force_data.index, force_data['LFz'], label='Left Fz')
        axs[2].set_ylabel('Fz')
        axs[2].legend()

        plt.xlabel('Time')
        plt.show()
        

    
    