import pandas as pd
import numpy as np

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
    
    # Calculate body part coordinates
    # Head coordinates (average of 4 markers)
    Head = np.column_stack([
        (raw.iloc[:, [0,3,6,9]].mean(axis=1) - Norm[0]),
        (raw.iloc[:, [1,4,7,10]].mean(axis=1) - Norm[1]),
        (raw.iloc[:, [2,5,8,11]].mean(axis=1))
    ])
    
    # Other body parts
    body_parts = {
        'LShould': ([24, 25, 29], [0, 0, None]),
        'LElb': ([33, 34, 35], [0, 0, None]),
        'Lwrist': ([39,42, 40,43, 41,44], [0, 0, None]),
        'RShould': ([48, 49, 50], [0, 0, None]),
        'RElb': ([54, 55, 56], [0, 0, None]),
        'Rwrist': ([60,63, 61,64, 62,65], [0, 0, None]),
        'Pelv': ([75,78, 79,73, 77,80], [0, 30, None]),
        'Lhip': ([69, 70, 71], [-30, 0, -70]),
        'Rhip': ([72, 73, 74], [30, 0, -70]),
        'Lknee': ([84, 85, 86], [0, 0, None]),
        'Lank': ([90, 91, 92], [0, 0, None]),
        'Rknee': ([102, 103, 104], [0, 0, None]),
        'Rank': ([108, 109, 110], [0, 0, None])
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
    
    # Find foot strike indices
    foot_strikes = df[(df.iloc[:, 1] == 'Right') & (df.iloc[:, 2] == 'Foot Strike')].index
    steps = foot_strikes[-22:-1]
    step_times = np.round(df.iloc[steps, 3].astype(float) * 100)
    
    return coordinates, step_times
