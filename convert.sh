#!/bin/bash

# python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 69_10 --out_id 69_10 
for i in {6..9}
do
    python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 69_0${i} --out_id 69_0${i}
done