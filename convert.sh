#!/bin/bash

python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 69_10 --out_id 69_10 
for i in {10..75}
do
    python motion_imitation/data_process/convert_cmu_mocap.py --amc_id 69_$i --out_id 69_$i
done