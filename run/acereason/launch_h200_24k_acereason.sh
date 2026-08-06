#!/bin/bash
cd /data/homes/ugadiarov_la/ugadiarov.la/projects/custom-verl
source activate.sh
export SETUP_PATH=setup_datasets_acereason.sh
setsid nohup bash grpo_24k_6gpu_acereason_h200.sh > run_h200_24k_acereason.log 2>&1 &
echo "TRAIN_PID=$!"
setsid nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 15 > gpu_util_24k_acereason.csv 2>/dev/null &
echo "GPULOG_PID=$!"
