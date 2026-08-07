#!/bin/bash
cd /data/homes/ugadiarov_la/ugadiarov.la/projects/custom-verl
source activate.sh
export SETUP_PATH=run/acereason/setup_datasets_acereason.sh
setsid nohup bash run/acereason/grpo_24k_4gpu_acereason_h200.sh > run_h200_24k_4gpu_acereason.log 2>&1 &
echo "TRAIN_PID=$!"
setsid nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 15 > gpu_util_24k_4gpu_acereason.csv 2>/dev/null &
echo "GPULOG_PID=$!"
