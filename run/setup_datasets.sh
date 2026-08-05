datasets_path="/data2/datasets"

export PROJECT_DIR="${PWD}"
export TRAIN_PATH="${datasets_path}/dapo/dapo-math-17k.parquet"
export TEST_PATH="${datasets_path}/dapo/aime-2024.parquet"
export NUM_GPU=6

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
