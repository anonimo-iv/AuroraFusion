#!/bin/bash

# Set environment variables
export MASTER_PORT=29500
export LOGLEVEL=INFO

# XPU optimizations
# export IPEX_XPU_ONEDNN_LAYOUT=1
# export IPEX_OFFLINE_COMPILER=1
# export SYCL_CACHE_PERSISTENT=1
# export SYCL_DEVICE_FILTER='level_zero:*'

# # CCL settings for multi-node
# export CCL_BACKEND=native
# export CCL_ATL_TRANSPORT=ofi
# export FI_PROVIDER=cxi
# export CCL_ZE_IPC_EXCHANGE=sockets
# export CCL_ZE_ENABLE=1

# Training parameters
export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export TRAIN_DATA_DIR="/home/binkma/bm_dif/AuroraFusion/Dataset/CT_Brain/train"  # training data directory
# VAL_DATA_DIR="/path/to/your/validation/images"  # Optional
export VAL_DATA_DIR="/home/binkma/bm_dif/AuroraFusion/Dataset/CT_Brain/val"  # validation data directory
export OUTPUT_DIR="/home/binkma/bm_dif/AuroraFusion/Output/sd3-unconditional-fsdp1"
export PYTORCH_ENABLE_XPU_FALLBACK=1


# Set a log file name using the PBS Job ID if available, otherwise use a timestamp

# export LOG_FILE="${OUTPUT_DIR}/training_log.txt"


# FSDP2 specific parameters
BATCH_SIZE=48 #12 tiles on 6 XPUs per node
GRADIENT_ACCUM_STEPS=1
LEARNING_RATE=1e-4
MIXED_PRECISION="no"  # or "bf16" or "no"
SHARDING_STRATEGY="SHARD_GRAD_OP"  # or "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"

# Calculate number of GPUs

NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=12
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"

echo "Running on $NNODES nodes with $NTOTRANKS total XPUs"

# Launch training with mpiexec
# cd $PBS_O_WORKDIR

mpiexec -n "${NGPUS}" \
    --ppn "${NGPU_PER_HOST}" \
    --hostfile="${PBS_NODEFILE}" \
    python train_sd3_unconditional_aurora_ddp.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --train_data_dir $TRAIN_DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VAL_DATA_DIR \
    --resolution 512 \
    --center_crop \
    --random_flip \
    --train_batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    --mixed_precision $MIXED_PRECISION \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
    --checkpointing_steps 500 \
    --validation_steps 100 \
    --seed 42 \
    --max_train_steps 50 \
    --use_ipex_optimize  

echo "Training completed!"