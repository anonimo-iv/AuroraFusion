# run on Aurora 
export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export TRAIN_DATA_DIR="/home/binkma/bm_dif/AuroraFusion/Dataset/CT_Brain/train"  # 
export TRAIN_DATA_DIR_VAL="/home/binkma/bm_dif/AuroraFusion/Dataset/CT_Brain/val"  # validation data directory
export OUTPUT_DIR="trained-sd3"
export PYTORCH_ENABLE_XPU_FALLBACK=1
export CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=0

# cat $PBS_NODEFILE > hostfile
# sed -e 's/$/ slots=12/' -i hostfile

NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=12
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"

export NUMEXPR_MAX_THREADS=128

mpiexec \
  --verbose \
  --envall \
  -n "${NGPUS}" \
  --ppn "${NGPU_PER_HOST}" \
  --hostfile="${PBS_NODEFILE}" \
  python train_sd3_unconditional_aurora_ds.py \
  --pretrained_model_name_or_path "$MODEL_NAME"\
  --train_data_dir "$TRAIN_DATA_DIR" \
  --val_data_dir "$TRAIN_DATA_DIR_VAL" \
  --output_dir "$OUTPUT_DIR" \
  --resolution 512 \
  --deepspeed_config ./ds_config.json \
  --train_batch_size 60 \
  --max_train_steps 50 \
  --validation_steps 60 \
  --checkpointing_steps 500 \
  # -use_ipex_optimize \