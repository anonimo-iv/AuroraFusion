export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export TRAIN_DATA_DIR="/home/binkma/bm_dif/diffusers/workspace/sd3_lora_colab/CT_Brain/train"  # 
export TRAIN_DATA_DIR_VAL="/home/binkma/bm_dif/diffusers/workspace/sd3_lora_colab/CT_Brain/val"  # 
export OUTPUT_DIR="trained-sd3"

deepspeed --include localhost:0,1,2,3 --master_port=29501 train_uncondition_sd3_ds.py \
  --deepspeed_config="ds_config.json" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --val_data_dir="$TRAIN_DATA_DIR_VAL" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --train_batch_size=4 \
  --max_train_steps=40 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --mixed_precision="bf16" \
  --validation_steps=10 \

