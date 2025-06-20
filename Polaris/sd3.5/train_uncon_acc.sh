export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export TRAIN_DATA_DIR="/home/binkma/bm_dif/diffusers/workspace/sd3_lora_colab/dog"  # 改名为更通用的TRAIN_DATA_DIR
export OUTPUT_DIR="trained-sd3"


accelerate launch --num_processes=4 train_uncondition_sd3_acc.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --val_data_dir="$TRAIN_DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --train_batch_size=4 \
  --max_train_steps=20 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --mixed_precision="bf16" \
  --validation_steps=10 \

