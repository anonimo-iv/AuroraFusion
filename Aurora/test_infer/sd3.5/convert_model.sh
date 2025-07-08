# optimum-cli export openvino \
#   --model tensorart/stable-diffusion-3.5-medium-turbo \
#   --task stable-diffusion \
#   --weight-format fp16 \
#   /lus/flare/projects/hp-ptycho/binkma/models/sd3.5

python OVIR_convert.py \
    --model_name_or_path "tensorart/stable-diffusion-3.5-medium-turbo" \
    --output_dir "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5_2" \
    --dtype "fp16" \
    --task "stable-diffusion"