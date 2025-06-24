# training example for Stable Diffusion 3.5 (SD3.5)

Task -- unconditional generation

The `train_uncondition_sd3.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). We also provide a LoRA implementation in the `train_uncondition_lora_sd3.py` script.

> [!NOTE]
> As the model is gated, before using it with diffusers you first need to go to the [Stable Diffusion 3 Medium Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

Then cd in the `/Polaris/sd3.5` folder and run
```bash
conda env create -f environment.yml
```

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

We provide two choices to run the training scrpts:

### run with acclerate


And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```
Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

then run

```bash
bash train_uncon_acc.sh
```

### run with Deepspeed

```bash
bash train_uncon_ds.sh
```



### Dog toy example

Now let's get our dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

### X-ray brain images example

For this example we will use some Xray brain images: https://www.kaggle.com/datasets/kmader/siim-medical-images?select=tiff_images

The images are converted into jpeg format. 

### train with Accelerate
 
Now, we can launch training using :

```bash
export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
export TRAIN_DATA_DIR="/home/binkma/bm_dif/diffusers/workspace/sd3_lora_colab/dog"  # 
export OUTPUT_DIR="trained-sd3"

accelerate launch --num_processes=4 train_uncondition_sd3.py \
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
```

### train with Deepspeed

Setup deepspeed configuration in the file `ds_config.json`


Launch the training script `train_uncon_sd.sh` with deepspeed command

```bash
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
```



