# Training example for Stable Diffusion 3.5 (Aurora)

Task -- unconditional generation 


### Installing the dependencies

Aurora use python venv for package management,

Create your venv environment and actviate it

```bash
module use /soft/modulefiles/
module load frameworks
python3 -m venv /path/to/new/venv --system-site-packages
source /path/to/new/venv
```

Install all the libs in the new venv

```bash
pip install -r requirements.txt
```


### train with Deepspeed

> [!NOTE]
> As Accelerate does not support training on Aurora, we use deepspeed for the distributed training framework.

Setup deepspeed configuration in the file `ds_config.json`

Launch the training script `train_ds.sh` with deepspeed command


### train with FSDP1

Launch the training script `train_fsdp.sh` 





