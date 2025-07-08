# Aurora SD3.5 Multi-GPU Inference

Multi-GPU Stable Diffusion 3.5 inference optimized for Aurora supercomputer with Intel Max 1550 GPUs and OpenVINO GenAI.

## Performance Comparison

**Key Finding**: Device selection (CPU vs GPU) is the primary performance factor, not compilation strategy.

| Method | Device | Avg Time | Speedup |
|--------|--------|----------|---------|
| OpenVINO CPU | CPU | ~14.49s | 1.0x |
| IPEX GPU | Intel GPU | ~8.12s | **1.78x** |
| OpenVINO GPU | Intel GPU | ~2.34s | **6.19x** |
| OpenVINO GPU Pipeline | Intel GPU | ~2.37s | **6.19x** |


- **OpenVINO compiled vs uncompiled**: Nearly identical inference speed once loaded
- **Compilation impact**: Only affects initialization time, not inference performance
- **Multi-GPU scaling**: Linear throughput gains with additional tiles

## Quick Start

```bash
# Basic usage
python openvino_sd3.5.py

# Test GPU discovery
python openvino_sd3.5.py --test

# Check GPU diagnostics
python openvino_sd3.5.py --diagnose

# Run without T5 encoder
python openvino_sd3.5.py --no-t5
```

## Requirements

- Aurora supercomputer access
- OpenVINO Runtime & GenAI
- Python 3.8+, Pillow, numpy

## Model Setup

Place SD3.5 models in directory structure:
```
/lus/flare/projects/xxxxxxxxx/models/sd3.5/
├── transformer/
├── vae_decoder/
├── text_encoder/
├── text_encoder_2/
├── text_encoder_3/  # Optional T5
└── scheduler/
```

## Basic Usage

```python
from pathlib import Path
from openvino_sd3_5 import SD35_Aurora_MultiGPU, ModelConfig, AuroraConfig

# Initialize pipeline
pipeline = SD35_Aurora_MultiGPU(
    model_config=ModelConfig(base_path=Path("/path/to/models")),
    aurora_config=AuroraConfig()
)

# Generate image
image, metadata = pipeline.generate_single(
    prompt="A cyberpunk aurora over Chicago",
    width=1024,
    height=1024,
    num_inference_steps=28
)

image.save("output.png")
```

## Features

- **Multi-GPU**: Auto-discovery of 6x Max 1550 GPUs (12 tile devices)
- **Batch Processing**: Intelligent load balancing with priority queues
- **Performance Monitoring**: Real-time statistics and throughput tracking
- **T5 Support**: Optional enhanced text encoder

## Command Options

| Option | Description |
|--------|-------------|
| `--diagnose` | Show GPU device info |
| `--test` | Test GPU discovery only |
| `--test-single` | Test single GPU pipeline |
| `--no-t5` | Disable T5 text encoder |

## Troubleshooting

- **GPU Not Found**: Use `--diagnose` to check visibility
- **T5 Errors**: Add `--no-t5` flag
- **Memory Issues**: Reduce batch size in config
- **Model Loading**: Verify paths in ModelConfig