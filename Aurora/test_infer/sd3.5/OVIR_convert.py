import argparse
from pathlib import Path
# from optimum.exporters.openvino import export, export_from_model
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
import openvino


def main():
    """
    A generalized script to export Hugging Face models to the OpenVINO format.
    """
    parser = argparse.ArgumentParser(description="Export a Hugging Face model to OpenVINO format.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path of the model to export (e.g., 'tensorart/stable-diffusion-3.5-medium-turbo')."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="The directory where the exported model will be saved."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="The data type (precision) for the exported model (e.g., 'fp16', 'fp32')."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="stable-diffusion",
        help="The task for which the model is being exported (e.g., 'stable-diffusion', 'text-generation')."
    )
    # Add other export arguments as needed, for example, to handle different input shapes
    # See the documentation for main_export for a full list of possible arguments.

    args = parser.parse_args()

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model: {args.model_name_or_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.dtype}")
    print(f"Task: {args.task}")

    # The main_export function from Optimum is designed to handle command-line style arguments.
    # We can construct the arguments list and pass it directly.
    # export_command = [
    #     f"--model_name_or_path={args.model_name_or_path}",
    #     f"--output={args.output_dir}",
    #     f"--task={args.task}",
    #     f"--dtype={args.dtype}",
    #     # Add other arguments here as needed
    # ]

    # You can either call the main_export function which is designed for command-line execution
    # main_export(export_command)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    ovirno_model = export_from_model(
        pipeline,
        output=args.output_dir,
        task=args.task,
        # dtype=args.dtype,
        # Add other parameters as needed
    )
    print("\nExport complete!")


if __name__ == "__main__":
    main()