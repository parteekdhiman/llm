from __future__ import annotations

import argparse

import torch

from .config import load_config
from .model import ParteekLLM
from .tokenizer import OwnTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model size before training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    project = load_config(args.config)
    tokenizer = OwnTokenizer(args.tokenizer)
    project.model.vocab_size = tokenizer.vocab_size
    model = ParteekLLM(project.model)
    parameters = model.parameter_count()
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Parameters: {parameters:,} ({parameters / 1_000_000:.2f}M)")
    print(f"FP32 weight storage: {parameters * 4 / (1024 ** 2):.1f} MiB")
    print(f"FP16 weight storage: {parameters * 2 / (1024 ** 2):.1f} MiB")
    print(f"Context length: {project.model.block_size}")
    effective_batch = (
        project.training.micro_batch_size
        * project.training.gradient_accumulation_steps
    )
    print(f"Effective batch sequences: {effective_batch}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {properties.total_memory / (1024 ** 3):.1f} GiB")
    else:
        print("CUDA GPU not detected. Training will be very slow on CPU.")


if __name__ == "__main__":
    main()
