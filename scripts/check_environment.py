from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.system()})")
    failures: list[str] = []
    for package in ("numpy", "sentencepiece", "pypdf", "sklearn", "tqdm"):
        try:
            module = __import__(package)
            print(f"{package}: {getattr(module, '__version__', 'installed')}")
        except ImportError:
            failures.append(package)
            print(f"{package}: MISSING")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(f"GPU: {properties.name}")
            print(f"VRAM: {properties.total_memory / (1024 ** 3):.1f} GiB")
        else:
            failures.append("CUDA-enabled PyTorch")
    except ImportError:
        failures.append("torch")
        print("torch: MISSING")

    if failures:
        print("\nEnvironment is not ready. Missing: " + ", ".join(failures))
        raise SystemExit(1)
    print("\nEnvironment is ready.")


if __name__ == "__main__":
    main()

