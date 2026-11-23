from __future__ import annotations

import argparse

from .tokenizer import OwnTokenizer, train_own_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer from scratch")
    parser.add_argument("--input", nargs="+", required=True, help="One or more UTF-8 corpus files")
    parser.add_argument("--output-dir", required=True, help="Directory for tokenizer files")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--model-name", default="parteek")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = train_own_tokenizer(
        args.input,
        args.output_dir,
        args.vocab_size,
        args.model_name,
    )
    tokenizer = OwnTokenizer(model_path)
    print(f"Tokenizer saved: {model_path}")
    print(f"Final vocabulary size: {tokenizer.vocab_size:,}")
    sample = "Hello Parteek! नमस्ते, आपका मॉडल तैयार है।"
    ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
    print(f"Sample IDs: {ids[:30]}")
    print(f"Decoded: {tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()

