from __future__ import annotations

import argparse

import torch

from .checkpoint import load_model_from_checkpoint
from .generation import DEFAULT_SYSTEM_PROMPT, Message, generate_reply
from .tokenizer import OwnTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with Parteek Own LLM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = OwnTokenizer(args.tokenizer)
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match")
    model.eval()
    history: list[Message] = []
    print(f"Loaded step {payload['step']:,} on {device}.")
    print("Commands: /reset clears memory, /exit closes chat.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_text:
            continue
        if user_text.lower() == "/exit":
            print("Goodbye!")
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("Conversation memory cleared.")
            continue
        history.append(Message("user", user_text))
        answer = generate_reply(
            model,
            tokenizer,
            history,
            system_prompt=args.system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        if not answer:
            answer = "[The model generated no text. More training may be required.]"
        print(f"Parteek LLM: {answer}")
        history.append(Message("assistant", answer))


if __name__ == "__main__":
    main()
