from __future__ import annotations

import argparse

import torch

from .checkpoint import load_model_from_checkpoint
from .generation import Message, generate_reply
from .rag import TfidfRetriever, discover_pdfs, extract_pdf_chunks, format_retrieved_context
from .tokenizer import OwnTokenizer


PDF_SYSTEM_PROMPT = (
    "You are Parteek's bilingual PDF assistant. Answer only from the supplied context. "
    "If the context is insufficient, say that the answer was not found in the PDFs. "
    "Reply in the language used by the user and never invent a source."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask Parteek Own LLM about local PDFs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 1 or not 0 <= args.min_score <= 1:
        raise ValueError("top-k must be positive and min-score must be between 0 and 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = OwnTokenizer(args.tokenizer)
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match")
    model.eval()

    pdf_paths = discover_pdfs(args.pdf_dir)
    print(f"Reading {len(pdf_paths)} PDF file(s)...")
    chunks = extract_pdf_chunks(pdf_paths)
    retriever = TfidfRetriever(chunks)
    print(f"Indexed {len(chunks):,} passages. Loaded model step {payload['step']:,} on {device}.")
    print("Commands: /reset clears memory, /exit closes chat.")
    history: list[Message] = []

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() == "/exit":
            print("Goodbye!")
            break
        if question.lower() == "/reset":
            history.clear()
            print("Conversation memory cleared.")
            continue

        results = retriever.search(question, args.top_k)
        if not results or results[0].score < args.min_score:
            print(
                "Parteek LLM: I could not find a sufficiently relevant passage in the PDFs. "
                "मुझे PDF में पर्याप्त रूप से संबंधित जानकारी नहीं मिली।"
            )
            continue
        context = format_retrieved_context(results)
        grounded_request = (
            f"<context>\n{context}\n</context>\n"
            f"Question: {question}\n"
            "Answer using only the context and mention the source page when possible."
        )
        request_messages = history[-6:] + [Message("user", grounded_request)]
        answer = generate_reply(
            model,
            tokenizer,
            request_messages,
            system_prompt=PDF_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        if not answer:
            answer = "[The model generated no text. More instruction tuning may be required.]"
        print(f"Parteek LLM: {answer}")
        print("Retrieved sources:")
        for result in results:
            print(
                f"- {result.chunk.source}, page {result.chunk.page} "
                f"(retrieval score {result.score:.3f})"
            )
        history.extend([Message("user", question), Message("assistant", answer)])


if __name__ == "__main__":
    main()
