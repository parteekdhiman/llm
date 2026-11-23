from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


SPECIAL_TOKENS = ("<system>", "<user>", "<assistant>", "<context>")


class OwnTokenizer:
    """Small wrapper around a tokenizer trained entirely on the user's corpus."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.pad_id = self.processor.pad_id()
        self.unk_id = self.processor.unk_id()
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        if min(self.pad_id, self.unk_id, self.bos_id, self.eos_id) < 0:
            raise ValueError("Tokenizer is missing pad, unknown, BOS, or EOS IDs")

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode([int(token_id) for token_id in ids])

    def token_id(self, piece: str) -> int:
        return self.processor.piece_to_id(piece)


def train_own_tokenizer(
    input_files: list[str | Path],
    output_dir: str | Path,
    vocab_size: int,
    model_name: str = "parteek",
) -> Path:
    files = [Path(path).resolve() for path in input_files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Corpus files not found: {', '.join(missing)}")
    if vocab_size < 512:
        raise ValueError("Use vocab_size >= 512 because byte fallback reserves 256 pieces")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_prefix = destination / model_name

    spm.SentencePieceTrainer.train(
        input=[str(path) for path in files],
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=list(SPECIAL_TOKENS),
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=False,
        remove_extra_whitespaces=True,
        max_sentence_length=8192,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )
    model_path = model_prefix.with_suffix(".model")
    tokenizer = OwnTokenizer(model_path)
    for token in SPECIAL_TOKENS:
        if tokenizer.token_id(token) == tokenizer.unk_id:
            raise RuntimeError(f"Special token was not created correctly: {token}")
    return model_path

