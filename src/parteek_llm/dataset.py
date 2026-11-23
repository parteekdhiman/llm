from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import OwnTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_signature(corpus_files: list[Path], tokenizer: OwnTokenizer) -> dict[str, Any]:
    return {
        "tokenizer_sha256": _sha256(tokenizer.model_path),
        "corpora": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in corpus_files
        ],
    }


def build_token_cache(
    corpus_files: list[str | Path],
    tokenizer: OwnTokenizer,
    cache_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    paths = [Path(path) for path in corpus_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Corpus files not found: {', '.join(missing)}")

    destination = Path(cache_path)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    signature = _cache_signature(paths, tokenizer)
    if destination.is_file() and metadata_path.is_file() and not force:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("signature") == signature:
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    token_count = 0
    buffer: list[int] = []
    with temporary.open("wb") as writer:
        for path in paths:
            with path.open("r", encoding="utf-8", errors="replace") as reader:
                for line in reader:
                    text = line.strip()
                    if not text:
                        continue
                    buffer.extend(tokenizer.encode(text, add_eos=True))
                    if len(buffer) >= 250_000:
                        array = np.asarray(buffer, dtype=dtype)
                        array.tofile(writer)
                        token_count += int(array.size)
                        buffer.clear()
        if buffer:
            array = np.asarray(buffer, dtype=dtype)
            array.tofile(writer)
            token_count += int(array.size)
    temporary.replace(destination)
    metadata = {
        "dtype": np.dtype(dtype).name,
        "token_count": token_count,
        "vocab_size": tokenizer.vocab_size,
        "signature": signature,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return destination


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, cache_path: str | Path, block_size: int):
        self.cache_path = Path(cache_path)
        metadata_path = self.cache_path.with_suffix(self.cache_path.suffix + ".json")
        if not self.cache_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Token cache or metadata is missing: {self.cache_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.tokens = np.memmap(
            self.cache_path,
            mode="r",
            dtype=np.dtype(self.metadata["dtype"]),
        )
        self.block_size = block_size
        self.block_count = max(0, (len(self.tokens) - 1) // block_size)
        if self.block_count < 1:
            raise ValueError(
                f"Cache has {len(self.tokens)} tokens; at least {block_size + 1} are required"
            )

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.block_count:
            raise IndexError(index)
        start = index * self.block_size
        values = np.asarray(
            self.tokens[start : start + self.block_size + 1], dtype=np.int64
        ).copy()
        return torch.from_numpy(values[:-1]), torch.from_numpy(values[1:])


def format_instruction(system: str, user: str) -> str:
    return (
        f"<system>\n{system.strip()}\n"
        f"<user>\n{user.strip()}\n"
        "<assistant>\n"
    )


class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: OwnTokenizer,
        block_size: int,
        default_system: str = "You are a helpful English and Hindi assistant.",
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Instruction file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    user = str(record["user"]).strip()
                    assistant = str(record["assistant"]).strip()
                    system = str(record.get("system", default_system)).strip()
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid JSONL record at {path}:{line_number}: {exc}"
                    ) from exc
                if not user or not assistant:
                    continue
                self.examples.append(self._encode_example(system, user, assistant))
        if not self.examples:
            raise ValueError(f"No usable instruction examples found in {path}")

    def _encode_example(
        self, system: str, user: str, assistant: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(
            format_instruction(system, user), add_bos=True
        )
        response_ids = self.tokenizer.encode(assistant, add_eos=True)
        max_sequence = self.block_size + 1
        response_reserve = min(128, max(8, self.block_size // 3))
        max_prompt = max_sequence - response_reserve
        if len(prompt_ids) > max_prompt:
            prompt_ids = [self.tokenizer.bos_id] + prompt_ids[-(max_prompt - 1) :]
        available_response = max_sequence - len(prompt_ids)
        response_ids = response_ids[:available_response]
        if not response_ids:
            raise ValueError("Instruction prompt leaves no room for an assistant response")
        response_ids[-1] = self.tokenizer.eos_id

        full = prompt_ids + response_ids
        input_ids = full[:-1]
        labels = full[1:]
        labels[: len(prompt_ids) - 1] = [-100] * (len(prompt_ids) - 1)
        pad_length = self.block_size - len(input_ids)
        if pad_length > 0:
            input_ids.extend([self.tokenizer.pad_id] * pad_length)
            labels.extend([-100] * pad_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def optimizer_steps_per_epoch(
    dataset_length: int, micro_batch_size: int, accumulation_steps: int
) -> int:
    batches = math.ceil(dataset_length / micro_batch_size)
    return math.ceil(batches / accumulation_steps)
