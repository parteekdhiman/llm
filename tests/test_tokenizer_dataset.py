from __future__ import annotations

import json
from pathlib import Path

from parteek_llm.dataset import InstructionDataset, TokenBlockDataset, build_token_cache
from parteek_llm.tokenizer import OwnTokenizer, train_own_tokenizer


def test_tokenizer_cache_and_instruction_dataset(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    lines = [
        "Hello, this is a tokenizer training sentence with useful English text.",
        "नमस्ते, यह हिंदी पाठ के लिए एक उपयोगी प्रशिक्षण वाक्य है।",
        "Artificial intelligence learns patterns from clean training examples.",
        "कृत्रिम बुद्धिमत्ता साफ उदाहरणों से पैटर्न सीखती है।",
    ]
    corpus.write_text("\n".join(lines * 50), encoding="utf-8")
    model_path = train_own_tokenizer([corpus], tmp_path / "tokenizer", 512, "test")
    tokenizer = OwnTokenizer(model_path)
    round_trip = tokenizer.decode(tokenizer.encode("Hello नमस्ते"))
    assert "Hello" in round_trip and "नमस्ते" in round_trip

    cache = build_token_cache([corpus], tokenizer, tmp_path / "tokens.bin")
    blocks = TokenBlockDataset(cache, block_size=16)
    input_ids, targets = blocks[0]
    assert input_ids.shape == targets.shape == (16,)

    instructions = tmp_path / "instructions.jsonl"
    record = {
        "system": "Be helpful.",
        "user": "भारत की राजधानी क्या है?",
        "assistant": "भारत की राजधानी नई दिल्ली है।",
    }
    instructions.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = InstructionDataset(instructions, tokenizer, block_size=64)
    sft_input, labels = dataset[0]
    assert sft_input.shape == labels.shape == (64,)
    assert (labels != -100).any()

