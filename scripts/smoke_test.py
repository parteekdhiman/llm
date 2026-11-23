from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from parteek_llm.config import ModelConfig  # noqa: E402
from parteek_llm.model import ParteekLLM  # noqa: E402
from parteek_llm.rag import DocumentChunk, TfidfRetriever  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=512,
        block_size=64,
        n_layers=2,
        n_heads=4,
        d_model=64,
        ffn_dim=128,
        dropout=0.0,
    )
    model = ParteekLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    targets = torch.randint(0, config.vocab_size, (2, 32))
    _, loss_before = model(input_ids, targets)
    assert loss_before is not None
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    _, loss_after = model(input_ids, targets)
    assert loss_after is not None and torch.isfinite(loss_after)

    generated = model.generate(input_ids[:1, :4], max_new_tokens=4, temperature=0)
    assert generated.shape == (1, 8)

    retriever = TfidfRetriever(
        [
            DocumentChunk("english.pdf", 1, "Cloud computing provides online servers."),
            DocumentChunk("hindi.pdf", 2, "भारत की राजधानी नई दिल्ली है।"),
        ]
    )
    result = retriever.search("भारत की राजधानी", top_k=1)[0]
    assert result.chunk.source == "hindi.pdf"
    print("Smoke test passed.")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Loss before: {float(loss_before):.4f} | after one step: {float(loss_after):.4f}")


if __name__ == "__main__":
    main()
