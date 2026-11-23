`# Parteek Own LLM

Train a small English/Hindi conversational language model with **randomly
initialized weights** on a 4–8 GB NVIDIA GPU. The project contains the complete
pipeline: corpus cleaning, tokenizer training, base-model pretraining,
instruction tuning, terminal chat, and PDF question answering.

This is an educational small language model. It can become useful in a narrow
domain with enough high-quality data, but it will not match ChatGPT-scale
systems.

## What is genuinely yours

- The SentencePiece tokenizer is trained on your corpus.
- Every neural-network weight starts from a random value.
- No pretrained language model or embedding model is downloaded.
- PDF retrieval uses character TF-IDF, which has no learned neural weights.
- Your checkpoints, instruction data, prompts, and inference code remain local.

## Architecture

- Decoder-only causal Transformer
- RMSNorm, rotary position embeddings, causal self-attention, and SwiGLU
- Tied input/output token embeddings
- Mixed-precision CUDA training
- Gradient accumulation, clipping, warmup, cosine decay, validation, and resume
- Two practical profiles:
  - `configs/small_4gb.json`: about 13M parameters
  - `configs/medium_8gb.json`: about 25M parameters

The exact parameter count depends on the tokenizer's final vocabulary size.

## 1. Requirements

- Windows or Linux
- Python 3.11 or 3.12 recommended
- NVIDIA GPU with 4–8 GB VRAM recommended
- Recent NVIDIA driver
- At least 20 GB free disk for experiments; large corpora need more

Create and activate an environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch build recommended for your system at
<https://pytorch.org/get-started/locally/>, then install this project:

```bash
pip install -e ".[dev]"
```

Verify the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Prepare your corpus

Place legally usable UTF-8 `.txt` files under `data/raw/`. Use a balanced mix
of English and Hindi. Each paragraph or sentence should be on its own line.

A very small demonstration corpus is included, but it is only for checking the
pipeline. Clean and combine your real files:

```bash
python -m parteek_llm.prepare_corpus \
  --input data/raw \
  --output data/cleaned_corpus.txt
```

Windows PowerShell can use the same command on one line.

Recommended first real experiment: 50–100 million tokens, balanced across the
languages and domains you need. Remove duplicated, private, copyrighted-without-
permission, low-quality, and unsafe material. See `DATA_GUIDE.md`.

## 3. Train your tokenizer

The production tokenizer target is 16,000 pieces:

```bash
python -m parteek_llm.train_tokenizer \
  --input data/cleaned_corpus.txt \
  --output-dir tokenizer \
  --vocab-size 16000
```

For the included tiny sample, request 512 pieces instead:

```bash
python -m parteek_llm.train_tokenizer \
  --input data/raw/sample_corpus.txt \
  --output-dir tokenizer_demo \
  --vocab-size 512
```

The tokenizer defines `<pad>`, `<unk>`, `<s>`, `</s>`, `<system>`, `<user>`,
`<assistant>`, and `<context>`.

## 4. Inspect the model before training

For a 4 GB GPU:

```bash
python -m parteek_llm.inspect_model \
  --config configs/small_4gb.json \
  --tokenizer tokenizer/parteek.model
```

For an 8 GB GPU, use `configs/medium_8gb.json`.

## 5. Pretrain the base model

```bash
python -m parteek_llm.pretrain \
  --config configs/small_4gb.json \
  --tokenizer tokenizer/parteek.model \
  --train-corpus data/cleaned_corpus.txt \
  --output-dir checkpoints/base
```

Optional held-out validation corpus:

```bash
python -m parteek_llm.pretrain \
  --config configs/small_4gb.json \
  --tokenizer tokenizer/parteek.model \
  --train-corpus data/train.txt \
  --val-corpus data/validation.txt \
  --output-dir checkpoints/base
```

Resume an interrupted run:

```bash
python -m parteek_llm.pretrain \
  --config configs/small_4gb.json \
  --tokenizer tokenizer/parteek.model \
  --train-corpus data/cleaned_corpus.txt \
  --output-dir checkpoints/base \
  --resume checkpoints/base/latest.pt
```

If CUDA runs out of memory, reduce `micro_batch_size` to 1, increase
`gradient_accumulation_steps`, or reduce `block_size` to 256.

## 6. Instruction-tune for conversation

Instruction data is JSON Lines, one object per line:

```json
{"system":"You are a helpful bilingual assistant.","user":"भारत की राजधानी क्या है?","assistant":"भारत की राजधानी नई दिल्ली है।"}
```

Replace the sample with a large, reviewed bilingual dataset that you have the
right to use. Then run:

```bash
python -m parteek_llm.sft \
  --config configs/small_4gb.json \
  --tokenizer tokenizer/parteek.model \
  --base-checkpoint checkpoints/base/latest.pt \
  --data data/instructions/train.jsonl \
  --output-dir checkpoints/chat
```

The included `sample_instructions.jsonl` only demonstrates the format.

## 7. Chat with your model

```bash
python -m parteek_llm.chat \
  --checkpoint checkpoints/chat/latest.pt \
  --tokenizer tokenizer/parteek.model
```

Commands inside chat:

- `/reset` clears conversation memory.
- `/exit` closes the program.

## 8. Ask questions about PDFs

Copy PDFs into `data/pdfs/`, then run:

```bash
python -m parteek_llm.pdf_chat \
  --checkpoint checkpoints/chat/latest.pt \
  --tokenizer tokenizer/parteek.model \
  --pdf-dir data/pdfs
```

The retriever supports English and Hindi using character n-grams. It supplies
the most relevant passages to your model. It does not permanently teach PDF
facts to the model, so replacing a PDF requires no retraining.

## 9. Fast validation

Run tests:

```bash
pytest
```

Run an end-to-end model smoke test without a trained checkpoint:

```bash
python scripts/smoke_test.py
```

On Windows, after installing the project and CUDA-enabled PyTorch, you can run
the entire tiny demonstration pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

This demonstration only checks that every stage runs. Its tiny data cannot
produce a useful conversational model.

## Training roadmap

1. Confirm the demo tokenizer and smoke test work.
2. Train on a small corpus for several hundred steps.
3. Verify that loss decreases and generation produces the correct scripts.
4. Scale to a clean 50–100M-token corpus.
5. Build at least thousands of reviewed English/Hindi instruction examples.
6. Evaluate on held-out prompts before trusting the model.

## Safety and limitations

The program does not automatically moderate, fact-check, or protect personal
data. A small model can hallucinate, repeat training data, produce unsafe text,
or answer incorrectly. Do not use it for medical, legal, financial, emergency,
or other high-stakes decisions without independent verification and additional
safety engineering.
