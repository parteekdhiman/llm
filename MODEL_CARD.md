# Parteek Own LLM — Model Card Template

Complete this file for every checkpoint you publish.

## Model

- Name: Parteek Own LLM
- Architecture: decoder-only causal Transformer
- Languages: English, Hindi, optional Hinglish
- Parameter count: run `python -m parteek_llm.inspect_model ...`
- Context length: record the training configuration
- Tokenizer: locally trained SentencePiece BPE

## Training

- Base checkpoint: randomly initialized
- Data sources and licenses: TODO
- Training tokens: TODO
- GPU and duration: TODO
- Instruction dataset: TODO

## Evaluation

- Validation loss/perplexity: TODO
- English prompt results: TODO
- Hindi prompt results: TODO
- PDF grounding accuracy: TODO
- Safety evaluation: TODO

## Intended use

Educational experimentation, small-domain text generation, bilingual chat, and
grounded answers from user-provided PDFs.

## Limitations

This small model may hallucinate, memorize data, misunderstand instructions,
produce biased or unsafe text, and fail on reasoning tasks. It is not suitable
for high-stakes decisions without independent verification and substantial
additional safety work.

