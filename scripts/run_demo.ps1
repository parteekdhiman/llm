$ErrorActionPreference = "Stop"

Write-Host "Training demonstration tokenizer..." -ForegroundColor Cyan
python -m parteek_llm.train_tokenizer `
  --input data/raw/sample_corpus.txt `
  --output-dir tokenizer_demo `
  --vocab-size 512

Write-Host "Running ten tiny pretraining steps..." -ForegroundColor Cyan
python -m parteek_llm.pretrain `
  --config configs/smoke_test.json `
  --tokenizer tokenizer_demo/parteek.model `
  --train-corpus data/raw/sample_corpus.txt `
  --output-dir checkpoints/demo_base

Write-Host "Running ten tiny instruction-tuning steps..." -ForegroundColor Cyan
python -m parteek_llm.sft `
  --config configs/smoke_test.json `
  --tokenizer tokenizer_demo/parteek.model `
  --base-checkpoint checkpoints/demo_base/latest.pt `
  --data data/instructions/sample_instructions.jsonl `
  --output-dir checkpoints/demo_chat `
  --max-steps 10

Write-Host "Demo pipeline completed." -ForegroundColor Green
Write-Host "The demo data is too small for useful answers; it only verifies the pipeline."
Write-Host "Start chat with:"
Write-Host "python -m parteek_llm.chat --checkpoint checkpoints/demo_chat/latest.pt --tokenizer tokenizer_demo/parteek.model"

