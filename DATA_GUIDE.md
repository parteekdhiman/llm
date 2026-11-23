# Data Guide

Model quality is controlled more by data quality than by adding layers.

## Pretraining corpus

Use UTF-8 plain text with one useful sentence or paragraph per line. Keep a
balanced mixture of:

- natural English prose;
- natural Hindi in Devanagari;
- a smaller amount of Hinglish if you want it;
- question-answer and dialogue-style text;
- material from the domains your assistant should understand.

For an initial 13–25M parameter experiment, start with 50–100 million tokens.
More clean and diverse data usually improves generalization, but training will
take longer.

## Data rights and privacy

Only train on content you created, licensed, received permission to use, or
that is genuinely public domain/openly licensed for your intended use. Retain
source and license records. Remove:

- passwords, secrets, API keys, and private URLs;
- phone numbers, email addresses, government IDs, and financial records;
- private conversations and confidential documents;
- duplicated pages and generated spam;
- corrupted text and navigation boilerplate.

## Language balance

A practical starting mixture is 50% English, 40% Hindi, and 10% high-quality
bilingual or Hinglish material. Measure token counts after tokenizer training;
file sizes are not directly comparable across scripts.

## Splits

Deduplicate before splitting. Reserve approximately 1–2% of documents for
validation. Never copy lines between training and validation files.

## Instruction data

Use JSONL with `system`, `user`, and `assistant` strings. Include:

- English questions with English answers;
- Hindi questions with Hindi answers;
- translation and language-switching requests;
- grounded context questions in the same format used by PDF chat;
- examples where the assistant says it does not know;
- safe refusals for clearly harmful requests.

Review instruction answers manually. A small set of excellent conversations is
more useful than a large set of incorrect or repetitive conversations.

## PDF grounding examples

Teach the model to use context with examples like:

```json
{"system":"Answer only from the supplied context. Say you do not know when the context is insufficient.","user":"<context>नई दिल्ली भारत की राजधानी है।</context>\nQuestion: भारत की राजधानी क्या है?","assistant":"संदर्भ के अनुसार, भारत की राजधानी नई दिल्ली है।"}
```

