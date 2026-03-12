# VLM OCR Mode Tasks

Each mode injects a fixed primary task before the user message to prevent raw text transcription and guide the model's output behavior.

## Implementation

The task is prepended to the user message in `_stream_from_model` inside `aws_bedrock_converse_provider.py`:

```python
effective_message = f"{task}\nAdditional instructions (respond in the exact same language as these instructions): {message}"
```

If no user message is provided, the task is used alone with a Spanish default.

---

## Modes

### 1. `vlm_qa_over_text` *(default)*
**Task:** Answer a question about the document.
```
Answer the following question about this document.
Additional instructions (respond in the exact same language as these instructions): {message}
```
- User message is **required** — it is the question.
- Message passed as-is; the task frames it as a question over a document.

---

### 2. `vlm_extract_fields`
**Task:** Extract specific fields listed by the user.
```
Extract the following fields from this document.
Additional instructions (respond in the exact same language as these instructions): {message}
```
- User message is **required** — it lists the fields to extract (e.g. "Name, Date, Amount").
- Model targets only the fields specified.

---

### 3. `vlm_summarize_doc`
**Task:** Summarize the document content.
```
Summarize this document.
Additional instructions (respond in the exact same language as these instructions): {message}
```
- User message is **optional** — can add focus (e.g. "focus on dates and totals").
- Without a message, the model produces a general summary.

---

### 4. `vlm_ocr_clean`
**Task:** Extract and structure all visible information.
```
Extract and structure all information visible in this document.
Additional instructions (respond in the exact same language as these instructions): {message}
```
- User message is **optional** — additional instructions or language modifiers.
- Without a message, the model extracts everything and defaults to Spanish.
- This task prevents character-by-character transcription and the repetition loop.

---

## Notes

- The `ocr_mode` field must be sent from the frontend (`vlmMode` in `CommandContext`) through `VlmMessageRequest` to the provider for mode-specific task injection.
- Currently all VLM calls use the `vlm_ocr_clean` task as default until the full chain is wired.
- The repetition loop is caused by temperature 0 + greedy decoding on overlapping text. `repetition_penalty` in `Qwen3VLConfig` is the hardware-level fix pending implementation.
