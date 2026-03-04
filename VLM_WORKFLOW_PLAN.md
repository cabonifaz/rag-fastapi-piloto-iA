# VLM Workflow Plan

Plan for implementing a Vision Language Model (VLM) workflow based on the existing RAG LangGraph architecture, replacing vector DB search with direct image+text inference.

---

## How the Existing RAG Workflow Works

The workflow is a LangGraph `StateGraph` — a directed graph where each node is an async function that reads from and mutates a shared `RAGState` TypedDict. There's no global mutable state between requests; each invocation gets its own state dict passed node-to-node.

### Node Pipeline (in order)

```
validate_inputs
    │ (stop if invalid)
    ▼
get_conversation_history  ← fetches from message service (DynamoDB/DB)
    ▼
clean_message             ← sanitizes input text
    ▼
context_gatekeeper        ← LLM call: does this query need conversation context?
    │
    ├── needs_context=True  → recontextualize_query  ← rewrites query with history
    │                              │
    └── skip ─────────────────────┘
                                   ▼
                            load_rag_config          ← fetches LLM/RAG config from DB
                                   ▼
                            create_or_use_chat       ← creates or reuses a chat session
                                   │ (stop if error)
                                   ▼
                            save_user_message        ← persists user message
                                   │ (stop if error)
                                   ▼
                            determine_query_for_search  ← picks: recontextualized or original
                                   ▼
                            generate_embedding          ← embeds the query
                                   ▼
                            search_vector_db            ← retrieves relevant chunks
                                   ▼
                            build_context               ← concatenates chunks into context_text
                                   ▼
                            select_history_for_prompt   ← trims conversation history
                                   ▼
                            build_rag_prompt            ← assembles final text prompt
                                   ▼
                            prepare_timestamps
                                   ▼
                                  END
```

After the graph finishes, `stream_llm_response()` (called by the endpoint, **outside** the graph) takes the final `state` and actually calls the LLM with the assembled prompt.

---

## VLM Workflow Design

For the VLM version the whole vector search branch is **removed** and replaced with a node that builds a **multimodal prompt** (text + one or more file URLs). Files are uploaded to S3 before the workflow runs (by the API endpoint), their keys are stored permanently and their presigned GET URLs are passed in as initial state.

A message can carry **zero or more attachments** — the design handles all cases uniformly via a list.

### What changes vs the RAG workflow

| RAG workflow | VLM workflow |
|---|---|
| `determine_query_for_search` | removed |
| `generate_embedding` | removed |
| `search_vector_db` | removed |
| `build_context` | removed |
| `build_rag_prompt` | → `build_vlm_prompt` (multimodal) |
| state: `query_embedding`, `search_results`, `context_text`, `query_for_search` | removed |
| state: `attachment_keys`, `attachment_urls` | added |

---

## Implementation

### 1. New `VLMState` — `app/workflows/states.py`

```python
class VLMState(TypedDict):
    # Input parameters
    user_id: int
    message: str
    company_id: int
    area_id: int
    created_at: str
    chat_id: Optional[str]
    request_timezone: Optional[str]
    attachment_keys: List[str]   # S3 keys — stored permanently in DynamoDB
    attachment_urls: List[str]   # presigned GET URLs (60s TTL) — used only in the prompt

    # Processing state
    cleaned_message: Optional[str]
    conversation_history: List[Dict[str, str]]
    gatekeeper_result: Dict[str, bool]
    recontextualized_query: Optional[str]
    rag_config: Optional[Dict[str, Any]]
    new_chat_created: bool
    new_chat_titulo: Optional[str]
    new_chat_timestamp: Optional[str]
    conversation_history_for_prompt: List[Dict[str, str]]
    vlm_prompt: Optional[Any]   # multimodal content list, not a plain string
    assistant_timestamp: Optional[str]
    assistant_timestamp_ms: Optional[int]
    utc_formatted: Optional[str]
    local_formatted: Optional[str]

    # Error handling
    error: Optional[str]
    should_stop: bool
```

### 2. New node: `create_build_vlm_prompt_node` — `app/workflows/nodes/prompt_nodes.py`

```python
def create_build_vlm_prompt_node():
    """Builds a multimodal prompt: one image block per attachment + text message."""
    async def build_vlm_prompt(state: VLMState) -> VLMState:
        attachment_urls = state.get("attachment_urls") or []
        message = state["cleaned_message"]

        # Build one image block per attachment (order: images first, then text)
        # Most VLM-compatible providers (Bedrock/Anthropic/OpenAI) accept this format
        content = [
            {
                "type": "image",
                "source": {
                    "type": "url",   # or "s3" if provider supports it natively
                    "url": url,
                }
            }
            for url in attachment_urls
        ]

        content.append({"type": "text", "text": message})

        state["vlm_prompt"] = content
        return state

    return build_vlm_prompt
```

> **Note:** If the provider uses Bedrock's native S3 format, each image source block would be
> `{"type": "s3", "s3_location": {"uri": f"s3://bucket/{key}"}}` — use `attachment_keys` in that case instead of `attachment_urls`.

### 3. New workflow file — `app/workflows/vlm_workflow.py`

```python
def create_vlm_workflow(
    session_factory,
    llm_provider,        # must support multimodal generate_stream
    message_service,
    ia_config_service,
    context_gatekeeper,
    recontextualizer,
) -> StateGraph:
    validate_inputs         = create_validate_inputs_node()
    get_conversation_history = create_get_conversation_history_node(message_service)
    clean_message           = create_clean_message_node()
    context_gatekeeper_node = create_context_gatekeeper_node(context_gatekeeper)
    recontextualize_query   = create_recontextualize_query_node(recontextualizer)
    load_rag_config         = create_load_rag_config_node(session_factory, ia_config_service)
    create_or_use_chat      = create_create_or_use_chat_node(session_factory)
    save_user_message       = create_save_user_message_node(message_service)
    select_history          = create_select_history_for_prompt_node()
    build_vlm_prompt        = create_build_vlm_prompt_node()   # NEW
    prepare_timestamps      = create_prepare_timestamps_node()

    workflow = StateGraph(VLMState)

    workflow.add_node("validate_inputs",          validate_inputs)
    workflow.add_node("get_conversation_history", get_conversation_history)
    workflow.add_node("clean_message",            clean_message)
    workflow.add_node("context_gatekeeper",       context_gatekeeper_node)
    workflow.add_node("recontextualize_query",    recontextualize_query)
    workflow.add_node("load_rag_config",          load_rag_config)
    workflow.add_node("create_or_use_chat",       create_or_use_chat)
    workflow.add_node("save_user_message",        save_user_message)
    workflow.add_node("select_history",           select_history)
    workflow.add_node("build_vlm_prompt",         build_vlm_prompt)
    workflow.add_node("prepare_timestamps",       prepare_timestamps)

    workflow.set_entry_point("validate_inputs")

    workflow.add_conditional_edges("validate_inputs",
        lambda s: "stop" if s.get("should_stop") else "continue",
        {"continue": "get_conversation_history", "stop": END})

    workflow.add_edge("get_conversation_history", "clean_message")
    workflow.add_edge("clean_message", "context_gatekeeper")

    workflow.add_conditional_edges("context_gatekeeper",
        lambda s: "recontextualize" if s.get("gatekeeper_result", {}).get("needs_context") else "skip",
        {"recontextualize": "recontextualize_query", "skip": "load_rag_config"})

    workflow.add_edge("recontextualize_query", "load_rag_config")
    workflow.add_edge("load_rag_config", "create_or_use_chat")

    workflow.add_conditional_edges("create_or_use_chat",
        lambda s: "stop" if s.get("should_stop") else "continue",
        {"continue": "save_user_message", "stop": END})

    workflow.add_conditional_edges("save_user_message",
        lambda s: "stop" if s.get("should_stop") else "continue",
        {"continue": "select_history", "stop": END})

    # No embedding/search/context nodes — goes straight to prompt building
    workflow.add_edge("select_history",    "build_vlm_prompt")
    workflow.add_edge("build_vlm_prompt",  "prepare_timestamps")
    workflow.add_edge("prepare_timestamps", END)

    return workflow
```

### 4. Updated streaming helper — `app/workflows/helpers/streaming_helpers.py`

The existing `stream_llm_response` reads `state["rag_prompt"]` (a plain string). The VLM version reads `state["vlm_prompt"]` (a multimodal list):

```python
async def stream_vlm_response(state, llm_provider, message_service, db):
    rag_config = state["rag_config"]
    vlm_prompt = state["vlm_prompt"]   # multimodal list, not a string
    messages   = state.get("conversation_history_for_prompt") or None

    async for chunk in llm_provider.generate_stream(
        model_id=rag_config['config']['LLM_MODEL'],
        prompt=vlm_prompt,            # provider must handle list format
        max_tokens=rag_config['config']['LLM_MAX_TOKENS'],
        temperature=rag_config['config']['LLM_TEMPERATURE'],
        top_p=rag_config['config']['LLM_TOP_P'],
        role_behavior=rag_config['config']['ROLE_BEHAVIOR'],
        messages=messages,
        ...
    ):
        ...
```

---

## API Endpoint Flow

```
POST /vlm/query
  multipart: { files: [<file>, ...], message: "...", chat_id: "...", ... }
  │
  ├─ 1. Upload all files to S3 (parallel)
  │       → attachment_keys: ["uploads/chat-5/img1.jpg", "uploads/chat-5/doc.pdf", ...]
  │       → attachment_urls: [presigned_get(key, ExpiresIn=60), ...]
  │
  ├─ 2. build_initial_vlm_state({ ..., attachment_keys, attachment_urls })
  │
  ├─ 3. run VLM workflow graph  →  final VLMState
  │
  └─ 4. stream_vlm_response(state, llm_provider, ...)  →  SSE chunks
```

Key points:
- **S3 uploads happen in the endpoint before the graph starts** — the graph never touches S3.
- `attachment_keys` (permanent) are passed to `save_user_message` → stored in DynamoDB.
- `attachment_urls` (60s presigned GET) are passed to `build_vlm_prompt` → used only in the LLM call.
- When `/v1/messages/chat` returns messages, the service layer generates fresh presigned GET URLs from the stored `attachment_keys` and appends them as `attachments: List[str]` in the response.

### DynamoDB message item (user message with attachments)

```json
{
  "chat_id": "chat-5",
  "created_at": "1709500000000",
  "sender": 0,
  "message": "What do you see in these images?",
  "attachment_keys": ["uploads/chat-5/img1.jpg", "uploads/chat-5/doc.pdf"],
  "id_estado_registro": 1,
  "chat_id#id_estado_registro": "chat-5#1"
}
```

Messages without attachments simply omit `attachment_keys` — no schema change needed.

---

## Checklist Before Implementing

- [ ] Check if `llm_provider.generate_stream()` accepts a `list` as `prompt` for multimodal, or only strings
- [ ] Confirm which image source format the provider expects: presigned HTTPS URL vs native Bedrock S3 URI (`s3://bucket/key`)
- [ ] Verify the target model (e.g. `anthropic.claude-3-5-sonnet`) supports vision via the current SDK wrapper
- [ ] Decide on S3 bucket/prefix for image uploads and IAM permissions
- [ ] Define per-file size/format validation at the endpoint level (max files per message, allowed MIME types)
- [ ] Add `VLMState` to `states.py` and export `create_build_vlm_prompt_node` from `nodes/__init__.py`
- [ ] Register `initialize_vlm_workflow()` in app startup (alongside existing workflow inits)
- [ ] Update `MessageRepository.create_message()` to accept `attachment_keys: Optional[List[str]]`
- [ ] Update the message read path (service layer) to generate presigned GET URLs (`ExpiresIn=60`) from `attachment_keys` before returning the response
- [ ] Update frontend `Message` type: add `attachments?: string[]` (list of presigned GET URLs)
