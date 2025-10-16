# TODO: Refactor Dependency Injection Pattern

## Problem Statement

Currently, the dependency injection pattern is mixing **singleton services** with **request-scoped resources** (database sessions), leading to architectural issues and potential bugs.

### Current Issues:

1. **Services created per request instead of singleton**
   - `RagService` is instantiated on every HTTP request
   - Wasteful memory and CPU usage
   - Should be created once and reused

2. **Database session passed at initialization**
   - `db` session is request-scoped (new session per request)
   - Being injected into service `__init__` method
   - Forces service recreation per request

3. **Repository initialization in `__init__`**
   - `ChatRepository` is created in `RagService.__init__`
   - Tightly couples service lifecycle to database session lifecycle
   - Prevents service from being singleton

### Current Implementation:

**File: `app/core/container.py`**
```python
def get_full_rag_chat_service(self, db=None) -> tuple[RagService, LLMPort]:
    # Creates NEW service instance per request - BAD
    rag_service = RagService(
        embeddings_provider=embeddings_provider,
        vectorstore=vectorstore,
        llm_provider=llm_provider,
        orchestrator=orchestrator,
        db=db  # Request-scoped resource passed to constructor
    )
    return rag_service, llm_provider
```

**File: `app/services/rag_service.py`**
```python
class RagService:
    def __init__(self, embeddings_provider, vectorstore, llm_provider, orchestrator, db=None):
        # ... other dependencies
        self.db = db  # Request-scoped - WRONG
        self.chat_repository = ChatRepository(db) if db else None  # Created in __init__ - WRONG
```

**File: `app/api/rag.py`**
```python
def get_full_rag_dependencies(db: Session = Depends(get_db)):
    # Creates new service per request
    rag_service, llm_provider = container.get_full_rag_chat_service(db=db)
    return rag_service, llm_provider
```

---

## Proposed Solution

### Architecture Principles:

1. **Stateless Services = Singleton**
   - Services should be stateless and created once
   - Reused across all requests

2. **Request-Scoped Resources = Method Parameters**
   - Database sessions should be passed to methods, not constructors
   - Repositories created per-request, not per-service

3. **Clear Separation of Concerns**
   - Container manages singleton services
   - FastAPI Depends() manages request-scoped resources

### Target Implementation:

**File: `app/core/container.py`**
```python
class DIContainer:
    def __init__(self):
        self._embeddings_provider = None
        self._vectorstore = None
        self._llm_provider = None
        self._orchestrator_analyzer = None
        self._rag_service = None  # Add singleton rag_service

    def get_rag_service(self) -> RagService:
        """Get rag service as singleton (NO db parameter)."""
        if self._rag_service is None:
            embeddings_provider = self.get_embeddings_provider()
            vectorstore = self.get_vectorstore()
            llm_provider = self.get_llm_provider()
            orchestrator = self.get_orchestrator_analyzer()

            # Create ONCE - no db parameter
            self._rag_service = RagService(
                embeddings_provider=embeddings_provider,
                vectorstore=vectorstore,
                llm_provider=llm_provider,
                orchestrator=orchestrator
            )

        return self._rag_service

    def get_llm_provider(self) -> LLMPort:
        # Keep as singleton
        if self._llm_provider is None:
            self._llm_provider = AWSBedrockConverseProvider(...)
        return self._llm_provider
```

**File: `app/services/rag_service.py`**
```python
class RagService:
    def __init__(self, embeddings_provider, vectorstore, llm_provider, orchestrator):
        """Initialize stateless service - NO db parameter."""
        self.embeddings_provider = embeddings_provider
        self.vectorstore = vectorstore
        self.llm_provider = llm_provider
        self.orchestrator = orchestrator
        self.message_service = MessageService()
        self.context_counter = ContextMessageCounter()
        # NO self.db
        # NO self.chat_repository

    async def process_rag_query_stream(
        self,
        db: Session,  # Pass db to method, not constructor
        user_id: int,
        user: str,
        message: str,
        # ... other parameters
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process RAG query with streaming."""

        # Create repository PER REQUEST
        chat_repository = ChatRepository(db)

        # Use repository
        if chat_id is None:
            new_chat_id = chat_repository.create_chat(
                id_usuario=user_id,
                id_area=area_id,
                id_empresa=company_id,
                titulo=titulo
            )

        # Update chat timestamp
        chat_repository.update_ultimo_mensaje_fecha(chat_id)

        # ... rest of logic
```

**File: `app/api/rag.py`**
```python
def get_rag_service():
    """Get singleton RagService from container."""
    return container.get_rag_service()

def get_llm_provider():
    """Get singleton LLM provider from container."""
    return container.get_llm_provider()

@router.post("/chat-streaming")
async def chat_streaming_endpoint(
    request: UnifiedRequest,
    rag_service: RagService = Depends(get_rag_service),  # Singleton service
    llm_provider: LLMPort = Depends(get_llm_provider),    # Singleton provider
    db: Session = Depends(get_db),                         # Request-scoped session
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_validation)
):
    """Streaming chat endpoint."""

    async def generate_stream():
        # Pass db to method, not constructor
        async for chunk_data in rag_service.process_rag_query_stream(
            db=db,  # Pass db as parameter
            user_id=request.user_id,
            user=request.user,
            message=request.message,
            # ... other parameters
        ):
            yield f"data: {json.dumps(chunk_data)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
```

---

## Benefits of Refactoring:

### 1. Performance
- Services created once (singleton) instead of per-request
- Reduces memory allocation and garbage collection
- Faster response times

### 2. Correctness
- Clear separation between stateless services and stateful resources
- Database sessions properly scoped to requests
- No risk of session leakage between requests

### 3. Maintainability
- Clearer dependency flow
- Easier to test (mock db at method level, not constructor)
- Follows FastAPI best practices

### 4. Scalability
- Singleton services can be safely shared across async requests
- Request-scoped resources properly isolated
- Thread-safe by design

---

## Migration Steps:

### Phase 1: Refactor RagService
- [ ] Remove `db` parameter from `RagService.__init__`
- [ ] Remove `self.db` and `self.chat_repository` from `__init__`
- [ ] Add `db: Session` parameter to `process_rag_query_stream` method
- [ ] Add `db: Session` parameter to `agent_orchestrator_stream` method
- [ ] Create `ChatRepository` inside methods that need it
- [ ] Update all calls to `load_ia_area_config` to pass `db` parameter

### Phase 2: Refactor Container
- [ ] Add `_rag_service` singleton to `DIContainer`
- [ ] Change `get_full_rag_chat_service()` to `get_rag_service()` (no db parameter)
- [ ] Return singleton `RagService` instead of creating new instance

### Phase 3: Refactor API Endpoints
- [ ] Split `get_full_rag_dependencies()` into separate dependencies
- [ ] Create `get_rag_service()` dependency (singleton)
- [ ] Create `get_llm_provider()` dependency (singleton)
- [ ] Keep `get_db()` dependency (request-scoped)
- [ ] Update endpoint signatures to use separate dependencies
- [ ] Pass `db` to service methods instead of constructor

### Phase 4: Update Other Services (if applicable)
- [ ] Apply same pattern to `ChatService`
- [ ] Apply same pattern to `MessageService`
- [ ] Review all service classes for similar issues

### Phase 5: Testing
- [ ] Unit test services without database
- [ ] Integration test with request-scoped db
- [ ] Load test to verify singleton behavior
- [ ] Verify no session leakage between concurrent requests

---

## Files to Modify:

1. **`app/services/rag_service.py`**
   - Remove `db` from `__init__`
   - Add `db` to method signatures
   - Create repositories in methods

2. **`app/core/container.py`**
   - Add singleton `_rag_service`
   - Refactor `get_full_rag_chat_service()` → `get_rag_service()`

3. **`app/api/rag.py`**
   - Split dependencies
   - Update endpoint signatures
   - Pass `db` to methods

4. **`app/services/chat_service.py` (optional)**
   - Consider same refactoring

5. **`app/services/message_service.py` (optional)**
   - Consider same refactoring

---

## Risk Assessment:

### Low Risk:
- Services are already stateless (no instance variables modified)
- Database sessions already request-scoped
- No shared state between requests

### Medium Risk:
- Need to update all method signatures
- Multiple files to modify
- Need comprehensive testing

### Mitigation:
- Implement incrementally (phase by phase)
- Test each phase before moving forward
- Keep old code commented for rollback if needed

---

## Priority: **MEDIUM**

This refactoring improves architecture and performance but doesn't fix critical bugs. Current implementation works, but is not optimal.

**Estimated Effort:** 4-6 hours
**Impact:** High (performance, maintainability, correctness)
**Complexity:** Medium
