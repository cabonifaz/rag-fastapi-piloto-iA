# Chat Storage Implementation Plan

## Table of Contents
- [Current Architecture Analysis](#current-architecture-analysis)
- [Database Design](#database-design)
- [Implementation Roadmap](#implementation-roadmap)
- [API Endpoints](#api-endpoints)
- [Request/Response Flow](#requestresponse-flow)
- [Key Design Decisions](#key-design-decisions)

---

## Current Architecture Analysis

### Existing Chat Request Structure (`chat_models.py`)

```python
class UnifiedRequest(BaseModel):
    user_id: str
    message: str
    id_empresa: int                         # Required, company ID
    empresa: str                            # Required, company name for search
    id_area: int                            # Required, area ID
    area: str                               # Required, area name for filtering
    top_k: Optional[int] = None             # Optional, defaults to env config
    similarity_threshold: Optional[float] = None
    alpha: Optional[float] = None           # Hybrid search alpha
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

### Current Authentication
- **JWT-based** with SQL Server backend
- **User table**: `USUARIOS` (existing in `dbo` schema)
- **Stored procedures** for authentication (`SP_VERIFY_USER_PASS`, `SP_USUARIO_LOGIN`)

---

## Database Design

### Table 1: `CHATS` (Chat Sessions)

**Purpose**: Store individual chat sessions/conversations for each user.

```sql
CREATE TABLE dbo.CHATS (
    -- =============================================
    -- Primary Key
    -- =============================================
    ID_CHAT INT IDENTITY(1,1) PRIMARY KEY,

    -- =============================================
    -- Foreign Keys (User, Company, Area)
    -- =============================================
    ID_USUARIO INT NOT NULL,
    ID_EMPRESA INT NOT NULL,                    -- Company ID
    ID_AREA INT NOT NULL,                       -- Area ID

    -- =============================================
    -- Chat Metadata
    -- =============================================
    TITULO NVARCHAR(200) NOT NULL,              -- Chat title (auto-generated or user-defined)

    -- =============================================
    -- Last Message Date
    -- =============================================
    ULTIMO_MENSAJE_FECHA DATETIME NULL,         -- Date/time of last message


    -- =============================================
    -- State
    -- =============================================
    ID_ESTADO_REGISTRO INT DEFAULT 1,                    -- 1=Active, 2=Archived, 3=Deleted

    -- =============================================
    -- Audit Fields
    -- =============================================
    FECHA_CREACION DATETIME DEFAULT GETDATE(),
    FECHA_MODIFICACION DATETIME DEFAULT GETDATE(),
    USUARIO_CREACION NVARCHAR(100) NULL,

);

-- Indexes for performance
CREATE INDEX IX_CHATS_USUARIO ON dbo.CHATS(ID_USUARIO);
CREATE INDEX IX_CHATS_EMPRESA ON dbo.CHATS(ID_EMPRESA);
CREATE INDEX IX_CHATS_AREA ON dbo.CHATS(ID_AREA);
CREATE INDEX IX_CHATS_USUARIO_EMPRESA_AREA ON dbo.CHATS(ID_USUARIO, ID_EMPRESA, ID_AREA);
CREATE INDEX IX_CHATS_ESTADO ON dbo.CHATS(ID_ESTADO);
CREATE INDEX IX_CHATS_FECHA ON dbo.CHATS(FECHA_CREACION DESC);
CREATE INDEX IX_CHATS_ULTIMO_MENSAJE_FECHA ON dbo.CHATS(ULTIMO_MENSAJE_FECHA DESC);
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `ID_CHAT` | INT | Primary key, auto-increment |
| `ID_USUARIO` | INT | Foreign key to USUARIOS table |
| `ID_EMPRESA` | INT | Company ID for access control and filtering |
| `ID_AREA` | INT | Area ID for access control and filtering |
| `TITULO` | NVARCHAR(200) | Chat title (auto-generated from first message or user-defined) |
| `DESCRIPCION` | NVARCHAR(500) | Optional user-provided description |
| `ULTIMO_MENSAJE_FECHA` | DATETIME | Date and time of the last message |
| `TOP_K` | INT | Number of documents to retrieve (NULL = use default) |
| `SIMILARITY_THRESHOLD` | DECIMAL(3,2) | Minimum similarity score (0.00-1.00) |
| `ALPHA` | DECIMAL(3,2) | Hybrid search weight (0.0=keyword, 1.0=vector) |
| `TEMPERATURE` | DECIMAL(3,2) | LLM temperature for response generation |
| `MAX_TOKENS` | INT | Maximum tokens in LLM response |
| `TOTAL_MENSAJES` | INT | Count of messages in this chat |
| `ID_ESTADO` | INT | 1=Active, 2=Archived, 3=Deleted |
| `FECHA_CREACION` | DATETIME | Chat creation timestamp |
| `FECHA_MODIFICACION` | DATETIME | Last modification timestamp |
| `USUARIO_CREACION` | NVARCHAR(100) | User who created the chat |

---

### Table 2: `MENSAJES` (Chat Messages)

**Purpose**: Store individual messages within each chat conversation.

```sql
CREATE TABLE dbo.MENSAJES (
    -- =============================================
    -- Primary Key
    -- =============================================
    ID_MENSAJE BIGINT IDENTITY(1,1) PRIMARY KEY,

    -- =============================================
    -- Foreign Key
    -- =============================================
    ID_CHAT INT NOT NULL,

    -- =============================================
    -- Message Content
    -- =============================================
    TIPO_MENSAJE NVARCHAR(20) NOT NULL,         -- 'user' or 'assistant'
    CONTENIDO NVARCHAR(MAX) NOT NULL,           -- Message text content

    -- =============================================
    -- RAG Context (for assistant messages)
    -- =============================================
    CONTEXTO_USADO NVARCHAR(MAX) NULL,          -- JSON: Retrieved documents used
    FUENTES NVARCHAR(MAX) NULL,                 -- JSON: Source metadata array
    NUM_DOCUMENTOS_RECUPERADOS INT NULL,        -- Number of documents retrieved

    -- =============================================
    -- LLM Metadata (for assistant messages)
    -- =============================================
    MODELO_USADO NVARCHAR(100) NULL,            -- e.g., 'anthropic.claude-3-haiku-20240307-v1:0'
    TOKENS_ENTRADA INT NULL,                    -- Input tokens count
    TOKENS_SALIDA INT NULL,                     -- Output tokens count
    LATENCIA_MS INT NULL,                       -- Response time in milliseconds

    -- =============================================
    -- RAG Parameters Used (actual values for this message)
    -- =============================================
    TOP_K_USADO INT NULL,
    SIMILARITY_THRESHOLD_USADO DECIMAL(3,2) NULL,
    ALPHA_USADO DECIMAL(3,2) NULL,
    TEMPERATURE_USADO DECIMAL(3,2) NULL,
    MAX_TOKENS_USADO INT NULL,

    -- =============================================
    -- Error Tracking
    -- =============================================
    ERROR_MENSAJE NVARCHAR(500) NULL,           -- Error message if failed
    ID_ESTADO INT DEFAULT 1,                    -- 1=Success, 2=Failed, 3=Partial

    -- =============================================
    -- Conversation Context
    -- =============================================
    NUMERO_SECUENCIA INT NOT NULL,              -- Message sequence number in chat
    INCLUIDO_EN_CONTEXTO BIT DEFAULT 1,         -- Was this message included in RAG context?

    -- =============================================
    -- Audit Fields
    -- =============================================
    FECHA_CREACION DATETIME DEFAULT GETDATE(),
    FECHA_MODIFICACION DATETIME NULL,

    -- =============================================
    -- Constraints & Indexes
    -- =============================================
    CONSTRAINT FK_MENSAJES_CHAT FOREIGN KEY (ID_CHAT)
        REFERENCES dbo.CHATS(ID_CHAT) ON DELETE CASCADE,
    CONSTRAINT CK_MENSAJES_TIPO CHECK (TIPO_MENSAJE IN ('user', 'assistant'))
);

-- Indexes for performance
CREATE INDEX IX_MENSAJES_CHAT ON dbo.MENSAJES(ID_CHAT);
CREATE INDEX IX_MENSAJES_CHAT_FECHA ON dbo.MENSAJES(ID_CHAT, FECHA_CREACION ASC);
CREATE INDEX IX_MENSAJES_TIPO ON dbo.MENSAJES(TIPO_MENSAJE);
CREATE INDEX IX_MENSAJES_SECUENCIA ON dbo.MENSAJES(ID_CHAT, NUMERO_SECUENCIA ASC);
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `ID_MENSAJE` | BIGINT | Primary key, auto-increment |
| `ID_CHAT` | INT | Foreign key to CHATS table (cascades on delete) |
| `TIPO_MENSAJE` | NVARCHAR(20) | 'user' or 'assistant' |
| `CONTENIDO` | NVARCHAR(MAX) | Message text content |
| `CONTEXTO_USADO` | NVARCHAR(MAX) | JSON array of retrieved documents used for RAG |
| `FUENTES` | NVARCHAR(MAX) | JSON array of source metadata (file names, pages, etc.) |
| `NUM_DOCUMENTOS_RECUPERADOS` | INT | Count of documents retrieved from vector DB |
| `MODELO_USADO` | NVARCHAR(100) | LLM model identifier used for this response |
| `TOKENS_ENTRADA` | INT | Number of input tokens (prompt + context) |
| `TOKENS_SALIDA` | INT | Number of output tokens (response) |
| `LATENCIA_MS` | INT | Response generation time in milliseconds |
| `TOP_K_USADO` | INT | Actual top_k value used for this message |
| `SIMILARITY_THRESHOLD_USADO` | DECIMAL(3,2) | Actual similarity threshold used |
| `ALPHA_USADO` | DECIMAL(3,2) | Actual alpha value used (hybrid search) |
| `TEMPERATURE_USADO` | DECIMAL(3,2) | Actual temperature used |
| `MAX_TOKENS_USADO` | INT | Actual max_tokens used |
| `ERROR_MENSAJE` | NVARCHAR(500) | Error message if generation failed |
| `ID_ESTADO` | INT | 1=Success, 2=Failed, 3=Partial response |
| `NUMERO_SECUENCIA` | INT | Message order in conversation (1, 2, 3...) |
| `INCLUIDO_EN_CONTEXTO` | BIT | Whether message was included in subsequent RAG contexts |
| `FECHA_CREACION` | DATETIME | Message creation timestamp |

---

## Implementation Roadmap

### Phase 1: Database & Models Setup

#### 1.1 Create SQL Migration Scripts

**File: `migrations/001_create_chats_table.sql`**
```sql
-- Create CHATS table
-- (See full SQL above)
```

**File: `migrations/002_create_mensajes_table.sql`**
```sql
-- Create MENSAJES table
-- (See full SQL above)
```

#### 1.2 Create Stored Procedures

**File: `migrations/003_create_stored_procedures.sql`**

```sql
-- =============================================
-- SP_CHAT_CREATE: Create new chat
-- =============================================
CREATE PROCEDURE dbo.SP_CHAT_CREATE
    @ID_USUARIO INT,
    @ID_EMPRESA INT,
    @ID_AREA INT,
    @TITULO NVARCHAR(200),
    @DESCRIPCION NVARCHAR(500) = NULL,
    @TOP_K INT = NULL,
    @SIMILARITY_THRESHOLD DECIMAL(3,2) = NULL,
    @ALPHA DECIMAL(3,2) = NULL,
    @TEMPERATURE DECIMAL(3,2) = NULL,
    @MAX_TOKENS INT = NULL,
    @USUARIO_CREACION NVARCHAR(100) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    INSERT INTO dbo.CHATS (
        ID_USUARIO, ID_EMPRESA, ID_AREA, TITULO,
        DESCRIPCION, TOP_K, SIMILARITY_THRESHOLD, ALPHA,
        TEMPERATURE, MAX_TOKENS, USUARIO_CREACION
    )
    VALUES (
        @ID_USUARIO, @ID_EMPRESA, @ID_AREA, @TITULO,
        @DESCRIPCION, @TOP_K, @SIMILARITY_THRESHOLD, @ALPHA,
        @TEMPERATURE, @MAX_TOKENS, @USUARIO_CREACION
    );

    -- Return the newly created chat
    SELECT * FROM dbo.CHATS WHERE ID_CHAT = SCOPE_IDENTITY();
END;
GO

-- =============================================
-- SP_CHAT_LIST: List user's chats
-- =============================================
CREATE PROCEDURE dbo.SP_CHAT_LIST
    @ID_USUARIO INT,
    @ID_ESTADO INT = 1,  -- Default: Active chats only
    @PAGE_NUMBER INT = 1,
    @PAGE_SIZE INT = 50
AS
BEGIN
    SET NOCOUNT ON;

    SELECT
        ID_CHAT,
        ID_USUARIO,
        ID_EMPRESA,
        ID_AREA,
        TITULO,
        DESCRIPCION,
        ULTIMO_MENSAJE_FECHA,
        TOTAL_MENSAJES,
        FECHA_CREACION,
        ID_ESTADO
    FROM dbo.CHATS
    WHERE ID_USUARIO = @ID_USUARIO
        AND ID_ESTADO = @ID_ESTADO
    ORDER BY ULTIMO_MENSAJE_FECHA DESC, FECHA_CREACION DESC
    OFFSET (@PAGE_NUMBER - 1) * @PAGE_SIZE ROWS
    FETCH NEXT @PAGE_SIZE ROWS ONLY;
END;
GO

-- =============================================
-- SP_CHAT_GET: Get chat details by ID
-- =============================================
CREATE PROCEDURE dbo.SP_CHAT_GET
    @ID_CHAT INT,
    @ID_USUARIO INT  -- For access control
AS
BEGIN
    SET NOCOUNT ON;

    SELECT *
    FROM dbo.CHATS
    WHERE ID_CHAT = @ID_CHAT
        AND ID_USUARIO = @ID_USUARIO;
END;
GO

-- =============================================
-- SP_CHAT_UPDATE: Update chat metadata
-- =============================================
CREATE PROCEDURE dbo.SP_CHAT_UPDATE
    @ID_CHAT INT,
    @ID_USUARIO INT,  -- For access control
    @TITULO NVARCHAR(200) = NULL,
    @DESCRIPCION NVARCHAR(500) = NULL,
    @ID_ESTADO INT = NULL
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE dbo.CHATS
    SET
        TITULO = COALESCE(@TITULO, TITULO),
        DESCRIPCION = COALESCE(@DESCRIPCION, DESCRIPCION),
        ID_ESTADO = COALESCE(@ID_ESTADO, ID_ESTADO),
        FECHA_MODIFICACION = GETDATE()
    WHERE ID_CHAT = @ID_CHAT
        AND ID_USUARIO = @ID_USUARIO;

    -- Return updated chat
    SELECT * FROM dbo.CHATS WHERE ID_CHAT = @ID_CHAT;
END;
GO

-- =============================================
-- SP_CHAT_DELETE: Soft delete chat
-- =============================================
CREATE PROCEDURE dbo.SP_CHAT_DELETE
    @ID_CHAT INT,
    @ID_USUARIO INT  -- For access control
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE dbo.CHATS
    SET
        ID_ESTADO = 3,  -- 3 = Deleted
        FECHA_MODIFICACION = GETDATE()
    WHERE ID_CHAT = @ID_CHAT
        AND ID_USUARIO = @ID_USUARIO;

    SELECT @@ROWCOUNT AS ROWS_AFFECTED;
END;
GO

-- =============================================
-- SP_MENSAJE_CREATE: Create message
-- =============================================
CREATE PROCEDURE dbo.SP_MENSAJE_CREATE
    @ID_CHAT INT,
    @TIPO_MENSAJE NVARCHAR(20),
    @CONTENIDO NVARCHAR(MAX),
    @CONTEXTO_USADO NVARCHAR(MAX) = NULL,
    @FUENTES NVARCHAR(MAX) = NULL,
    @NUM_DOCUMENTOS_RECUPERADOS INT = NULL,
    @MODELO_USADO NVARCHAR(100) = NULL,
    @TOKENS_ENTRADA INT = NULL,
    @TOKENS_SALIDA INT = NULL,
    @LATENCIA_MS INT = NULL,
    @TOP_K_USADO INT = NULL,
    @SIMILARITY_THRESHOLD_USADO DECIMAL(3,2) = NULL,
    @ALPHA_USADO DECIMAL(3,2) = NULL,
    @TEMPERATURE_USADO DECIMAL(3,2) = NULL,
    @MAX_TOKENS_USADO INT = NULL,
    @ERROR_MENSAJE NVARCHAR(500) = NULL,
    @ID_ESTADO INT = 1
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @NUMERO_SECUENCIA INT;

    -- Get next sequence number
    SELECT @NUMERO_SECUENCIA = COALESCE(MAX(NUMERO_SECUENCIA), 0) + 1
    FROM dbo.MENSAJES
    WHERE ID_CHAT = @ID_CHAT;

    -- Insert message
    INSERT INTO dbo.MENSAJES (
        ID_CHAT, TIPO_MENSAJE, CONTENIDO, NUMERO_SECUENCIA,
        CONTEXTO_USADO, FUENTES, NUM_DOCUMENTOS_RECUPERADOS,
        MODELO_USADO, TOKENS_ENTRADA, TOKENS_SALIDA, LATENCIA_MS,
        TOP_K_USADO, SIMILARITY_THRESHOLD_USADO, ALPHA_USADO,
        TEMPERATURE_USADO, MAX_TOKENS_USADO, ERROR_MENSAJE, ID_ESTADO
    )
    VALUES (
        @ID_CHAT, @TIPO_MENSAJE, @CONTENIDO, @NUMERO_SECUENCIA,
        @CONTEXTO_USADO, @FUENTES, @NUM_DOCUMENTOS_RECUPERADOS,
        @MODELO_USADO, @TOKENS_ENTRADA, @TOKENS_SALIDA, @LATENCIA_MS,
        @TOP_K_USADO, @SIMILARITY_THRESHOLD_USADO, @ALPHA_USADO,
        @TEMPERATURE_USADO, @MAX_TOKENS_USADO, @ERROR_MENSAJE, @ID_ESTADO
    );

    -- Update chat metadata with last message date
    UPDATE dbo.CHATS
    SET
        TOTAL_MENSAJES = TOTAL_MENSAJES + 1,
        ULTIMO_MENSAJE_FECHA = GETDATE(),
        FECHA_MODIFICACION = GETDATE()
    WHERE ID_CHAT = @ID_CHAT;

    -- Return the newly created message
    SELECT * FROM dbo.MENSAJES WHERE ID_MENSAJE = SCOPE_IDENTITY();
END;
GO

-- =============================================
-- SP_MENSAJE_LIST: Get chat messages (paginated)
-- =============================================
CREATE PROCEDURE dbo.SP_MENSAJE_LIST
    @ID_CHAT INT,
    @ID_USUARIO INT,  -- For access control
    @PAGE_NUMBER INT = 1,
    @PAGE_SIZE INT = 50,
    @ORDER_ASC BIT = 1  -- 1 = Ascending (oldest first), 0 = Descending
AS
BEGIN
    SET NOCOUNT ON;

    -- Verify user owns this chat
    IF NOT EXISTS (
        SELECT 1 FROM dbo.CHATS
        WHERE ID_CHAT = @ID_CHAT AND ID_USUARIO = @ID_USUARIO
    )
    BEGIN
        RAISERROR('Access denied: Chat does not belong to user', 16, 1);
        RETURN;
    END

    -- Return messages
    IF @ORDER_ASC = 1
    BEGIN
        SELECT *
        FROM dbo.MENSAJES
        WHERE ID_CHAT = @ID_CHAT
        ORDER BY NUMERO_SECUENCIA ASC, FECHA_CREACION ASC
        OFFSET (@PAGE_NUMBER - 1) * @PAGE_SIZE ROWS
        FETCH NEXT @PAGE_SIZE ROWS ONLY;
    END
    ELSE
    BEGIN
        SELECT *
        FROM dbo.MENSAJES
        WHERE ID_CHAT = @ID_CHAT
        ORDER BY NUMERO_SECUENCIA DESC, FECHA_CREACION DESC
        OFFSET (@PAGE_NUMBER - 1) * @PAGE_SIZE ROWS
        FETCH NEXT @PAGE_SIZE ROWS ONLY;
    END
END;
GO

-- =============================================
-- SP_CHAT_HISTORY_GET: Get chat history for RAG context
-- =============================================
CREATE PROCEDURE dbo.SP_CHAT_HISTORY_GET
    @ID_CHAT INT,
    @LAST_N_MESSAGES INT = 10  -- Get last N messages for context
AS
BEGIN
    SET NOCOUNT ON;

    SELECT TOP (@LAST_N_MESSAGES)
        TIPO_MENSAJE,
        CONTENIDO,
        NUMERO_SECUENCIA,
        FECHA_CREACION
    FROM dbo.MENSAJES
    WHERE ID_CHAT = @ID_CHAT
        AND ID_ESTADO = 1  -- Only successful messages
    ORDER BY NUMERO_SECUENCIA DESC;
END;
GO
```

#### 1.3 Create SQLAlchemy Models

**File: `app/models/chat_models.py`** (Update existing file)

```python
"""Chat and message models for multi-turn conversations."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, BigInteger, DECIMAL, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


# =============================================
# SQLAlchemy ORM Models
# =============================================

class Chat(Base):
    """SQLAlchemy model for CHATS table"""
    __tablename__ = "CHATS"
    __table_args__ = {'schema': 'dbo'}

    # Primary Key
    ID_CHAT = Column(Integer, primary_key=True, index=True)

    # Foreign Keys
    ID_USUARIO = Column(Integer, nullable=False, index=True)
    ID_EMPRESA = Column(Integer, nullable=False, index=True)
    ID_AREA = Column(Integer, nullable=False, index=True)

    # Chat Metadata
    TITULO = Column(String(200), nullable=False)
    DESCRIPCION = Column(String(500), nullable=True)

    # Last Message Date
    ULTIMO_MENSAJE_FECHA = Column(DateTime, nullable=True)

    # RAG Configuration
    TOP_K = Column(Integer, nullable=True)
    SIMILARITY_THRESHOLD = Column(DECIMAL(3, 2), nullable=True)
    ALPHA = Column(DECIMAL(3, 2), nullable=True)
    TEMPERATURE = Column(DECIMAL(3, 2), nullable=True)
    MAX_TOKENS = Column(Integer, nullable=True)

    # State & Tracking
    TOTAL_MENSAJES = Column(Integer, default=0)
    ID_ESTADO = Column(Integer, default=1)

    # Audit Fields
    FECHA_CREACION = Column(DateTime, default=datetime.utcnow)
    FECHA_MODIFICACION = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    USUARIO_CREACION = Column(String(100), nullable=True)

    # Relationships
    mensajes = relationship("Mensaje", back_populates="chat", cascade="all, delete-orphan")


class Mensaje(Base):
    """SQLAlchemy model for MENSAJES table"""
    __tablename__ = "MENSAJES"
    __table_args__ = {'schema': 'dbo'}

    # Primary Key
    ID_MENSAJE = Column(BigInteger, primary_key=True, index=True)

    # Foreign Key
    ID_CHAT = Column(Integer, ForeignKey('dbo.CHATS.ID_CHAT'), nullable=False, index=True)

    # Message Content
    TIPO_MENSAJE = Column(String(20), nullable=False)
    CONTENIDO = Column(Text, nullable=False)

    # RAG Context
    CONTEXTO_USADO = Column(Text, nullable=True)
    FUENTES = Column(Text, nullable=True)
    NUM_DOCUMENTOS_RECUPERADOS = Column(Integer, nullable=True)

    # LLM Metadata
    MODELO_USADO = Column(String(100), nullable=True)
    TOKENS_ENTRADA = Column(Integer, nullable=True)
    TOKENS_SALIDA = Column(Integer, nullable=True)
    LATENCIA_MS = Column(Integer, nullable=True)

    # RAG Parameters Used
    TOP_K_USADO = Column(Integer, nullable=True)
    SIMILARITY_THRESHOLD_USADO = Column(DECIMAL(3, 2), nullable=True)
    ALPHA_USADO = Column(DECIMAL(3, 2), nullable=True)
    TEMPERATURE_USADO = Column(DECIMAL(3, 2), nullable=True)
    MAX_TOKENS_USADO = Column(Integer, nullable=True)

    # Error Tracking
    ERROR_MENSAJE = Column(String(500), nullable=True)
    ID_ESTADO = Column(Integer, default=1)

    # Conversation Context
    NUMERO_SECUENCIA = Column(Integer, nullable=False)
    INCLUIDO_EN_CONTEXTO = Column(Boolean, default=True)

    # Audit Fields
    FECHA_CREACION = Column(DateTime, default=datetime.utcnow)
    FECHA_MODIFICACION = Column(DateTime, nullable=True)

    # Relationships
    chat = relationship("Chat", back_populates="mensajes")


# =============================================
# Pydantic Request/Response Models
# =============================================

class ChatCreateRequest(BaseModel):
    """Request to create a new chat"""
    titulo: Optional[str] = None  # If None, auto-generate from first message
    descripcion: Optional[str] = None
    company_id: str
    area: str
    id_ia_area: int
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    alpha: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatUpdateRequest(BaseModel):
    """Request to update chat metadata"""
    titulo: Optional[str] = None
    descripcion: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat details"""
    id_chat: int
    id_usuario: int
    id_empresa: int
    id_area: int
    titulo: str
    descripcion: Optional[str]
    ultimo_mensaje_fecha: Optional[datetime]
    total_mensajes: int
    fecha_creacion: datetime
    id_estado: int

    # RAG configuration
    top_k: Optional[int]
    similarity_threshold: Optional[float]
    alpha: Optional[float]
    temperature: Optional[float]
    max_tokens: Optional[int]

    class Config:
        from_attributes = True


class ChatListItem(BaseModel):
    """Lightweight chat item for list view"""
    id_chat: int
    id_usuario: int
    id_empresa: int
    id_area: int
    titulo: str
    descripcion: Optional[str]
    ultimo_mensaje_fecha: Optional[datetime]
    total_mensajes: int
    fecha_creacion: datetime

    class Config:
        from_attributes = True


class ChatListResponse(BaseModel):
    """Response for chat list endpoint"""
    chats: List[ChatListItem]
    total_count: int
    page: int
    page_size: int


class MessageResponse(BaseModel):
    """Response model for message details"""
    id_mensaje: int
    id_chat: int
    tipo_mensaje: str
    contenido: str
    numero_secuencia: int
    fecha_creacion: datetime

    # Optional metadata (for assistant messages)
    fuentes: Optional[List[Dict[str, Any]]] = None
    num_documentos_recuperados: Optional[int] = None
    modelo_usado: Optional[str] = None
    tokens_entrada: Optional[int] = None
    tokens_salida: Optional[int] = None
    latencia_ms: Optional[int] = None
    error_mensaje: Optional[str] = None
    id_estado: int = 1

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """Response for message list endpoint"""
    messages: List[MessageResponse]
    total_count: int
    page: int
    page_size: int


class UnifiedRequest(BaseModel):
    """Request Schema for streaming RAG endpoints with chat support"""
    user_id: str
    message: str

    # Chat identification
    chat_id: Optional[int] = None  # If None, creates new chat

    # Required for new chats or chat context
    id_empresa: int                         # Company ID for database storage
    empresa: str                            # Company name for vector search filtering
    id_area: int                            # Area ID for database storage
    area: str                               # Area name for vector search filtering

    # Optional RAG parameters
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    alpha: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # Optional chat metadata (for new chats)
    chat_titulo: Optional[str] = None


class AgentStreamingRequest(BaseModel):
    """Request Schema for agent streaming endpoint with external token"""
    user_id: str
    message: str

    # Chat identification
    chat_id: Optional[int] = None

    # Required
    id_empresa: int                         # Company ID for database storage
    empresa: str                            # Company name for vector search filtering
    id_area: int                            # Area ID for database storage
    area: str                               # Area name for vector search filtering
    external_token: str

    # Optional RAG parameters
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    alpha: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

---

### Phase 2: Repository Layer

#### 2.1 Create Chat Repository

**File: `app/infrastructure/repositories/chat_repository.py`**

```python
"""Repository for chat and message database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text, desc, func
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging

from app.models.chat_models import Chat, Mensaje, ChatCreateRequest, MessageResponse

logger = logging.getLogger(__name__)


class ChatRepository:
    """Repository for CHATS and MENSAJES tables"""

    def __init__(self, db: Session):
        self.db = db

    # =============================================
    # Chat Operations
    # =============================================

    async def create_chat(
        self,
        user_id: int,
        id_ia_area: int,
        titulo: str,
        company_id: str,
        area: str,
        descripcion: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        usuario_creacion: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new chat using stored procedure"""
        try:
            query = text("""
                EXEC dbo.SP_CHAT_CREATE
                    @ID_USUARIO = :user_id,
                    @ID_IA_AREA = :id_ia_area,
                    @TITULO = :titulo,
                    @COMPANY_ID = :company_id,
                    @AREA = :area,
                    @DESCRIPCION = :descripcion,
                    @TOP_K = :top_k,
                    @SIMILARITY_THRESHOLD = :similarity_threshold,
                    @ALPHA = :alpha,
                    @TEMPERATURE = :temperature,
                    @MAX_TOKENS = :max_tokens,
                    @USUARIO_CREACION = :usuario_creacion
            """)

            result = self.db.execute(query, {
                'user_id': user_id,
                'id_ia_area': id_ia_area,
                'titulo': titulo,
                'company_id': company_id,
                'area': area,
                'descripcion': descripcion,
                'top_k': top_k,
                'similarity_threshold': similarity_threshold,
                'alpha': alpha,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'usuario_creacion': usuario_creacion
            })

            chat_data = result.fetchone()
            self.db.commit()

            if chat_data:
                return dict(chat_data._mapping)
            return None

        except Exception as e:
            logger.error(f"Error creating chat: {e}")
            self.db.rollback()
            raise

    async def get_chat_by_id(self, chat_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Get chat by ID with access control"""
        try:
            query = text("""
                EXEC dbo.SP_CHAT_GET
                    @ID_CHAT = :chat_id,
                    @ID_USUARIO = :user_id
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'user_id': user_id
            })

            chat_data = result.fetchone()

            if chat_data:
                return dict(chat_data._mapping)
            return None

        except Exception as e:
            logger.error(f"Error getting chat {chat_id}: {e}")
            return None

    async def list_user_chats(
        self,
        user_id: int,
        id_estado: int = 1,
        page: int = 1,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """List user's chats with pagination"""
        try:
            query = text("""
                EXEC dbo.SP_CHAT_LIST
                    @ID_USUARIO = :user_id,
                    @ID_ESTADO = :id_estado,
                    @PAGE_NUMBER = :page,
                    @PAGE_SIZE = :page_size
            """)

            result = self.db.execute(query, {
                'user_id': user_id,
                'id_estado': id_estado,
                'page': page,
                'page_size': page_size
            })

            chats = []
            for row in result.fetchall():
                chats.append(dict(row._mapping))

            return chats

        except Exception as e:
            logger.error(f"Error listing chats for user {user_id}: {e}")
            return []

    async def update_chat(
        self,
        chat_id: int,
        user_id: int,
        titulo: Optional[str] = None,
        descripcion: Optional[str] = None,
        id_estado: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Update chat metadata"""
        try:
            query = text("""
                EXEC dbo.SP_CHAT_UPDATE
                    @ID_CHAT = :chat_id,
                    @ID_USUARIO = :user_id,
                    @TITULO = :titulo,
                    @DESCRIPCION = :descripcion,
                    @ID_ESTADO = :id_estado
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'user_id': user_id,
                'titulo': titulo,
                'descripcion': descripcion,
                'id_estado': id_estado
            })

            chat_data = result.fetchone()
            self.db.commit()

            if chat_data:
                return dict(chat_data._mapping)
            return None

        except Exception as e:
            logger.error(f"Error updating chat {chat_id}: {e}")
            self.db.rollback()
            raise

    async def delete_chat(self, chat_id: int, user_id: int) -> bool:
        """Soft delete chat"""
        try:
            query = text("""
                EXEC dbo.SP_CHAT_DELETE
                    @ID_CHAT = :chat_id,
                    @ID_USUARIO = :user_id
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'user_id': user_id
            })

            affected = result.fetchone()
            self.db.commit()

            return affected and affected[0] > 0

        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
            self.db.rollback()
            return False

    # =============================================
    # Message Operations
    # =============================================

    async def create_message(
        self,
        chat_id: int,
        tipo_mensaje: str,
        contenido: str,
        contexto_usado: Optional[List[Dict[str, Any]]] = None,
        fuentes: Optional[List[Dict[str, Any]]] = None,
        num_documentos_recuperados: Optional[int] = None,
        modelo_usado: Optional[str] = None,
        tokens_entrada: Optional[int] = None,
        tokens_salida: Optional[int] = None,
        latencia_ms: Optional[int] = None,
        top_k_usado: Optional[int] = None,
        similarity_threshold_usado: Optional[float] = None,
        alpha_usado: Optional[float] = None,
        temperature_usado: Optional[float] = None,
        max_tokens_usado: Optional[int] = None,
        error_mensaje: Optional[str] = None,
        id_estado: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Create a new message using stored procedure"""
        try:
            # Convert lists to JSON strings
            contexto_json = json.dumps(contexto_usado) if contexto_usado else None
            fuentes_json = json.dumps(fuentes) if fuentes else None

            query = text("""
                EXEC dbo.SP_MENSAJE_CREATE
                    @ID_CHAT = :chat_id,
                    @TIPO_MENSAJE = :tipo_mensaje,
                    @CONTENIDO = :contenido,
                    @CONTEXTO_USADO = :contexto_usado,
                    @FUENTES = :fuentes,
                    @NUM_DOCUMENTOS_RECUPERADOS = :num_docs,
                    @MODELO_USADO = :modelo,
                    @TOKENS_ENTRADA = :tokens_in,
                    @TOKENS_SALIDA = :tokens_out,
                    @LATENCIA_MS = :latencia,
                    @TOP_K_USADO = :top_k,
                    @SIMILARITY_THRESHOLD_USADO = :threshold,
                    @ALPHA_USADO = :alpha,
                    @TEMPERATURE_USADO = :temp,
                    @MAX_TOKENS_USADO = :max_tokens,
                    @ERROR_MENSAJE = :error,
                    @ID_ESTADO = :estado
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'tipo_mensaje': tipo_mensaje,
                'contenido': contenido,
                'contexto_usado': contexto_json,
                'fuentes': fuentes_json,
                'num_docs': num_documentos_recuperados,
                'modelo': modelo_usado,
                'tokens_in': tokens_entrada,
                'tokens_out': tokens_salida,
                'latencia': latencia_ms,
                'top_k': top_k_usado,
                'threshold': similarity_threshold_usado,
                'alpha': alpha_usado,
                'temp': temperature_usado,
                'max_tokens': max_tokens_usado,
                'error': error_mensaje,
                'estado': id_estado
            })

            message_data = result.fetchone()
            self.db.commit()

            if message_data:
                return dict(message_data._mapping)
            return None

        except Exception as e:
            logger.error(f"Error creating message: {e}")
            self.db.rollback()
            raise

    async def get_chat_messages(
        self,
        chat_id: int,
        user_id: int,
        page: int = 1,
        page_size: int = 50,
        order_asc: bool = True
    ) -> List[Dict[str, Any]]:
        """Get chat messages with pagination"""
        try:
            query = text("""
                EXEC dbo.SP_MENSAJE_LIST
                    @ID_CHAT = :chat_id,
                    @ID_USUARIO = :user_id,
                    @PAGE_NUMBER = :page,
                    @PAGE_SIZE = :page_size,
                    @ORDER_ASC = :order_asc
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'user_id': user_id,
                'page': page,
                'page_size': page_size,
                'order_asc': 1 if order_asc else 0
            })

            messages = []
            for row in result.fetchall():
                msg_dict = dict(row._mapping)

                # Parse JSON fields
                if msg_dict.get('FUENTES'):
                    try:
                        msg_dict['FUENTES'] = json.loads(msg_dict['FUENTES'])
                    except:
                        pass

                if msg_dict.get('CONTEXTO_USADO'):
                    try:
                        msg_dict['CONTEXTO_USADO'] = json.loads(msg_dict['CONTEXTO_USADO'])
                    except:
                        pass

                messages.append(msg_dict)

            return messages

        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}: {e}")
            return []

    async def get_chat_history_for_context(
        self,
        chat_id: int,
        last_n_messages: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent chat history for RAG context building"""
        try:
            query = text("""
                EXEC dbo.SP_CHAT_HISTORY_GET
                    @ID_CHAT = :chat_id,
                    @LAST_N_MESSAGES = :last_n
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id,
                'last_n': last_n_messages
            })

            history = []
            for row in result.fetchall():
                row_dict = dict(row._mapping)
                history.append({
                    'tipo_mensaje': row_dict['TIPO_MENSAJE'],
                    'contenido': row_dict['CONTENIDO']
                })

            # Reverse to get chronological order (oldest to newest)
            return list(reversed(history))

        except Exception as e:
            logger.error(f"Error getting chat history for chat {chat_id}: {e}")
            return []

    async def get_chat_count(self, user_id: int, id_estado: int = 1) -> int:
        """Get total count of user's chats"""
        try:
            query = text("""
                SELECT COUNT(*) as total
                FROM dbo.CHATS
                WHERE ID_USUARIO = :user_id
                    AND ID_ESTADO = :id_estado
            """)

            result = self.db.execute(query, {
                'user_id': user_id,
                'id_estado': id_estado
            })

            count_data = result.fetchone()
            return count_data[0] if count_data else 0

        except Exception as e:
            logger.error(f"Error getting chat count: {e}")
            return 0

    async def get_message_count(self, chat_id: int) -> int:
        """Get total count of messages in chat"""
        try:
            query = text("""
                SELECT COUNT(*) as total
                FROM dbo.MENSAJES
                WHERE ID_CHAT = :chat_id
            """)

            result = self.db.execute(query, {
                'chat_id': chat_id
            })

            count_data = result.fetchone()
            return count_data[0] if count_data else 0

        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0
```

---

### Phase 3: Service Layer

#### 3.1 Create Chat Management Service

**File: `app/services/chat_management_service.py`**

```python
"""Service for managing chat sessions and messages."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import logging

from app.infrastructure.repositories.chat_repository import ChatRepository
from app.models.chat_models import (
    ChatCreateRequest,
    ChatUpdateRequest,
    ChatResponse,
    ChatListResponse,
    ChatListItem,
    MessageResponse,
    MessageListResponse
)

logger = logging.getLogger(__name__)


class ChatManagementService:
    """Service for chat and message management operations"""

    def __init__(self, db: Session):
        self.db = db
        self.repository = ChatRepository(db)

    async def create_chat(
        self,
        user_id: int,
        request: ChatCreateRequest,
        auto_title: bool = True
    ) -> Optional[ChatResponse]:
        """Create a new chat session"""
        try:
            # Auto-generate title if not provided
            titulo = request.titulo or "New Chat"
            if auto_title and not request.titulo:
                titulo = await self._generate_auto_title()

            chat_data = await self.repository.create_chat(
                user_id=user_id,
                id_ia_area=request.id_ia_area,
                titulo=titulo,
                company_id=request.company_id,
                area=request.area,
                descripcion=request.descripcion,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                alpha=request.alpha,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            if chat_data:
                return self._map_to_chat_response(chat_data)
            return None

        except Exception as e:
            logger.error(f"Error in create_chat service: {e}")
            raise

    async def get_chat(self, chat_id: int, user_id: int) -> Optional[ChatResponse]:
        """Get chat details by ID"""
        try:
            chat_data = await self.repository.get_chat_by_id(chat_id, user_id)
            if chat_data:
                return self._map_to_chat_response(chat_data)
            return None
        except Exception as e:
            logger.error(f"Error in get_chat service: {e}")
            return None

    async def list_chats(
        self,
        user_id: int,
        page: int = 1,
        page_size: int = 50
    ) -> ChatListResponse:
        """List user's active chats"""
        try:
            chats_data = await self.repository.list_user_chats(
                user_id=user_id,
                id_estado=1,
                page=page,
                page_size=page_size
            )

            total_count = await self.repository.get_chat_count(user_id, id_estado=1)

            chat_items = [self._map_to_chat_list_item(chat) for chat in chats_data]

            return ChatListResponse(
                chats=chat_items,
                total_count=total_count,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"Error in list_chats service: {e}")
            return ChatListResponse(chats=[], total_count=0, page=page, page_size=page_size)

    async def update_chat(
        self,
        chat_id: int,
        user_id: int,
        request: ChatUpdateRequest
    ) -> Optional[ChatResponse]:
        """Update chat metadata"""
        try:
            chat_data = await self.repository.update_chat(
                chat_id=chat_id,
                user_id=user_id,
                titulo=request.titulo,
                descripcion=request.descripcion
            )

            if chat_data:
                return self._map_to_chat_response(chat_data)
            return None

        except Exception as e:
            logger.error(f"Error in update_chat service: {e}")
            raise

    async def delete_chat(self, chat_id: int, user_id: int) -> bool:
        """Soft delete a chat"""
        try:
            return await self.repository.delete_chat(chat_id, user_id)
        except Exception as e:
            logger.error(f"Error in delete_chat service: {e}")
            return False

    async def get_messages(
        self,
        chat_id: int,
        user_id: int,
        page: int = 1,
        page_size: int = 50
    ) -> MessageListResponse:
        """Get chat messages with pagination"""
        try:
            messages_data = await self.repository.get_chat_messages(
                chat_id=chat_id,
                user_id=user_id,
                page=page,
                page_size=page_size,
                order_asc=True
            )

            total_count = await self.repository.get_message_count(chat_id)

            message_items = [self._map_to_message_response(msg) for msg in messages_data]

            return MessageListResponse(
                messages=message_items,
                total_count=total_count,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"Error in get_messages service: {e}")
            return MessageListResponse(messages=[], total_count=0, page=page, page_size=page_size)

    async def generate_chat_title_from_message(self, message: str) -> str:
        """Generate a concise chat title from first message"""
        try:
            # Simple implementation: truncate first message
            # TODO: Use LLM to generate better titles
            max_length = 50
            if len(message) <= max_length:
                return message
            return message[:max_length].rsplit(' ', 1)[0] + "..."

        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "New Chat"

    # =============================================
    # Helper Methods
    # =============================================

    async def _generate_auto_title(self) -> str:
        """Generate default title for new chat"""
        from datetime import datetime
        return f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def _map_to_chat_response(self, chat_data: Dict[str, Any]) -> ChatResponse:
        """Map database result to ChatResponse"""
        return ChatResponse(
            id_chat=chat_data['ID_CHAT'],
            titulo=chat_data['TITULO'],
            descripcion=chat_data.get('DESCRIPCION'),
            company_id=chat_data['COMPANY_ID'],
            area=chat_data['AREA'],
            id_ia_area=chat_data['ID_IA_AREA'],
            total_mensajes=chat_data['TOTAL_MENSAJES'],
            ultimo_mensaje_at=chat_data.get('ULTIMO_MENSAJE_AT'),
            fecha_creacion=chat_data['FECHA_CREACION'],
            id_estado=chat_data['ID_ESTADO'],
            top_k=chat_data.get('TOP_K'),
            similarity_threshold=float(chat_data['SIMILARITY_THRESHOLD']) if chat_data.get('SIMILARITY_THRESHOLD') else None,
            alpha=float(chat_data['ALPHA']) if chat_data.get('ALPHA') else None,
            temperature=float(chat_data['TEMPERATURE']) if chat_data.get('TEMPERATURE') else None,
            max_tokens=chat_data.get('MAX_TOKENS')
        )

    def _map_to_chat_list_item(self, chat_data: Dict[str, Any]) -> ChatListItem:
        """Map database result to ChatListItem"""
        return ChatListItem(
            id_chat=chat_data['ID_CHAT'],
            titulo=chat_data['TITULO'],
            descripcion=chat_data.get('DESCRIPCION'),
            total_mensajes=chat_data['TOTAL_MENSAJES'],
            ultimo_mensaje_at=chat_data.get('ULTIMO_MENSAJE_AT'),
            fecha_creacion=chat_data['FECHA_CREACION'],
            company_id=chat_data['COMPANY_ID'],
            area=chat_data['AREA']
        )

    def _map_to_message_response(self, msg_data: Dict[str, Any]) -> MessageResponse:
        """Map database result to MessageResponse"""
        return MessageResponse(
            id_mensaje=msg_data['ID_MENSAJE'],
            id_chat=msg_data['ID_CHAT'],
            tipo_mensaje=msg_data['TIPO_MENSAJE'],
            contenido=msg_data['CONTENIDO'],
            numero_secuencia=msg_data['NUMERO_SECUENCIA'],
            fecha_creacion=msg_data['FECHA_CREACION'],
            fuentes=msg_data.get('FUENTES'),
            num_documentos_recuperados=msg_data.get('NUM_DOCUMENTOS_RECUPERADOS'),
            modelo_usado=msg_data.get('MODELO_USADO'),
            tokens_entrada=msg_data.get('TOKENS_ENTRADA'),
            tokens_salida=msg_data.get('TOKENS_SALIDA'),
            latencia_ms=msg_data.get('LATENCIA_MS'),
            error_mensaje=msg_data.get('ERROR_MENSAJE'),
            id_estado=msg_data['ID_ESTADO']
        )
```

#### 3.2 Update Chat Service for RAG

**File: `app/services/chat_service.py`** (Add these methods to existing service)

```python
# Add to existing ChatService class:

from app.infrastructure.repositories.chat_repository import ChatRepository

class ChatService:
    # ... existing code ...

    def __init__(self, llm_provider, embeddings_provider, vectorstore_repository, db: Session):
        self.llm_provider = llm_provider
        self.embeddings_provider = embeddings_provider
        self.vectorstore_repository = vectorstore_repository
        self.db = db
        self.chat_repository = ChatRepository(db)  # Add this

    async def build_conversation_context(self, chat_id: Optional[int], max_messages: int = 10) -> str:
        """Build conversation context from chat history"""
        if not chat_id:
            return ""

        try:
            history = await self.chat_repository.get_chat_history_for_context(
                chat_id=chat_id,
                last_n_messages=max_messages
            )

            if not history:
                return ""

            # Format: [User]: message\n[Assistant]: response\n...
            context_lines = []
            for msg in history:
                role = "User" if msg['tipo_mensaje'] == 'user' else "Assistant"
                context_lines.append(f"[{role}]: {msg['contenido']}")

            return "\n".join(context_lines)

        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return ""

    async def save_user_message(
        self,
        chat_id: int,
        message: str
    ) -> Optional[Dict[str, Any]]:
        """Save user message to database"""
        try:
            return await self.chat_repository.create_message(
                chat_id=chat_id,
                tipo_mensaje='user',
                contenido=message
            )
        except Exception as e:
            logger.error(f"Error saving user message: {e}")
            return None

    async def save_assistant_message(
        self,
        chat_id: int,
        contenido: str,
        fuentes: Optional[List[Dict[str, Any]]] = None,
        contexto_usado: Optional[List[Dict[str, Any]]] = None,
        num_documentos: Optional[int] = None,
        modelo: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        latencia: Optional[int] = None,
        rag_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Save assistant message to database"""
        try:
            return await self.chat_repository.create_message(
                chat_id=chat_id,
                tipo_mensaje='assistant',
                contenido=contenido,
                fuentes=fuentes,
                contexto_usado=contexto_usado,
                num_documentos_recuperados=num_documentos,
                modelo_usado=modelo,
                tokens_entrada=tokens_in,
                tokens_salida=tokens_out,
                latencia_ms=latencia,
                top_k_usado=rag_params.get('top_k') if rag_params else None,
                similarity_threshold_usado=rag_params.get('similarity_threshold') if rag_params else None,
                alpha_usado=rag_params.get('alpha') if rag_params else None,
                temperature_usado=rag_params.get('temperature') if rag_params else None,
                max_tokens_usado=rag_params.get('max_tokens') if rag_params else None
            )
        except Exception as e:
            logger.error(f"Error saving assistant message: {e}")
            return None
```

---

### Phase 4: API Layer

#### 4.1 Create Chat Management Endpoints

**File: `app/api/chat_management.py`** (New file)

```python
"""API endpoints for chat and message management."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db
from app.services.chat_management_service import ChatManagementService
from app.models.chat_models import (
    ChatCreateRequest,
    ChatUpdateRequest,
    ChatResponse,
    ChatListResponse,
    MessageListResponse
)
from app.utils.jwt_auth import verify_jwt_token

router = APIRouter(prefix="/api/v1/chats", tags=["Chat Management"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    request: ChatCreateRequest,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    Create a new chat session

    - **titulo**: Optional chat title (auto-generated if not provided)
    - **descripcion**: Optional description
    - **company_id**: Company identifier for filtering
    - **area**: Area for filtering
    - **id_ia_area**: Area ID from actual_company_area
    - **RAG params**: Optional default parameters for this chat
    """
    user_id = user_data.get('ID_USUARIO')

    service = ChatManagementService(db)
    chat = await service.create_chat(user_id=user_id, request=request)

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat"
        )

    return chat


@router.get("", response_model=ChatListResponse)
async def list_chats(
    page: int = 1,
    page_size: int = 50,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    List user's active chats

    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 50, max: 100)
    """
    if page_size > 100:
        page_size = 100

    user_id = user_data.get('ID_USUARIO')

    service = ChatManagementService(db)
    return await service.list_chats(user_id=user_id, page=page, page_size=page_size)


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: int,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    Get chat details by ID

    Returns full chat information including RAG configuration
    """
    user_id = user_data.get('ID_USUARIO')

    service = ChatManagementService(db)
    chat = await service.get_chat(chat_id=chat_id, user_id=user_id)

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found or access denied"
        )

    return chat


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: int,
    request: ChatUpdateRequest,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    Update chat title and/or description

    - **titulo**: New chat title
    - **descripcion**: New description
    """
    user_id = user_data.get('ID_USUARIO')

    service = ChatManagementService(db)
    chat = await service.update_chat(chat_id=chat_id, user_id=user_id, request=request)

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found or access denied"
        )

    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: int,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    Delete a chat (soft delete)

    Marks chat as deleted (ID_ESTADO = 3)
    """
    user_id = user_data.get('ID_USUARIO')

    service = ChatManagementService(db)
    success = await service.delete_chat(chat_id=chat_id, user_id=user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found or access denied"
        )


@router.get("/{chat_id}/messages", response_model=MessageListResponse)
async def get_chat_messages(
    chat_id: int,
    page: int = 1,
    page_size: int = 50,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    Get messages from a chat

    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 50, max: 100)

    Returns messages in chronological order (oldest to newest)
    """
    if page_size > 100:
        page_size = 100

    user_id = user_data.get('ID_USUARIO')

    service = ChatManagementService(db)
    return await service.get_messages(
        chat_id=chat_id,
        user_id=user_id,
        page=page,
        page_size=page_size
    )
```

#### 4.2 Update Main Application

**File: `main.py`** (Add chat management router)

```python
from app.api import chat_management

# Add to existing routers
app.include_router(chat_management.router)
```

---

### Phase 5: Integration & Features

#### 5.1 Update RAG Streaming Endpoint

**File: `app/api/chat.py`** (Modify existing streaming endpoint)

```python
from app.services.chat_management_service import ChatManagementService
from app.infrastructure.repositories.chat_repository import ChatRepository

@router.post("/chat-streaming")
async def chat_streaming(
    request: UnifiedRequest,
    user_data: dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
):
    """
    Streaming RAG endpoint with chat persistence

    - If chat_id is provided: Continue existing conversation
    - If chat_id is None: Create new chat
    """

    async def generate_response():
        start_time = time.time()
        full_response = ""
        chat_id = request.chat_id

        try:
            # Step 1: Create or get chat
            if not chat_id:
                chat_mgmt_service = ChatManagementService(db)

                # Generate title from message
                titulo = request.chat_titulo or await chat_mgmt_service.generate_chat_title_from_message(request.message)

                # Create new chat
                from app.models.chat_models import ChatCreateRequest
                chat_request = ChatCreateRequest(
                    titulo=titulo,
                    company_id=request.company_id,
                    area=request.area,
                    id_ia_area=request.id_ia_area,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    alpha=request.alpha,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                new_chat = await chat_mgmt_service.create_chat(
                    user_id=user_data['ID_USUARIO'],
                    request=chat_request,
                    auto_title=False
                )

                if not new_chat:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Failed to create chat'})}\n\n"
                    return

                chat_id = new_chat.id_chat

                # Send chat ID to client
                yield f"data: {json.dumps({'type': 'chat_created', 'chat_id': chat_id})}\n\n"

            # Step 2: Save user message
            chat_repo = ChatRepository(db)
            await chat_repo.create_message(
                chat_id=chat_id,
                tipo_mensaje='user',
                contenido=request.message
            )

            # Step 3: Build conversation context
            conversation_context = await chat_service.build_conversation_context(
                chat_id=chat_id,
                max_messages=10
            )

            # Step 4: Perform RAG (existing logic)
            # ... existing RAG code ...

            # Step 5: Stream response (existing logic)
            async for chunk in llm_provider.stream_generate(...):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            # Step 6: Save assistant message
            end_time = time.time()
            latencia_ms = int((end_time - start_time) * 1000)

            await chat_repo.create_message(
                chat_id=chat_id,
                tipo_mensaje='assistant',
                contenido=full_response,
                fuentes=sources_metadata,
                contexto_usado=retrieved_docs,
                num_documentos_recuperados=len(retrieved_docs),
                modelo_usado=settings.llm_model_id,
                tokens_entrada=input_tokens,
                tokens_salida=output_tokens,
                latencia_ms=latencia_ms,
                top_k_usado=actual_top_k,
                similarity_threshold_usado=actual_threshold,
                alpha_usado=actual_alpha,
                temperature_usado=actual_temp,
                max_tokens_usado=actual_max_tokens
            )

            yield f"data: {json.dumps({'type': 'complete', 'status': 'success'})}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
```

---

## API Endpoints

### Chat Management Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/chats` | Create new chat | JWT |
| `GET` | `/api/v1/chats` | List user's chats (paginated) | JWT |
| `GET` | `/api/v1/chats/{id}` | Get chat details | JWT |
| `PATCH` | `/api/v1/chats/{id}` | Update chat title/description | JWT |
| `DELETE` | `/api/v1/chats/{id}` | Soft delete chat | JWT |
| `GET` | `/api/v1/chats/{id}/messages` | Get chat messages (paginated) | JWT |

### Updated RAG Endpoint

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/rag/chat-streaming` | Streaming RAG with chat persistence | JWT |

---

## Request/Response Flow

### Flow 1: New Chat Creation

```
1. User sends first message (chat_id = null)
   POST /api/v1/rag/chat-streaming
   {
     "user_id": "123",
     "message": "What is quantum computing?",
     "chat_id": null,  // Creates new chat
     "company_id": "company_001",
     "area": "technology",
     "id_ia_area": 5
   }

2. Backend creates new chat with auto-generated title
3. Backend saves user message to MENSAJES table
4. RAG processes request (no conversation context)
5. Backend streams response
6. Backend saves assistant response with metadata
7. Response includes chat_id for subsequent messages
```

### Flow 2: Continuing Existing Chat

```
1. User sends message to existing chat (chat_id = 456)
   POST /api/v1/rag/chat-streaming
   {
     "user_id": "123",
     "message": "Can you explain quantum entanglement?",
     "chat_id": 456,  // Continues existing chat
     "company_id": "company_001",
     "area": "technology",
     "id_ia_area": 5
   }

2. Backend validates user owns chat 456
3. Backend loads last 10 messages as conversation context
4. Backend saves new user message
5. RAG processes with conversation context included
6. Backend streams response
7. Backend saves assistant response
8. CHATS.ULTIMO_MENSAJE_AT updated automatically
```

### Flow 3: Chat Management

```
1. List all chats
   GET /api/v1/chats?page=1&page_size=20

2. Get specific chat
   GET /api/v1/chats/456

3. Update chat title
   PATCH /api/v1/chats/456
   {
     "titulo": "Quantum Physics Discussion",
     "descripcion": "Deep dive into quantum mechanics"
   }

4. Get chat messages
   GET /api/v1/chats/456/messages?page=1&page_size=50

5. Delete chat
   DELETE /api/v1/chats/456
```

---

## Key Design Decisions

### 1. **Chat-First Architecture**
- Every conversation is a chat session
- Messages always belong to a chat
- Orphaned messages are prevented by foreign key constraints

### 2. **Soft Deletes**
- `ID_ESTADO` field enables soft deletion
- States: 1=Active, 2=Archived, 3=Deleted
- Preserves data for audit/recovery purposes

### 3. **Comprehensive Audit Trail**
- Creation timestamps on all records
- Modification tracking on chats
- User creation field for compliance

### 4. **RAG Configuration Persistence**
- Per-chat default parameters stored in CHATS table
- Actual parameters used stored per-message in MENSAJES
- Enables reproducibility and debugging

### 5. **JSON Storage for Flexibility**
- `CONTEXTO_USADO`: Full retrieved documents
- `FUENTES`: Source metadata (filenames, pages, scores)
- NVARCHAR(MAX) allows flexible schema evolution

### 6. **Cascading Deletes**
- Messages cascade when chat is hard deleted
- Stored procedure handles soft deletes (more common)

### 7. **Performance Optimization**
- Strategic indexes on foreign keys
- Date indexes for sorting by recency
- Pagination support in all list endpoints

### 8. **Token & Cost Tracking**
- Input/output tokens tracked per message
- Latency measured for SLA monitoring
- Model name stored for cost attribution

### 9. **Conversation Context Management**
- `SP_CHAT_HISTORY_GET` retrieves last N messages
- Configurable context window (default: 10 messages)
- Future: Token-based context limiting

### 10. **Security & Access Control**
- All stored procedures validate user ownership
- Chat access verified before message retrieval
- JWT token required for all operations

---

## Future Enhancements

### Phase 6: Advanced Features

1. **LLM-Generated Chat Titles**
   - Use small LLM to generate concise titles from first message
   - Fallback to truncated message if LLM unavailable

2. **Conversation Branching**
   - Allow users to branch conversations from any message
   - Create new chat with history up to branch point

3. **Chat Sharing**
   - Add `COMPARTIDO` flag and sharing URLs
   - Public/private/link-only visibility options

4. **Export Functionality**
   - Export chat as PDF, MD, or JSON
   - Include sources and metadata

5. **Search Across Chats**
   - Full-text search on message content
   - Filter by date, area, company

6. **Analytics Dashboard**
   - Token usage by user/chat/timeframe
   - Popular topics extraction
   - Response latency metrics

7. **Message Reactions**
   - Thumbs up/down on assistant responses
   - Feedback collection for model improvement

8. **Token-Based Context Window**
   - Limit context by token count instead of message count
   - Dynamic context sizing based on model limits

9. **Multi-Model Support**
   - Store model ID per message
   - Allow switching models mid-conversation

10. **Rate Limiting**
    - Per-user message limits
    - Quota management in database

---

## Migration Checklist

- [ ] Run `001_create_chats_table.sql`
- [ ] Run `002_create_mensajes_table.sql`
- [ ] Run `003_create_stored_procedures.sql`
- [ ] Update `app/models/chat_models.py`
- [ ] Create `app/infrastructure/repositories/chat_repository.py`
- [ ] Create `app/services/chat_management_service.py`
- [ ] Update `app/services/chat_service.py`
- [ ] Create `app/api/chat_management.py`
- [ ] Update `app/api/chat.py`
- [ ] Update `main.py` to include new router
- [ ] Update `app/core/database.py` to create tables on startup (optional)
- [ ] Test all endpoints with Postman/curl
- [ ] Update API documentation
- [ ] Deploy to staging environment
- [ ] Perform integration testing
- [ ] Deploy to production

---

## Testing Recommendations

### Unit Tests
- Test stored procedures with various inputs
- Test repository methods in isolation
- Test service layer business logic

### Integration Tests
- Test full RAG flow with chat persistence
- Test concurrent chat creation
- Test pagination edge cases

### Performance Tests
- Load test with 1000+ messages per chat
- Test query performance with indexes
- Measure streaming latency impact

### Security Tests
- Verify user cannot access other users' chats
- Test SQL injection prevention in stored procedures
- Validate JWT token expiration handling

---

## Conclusion

This implementation plan provides a robust, scalable foundation for multi-turn conversational RAG with full persistence, audit trails, and extensibility. The hexagonal architecture is maintained throughout, ensuring clean separation of concerns and testability.

**Key Benefits:**
- ✅ Full conversation history
- ✅ Multi-tenant support
- ✅ Cost tracking & analytics
- ✅ Audit compliance
- ✅ Scalable architecture
- ✅ Future-proof design

Ready for implementation!
