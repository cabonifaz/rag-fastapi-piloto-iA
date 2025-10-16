# Qamaq RAG (Retrieval-Augmented Generation with FastAPI)

Qamaq RAG is a **FastAPI** project implementing **Hexagonal Architecture (Ports & Adapters)** for **Retrieval-Augmented Generation (RAG)** using AWS Bedrock, Weaviate Cloud Service, DynamoDB for message storage, and SQL Server for authentication and chat management.

The system provides streaming RAG responses with JWT authentication, multi-tenant company/area-specific document filtering, and persistent chat sessions with message history.

---

## 🚀 Features

- **Hexagonal Architecture** (Domain Ports, Infrastructure Adapters, Services, API layers)
- **Streaming RAG** with Server-Sent Events (SSE)
- **AWS Bedrock Integration**:
  - LLM: Claude 3.5 Sonnet, Claude 3 Haiku/Sonnet
  - Embeddings: Amazon Titan Text Embeddings V2
  - Converse API with timeout protection and async thread pool execution
- **Vector Database**: Weaviate Cloud Service (WCS) with hybrid search (vector + BM25)
- **Message Storage**: DynamoDB for chat message persistence
- **Chat Management**: SQL Server with stored procedures for chat CRUD operations
- **Authentication**: JWT-based with SQL Server backend and user profile management
- **Multi-tenant Architecture**:
  - Company-scoped collections (EMPR{id})
  - Area-based filtering (AREA{id})
  - Default/shared documents across all areas
- **Agent Orchestrator**: Intelligent query analysis with API integration capabilities
- **Task Decomposition**: Dynamic workflow generation based on user queries
- **Dependency Injection**: Custom DI container for clean architecture
- **Real-time streaming**: Live RAG responses via SSE with timeout protection
- **Pydantic v2**: Modern validation with field validators and ConfigDict
- **Configurable**: Full environment-based configuration with extensive validation

---

## 📂 Current Project Structure

```
rag-fastapi-piloto-iA/
├── main.py                          # FastAPI entry point & middleware
├── app/
│   ├── api/                        # API layer (routes/endpoints)
│   │   ├── chat.py                 # RAG streaming endpoints (chat & agent)
│   │   ├── chats.py                # Chat management (CRUD operations)
│   │   ├── messages.py             # Message retrieval from DynamoDB
│   │   ├── auth.py                 # Authentication endpoints
│   │   └── health.py               # Health check endpoints
│   ├── core/                       # Configuration & infrastructure
│   │   ├── config.py               # Environment settings & validation (Pydantic v2)
│   │   ├── database.py             # SQL Server connection
│   │   ├── database_config.py      # Database configuration (Pydantic v2)
│   │   └── container.py            # Dependency injection container
│   ├── domain/ports/               # Hexagonal architecture interfaces
│   │   ├── llm_port.py            # LLM provider interface
│   │   ├── embeddings_port.py     # Embeddings provider interface
│   │   ├── vectorstore_port.py    # Vector database interface
│   │   └── task_decomposition_port.py # Query analysis interface
│   ├── infrastructure/            # External service adapters
│   │   ├── llm/
│   │   │   ├── aws_bedrock_converse_provider.py  # AWS Bedrock Converse API with timeouts
│   │   │   └── model_configs/     # Model-specific configurations
│   │   │       ├── base_config.py
│   │   │       ├── claude_config.py
│   │   │       └── llama_config.py
│   │   ├── embeddings/
│   │   │   └── aws_embeddings.py  # AWS Bedrock Titan embeddings
│   │   ├── vectorstores/
│   │   │   └── weaviate_repository.py # Weaviate Cloud with hybrid search
│   │   ├── task_decomposition/
│   │   │   ├── bedrock_orchestrator.py # Agent orchestrator implementation
│   │   │   └── task_generator.py       # Task generation from analysis
│   │   └── api_clients/
│   │       └── api_client.py      # External API integration (httpx)
│   ├── services/                  # Business logic services
│   │   ├── rag_service.py         # RAG orchestration with streaming
│   │   ├── chat_service.py        # Chat CRUD operations with stored procedures
│   │   ├── message_service.py     # DynamoDB message operations
│   │   └── auth_service.py        # Authentication business logic
│   ├── models/                    # Pydantic data models (v2)
│   │   ├── chat_models.py         # Chat request/response models
│   │   ├── task_models.py         # Task decomposition models
│   │   ├── response_models.py     # Standardized API responses
│   │   └── user_models.py         # User & authentication models
│   └── utils/                     # Utility functions
│       ├── jwt_auth.py           # JWT token management
│       └── password_utils.py     # Password hashing utilities
├── requirements.txt               # Python dependencies
├── .env                          # Environment configuration
└── README.md                     # This documentation
```

---

## ⚙️ Configuration

All configuration is managed through environment variables in `.env`.

### Example `.env`

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=false

# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default  # Optional

# Embeddings Configuration
EMBEDDINGS_PROVIDER=aws
EMBEDDINGS_MODEL_ID=amazon.titan-embed-text-v2:0
EMBEDDINGS_REGION=us-east-1
EMBEDDINGS_DIMENSIONS=1024

# LLM Configuration
LLM_PROVIDER=aws
LLM_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
LLM_REGION=us-east-1
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_ROLE_BEHAVIOR=You are a helpful AI assistant. Answer questions based on the provided context.

# Orchestrator Configuration (Agent Mode)
ORCHESTRATOR_PROVIDER=aws
ORCHESTRATOR_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
ORCHESTRATOR_REGION=us-east-1
ORCHESTRATOR_MAX_TOKENS=2048
ORCHESTRATOR_TEMPERATURE=0.3

# Weaviate Configuration
WEAVIATE_URL=https://your-cluster.weaviate.cloud
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_CLASS_NAME=Documents
WEAVIATE_GRPC=your-cluster.weaviate.cloud:443

# RAG Configuration
RAG_TOP_K_RESULTS=5
RAG_SIMILARITY_THRESHOLD=0.7
RAG_HYBRID_ALPHA=0.5  # 0.0=keyword only, 1.0=vector only, 0.5=balanced

# DynamoDB Configuration (Message Storage)
DYNAMODB_TABLE_MESSAGES=chat_messages
DYNAMODB_REGION=us-east-1

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-at-least-32-characters-long
JWT_EXPIRATION_HOURS=8
JWT_ALGORITHM=HS256

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Database Configuration (SQL Server)
DB_SERVER=your-sql-server.database.windows.net
DB_NAME=your_database_name
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_DRIVER=ODBC Driver 17 for SQL Server

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Environment
ENVIRONMENT=development
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- SQL Server (Azure SQL Database or local)
- AWS account with Bedrock access
- Weaviate Cloud Service account
- DynamoDB access (AWS)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd rag-fastapi-piloto-iA

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Database Setup (SQL Server)

```bash
# Install ODBC Driver for SQL Server (Linux)
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" | \
  sudo tee /etc/apt/sources.list.d/msprod.list

sudo apt-get update
sudo apt-get install -y msodbcsql17 unixodbc-dev
```

### Required SQL Server Stored Procedures

The application uses the following stored procedures:

- `SP_LOGIN` - User authentication and profile loading
- `SP_GET_USER_CHATS` - Retrieve user's chat list
- `SP_CREATE_CHAT` - Create new chat session
- `SP_UPDATE_CHAT_TITULO` - Update chat title
- `SP_UPDATE_CHAT_ESTADO_REGISTRO` - Soft delete chat (set to inactive)
- `SP_UPDATE_CHAT_ULTIMO_MENSAJE_FECHA` - Update last message timestamp
- `SP_IA_AREA_CONFIG_LOAD` - Load area-specific AI configuration

### DynamoDB Table Setup

Create a DynamoDB table with the following structure:

**Table Name**: `chat_messages`

**Primary Key**:
- Partition Key: `chat_id` (String) - Format: `"chat-{id}"`
- Sort Key: `created_at` (String) - Timestamp in milliseconds

**Global Secondary Index**:
- Index Name: `chat_id-estado-index`
- Partition Key: `chat_id#id_estado_registro` (String)
- Sort Key: `created_at` (String)

**Attributes**:
- `chat_id`: String (e.g., "chat-123")
- `created_at`: String (timestamp in milliseconds)
- `sender`: Number (0 = user, 1 = assistant, 2 = agent)
- `message`: String (message content)
- `id_estado_registro`: Number (1 = active, 0 = deleted)
- `chat_id#id_estado_registro`: String (composite key for GSI)

---

## ▶️ Running the API

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

API will be available at:
- **Main API**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/api/v1/health/check

---

## 📡 API Usage

### Authentication

```bash
# Login
POST /api/v1/auth/login
Content-Type: application/json

{
  "usuario": "your_username",
  "clave_acceso": "your_password"
}

# Response
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "ID_USUARIO": 1,
    "USUARIO": "your_username",
    "actual_company_area": {
      "ID_EMPRESA": 1,
      "ID_AREA": 2,
      "ID_IA_AREA": 3
    }
  },
  "result": {
    "ID_TIPO_MENSAJE": 2,
    "MENSAJE": "Login successful"
  }
}
```

### Streaming RAG Chat

```bash
# Standard RAG endpoint
POST /api/v1/rag/chat-streaming
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "message": "What is quantum computing?",
  "chat_id": "123",           // Optional - creates new chat if not provided
  "top_k": 5,                 // Optional
  "similarity_threshold": 0.7, // Optional
  "alpha": 0.5,               // Optional - hybrid search balance
  "temperature": 0.7,         // Optional
  "max_tokens": 4096          // Optional
}

# Response (Server-Sent Events)
data: {"type": "metadata", "llm_model_used": "anthropic.claude-3-5-sonnet-20241022-v2:0", "chat_id": 123}

data: {"type": "chat_created", "chat": {"ID_CHAT": 123, "ID_AREA": 2, "ID_EMPRESA": 1, "TITULO": "Nueva conversación 16/10/2025 14:30", "ULTIMO_MENSAJE_FECHA": "2025-10-16T14:30:00", "ID_ESTADO_REGISTRO": 1}}

data: {"type": "assistant_metadata", "sender": 1, "created_at": "1729098600000"}

data: {"type": "chunk", "content": "Quantum computing is a revolutionary..."}

data: {"type": "complete", "status": "success"}
```

### Agent RAG with API Integration

```bash
# Agent orchestrator endpoint with external API access
POST /api/v1/rag/agent-streaming
Authorization: Bearer <jwt_token>
X-External-Token: <external_api_token>
Content-Type: application/json

{
  "message": "List all developers with React skills",
  "top_k": 5,                 // Optional
  "similarity_threshold": 0.7, // Optional
  "alpha": 0.5,               // Optional
  "temperature": 0.3,         // Optional
  "max_tokens": 2048          // Optional
}

# Response includes API data in context
data: {"type": "assistant_metadata", "sender": 2, "created_at": "1729098600000"}

data: {"type": "chunk", "content": "Based on the talent database..."}

data: {"type": "complete", "status": "success"}
```

### Chat Management

```bash
# Get user's chats
GET /api/v1/chats/get_chats
Authorization: Bearer <jwt_token>

# Update chat title
PATCH /api/v1/chats/{chat_id}
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "titulo": "New Chat Title"
}

# Delete chat (soft delete)
DELETE /api/v1/chats/{chat_id}
Authorization: Bearer <jwt_token>

# Get messages for a chat
GET /api/v1/messages/chat/{chat_id}
Authorization: Bearer <jwt_token>
```

---

## 🏗️ Architecture Details

### Hexagonal Architecture Implementation

1. **Domain Layer** (`/domain/ports`): Defines interfaces for external services
2. **Infrastructure Layer** (`/infrastructure`): Implements domain interfaces for specific technologies
3. **Services Layer** (`/services`): Contains business logic and orchestrates domain operations
4. **API Layer** (`/api`): Handles HTTP requests/responses and dependency injection
5. **Core Layer** (`/core`): Configuration, database, and dependency injection container

### Key Components

- **RagService**: Orchestrates the complete RAG flow (embeddings → hybrid search → LLM streaming)
- **ChatService**: Manages chat CRUD operations using SQL Server stored procedures
- **MessageService**: Handles DynamoDB message persistence and retrieval
- **Container**: Dependency injection using Service Locator pattern
- **AWS Bedrock Converse Provider**: Implements streaming LLM with async thread pool and timeout protection
- **Weaviate Repository**: Vector database operations with hybrid search and dual-property filtering
- **Agent Orchestrator**: Intelligent query analysis and workflow generation
- **JWT Authentication**: Secure token-based authentication with user profile management

### RAG Flow (Standard Mode)

1. **Authentication**: JWT token validation and user extraction
2. **Chat Creation**: Auto-create chat session if not provided (SP_CREATE_CHAT)
3. **Message Storage**: Save user message to DynamoDB
4. **Configuration Loading**: Load area-specific AI configuration (SP_IA_AREA_CONFIG_LOAD)
5. **Embedding Generation**: Convert user query to vector using AWS Bedrock Titan
6. **Hybrid Search**: Query Weaviate with vector + BM25, filtering by company/area
   - Collection: `EMPR{company_id}`
   - Filter: `area_id == "AREA{area_id}" OR area == "Default"`
7. **Context Preparation**: Format retrieved documents with source metadata
8. **LLM Generation**: Stream response using AWS Bedrock Converse API
9. **Message Persistence**: Save assistant response to DynamoDB
10. **Chat Update**: Update last message timestamp (SP_UPDATE_CHAT_ULTIMO_MENSAJE_FECHA)
11. **Response Streaming**: Real-time delivery via Server-Sent Events

### Agent Mode Flow

1. **Query Analysis**: Orchestrator analyzes user intent and determines workflow
2. **Task Generation**: Dynamic task list (embedding, retrieval, API calls, LLM response)
3. **Task Execution**: Sequential execution with failure handling
4. **API Integration**: External API calls with httpx (GET/POST)
5. **Context Aggregation**: Combine vector search results and API responses
6. **LLM Response**: Stream final answer with all gathered context

### Multi-Tenant Architecture

**Company Isolation**:
- Each company has its own Weaviate collection
- Collection naming: `EMPR{company_id}` (e.g., "EMPR123")

**Area Filtering**:
- Documents tagged with `area_id` property (e.g., "AREA456")
- Default documents tagged with `area` property set to "Default"
- Dual-property filter: `area_id == "AREA{id}" OR area == "Default"`

**Benefits**:
- Complete data isolation between companies
- Area-specific document access within companies
- Shared/default documents available to all areas

### Timeout Protection

**AWS Bedrock Converse API**:
- Connection timeout: 30 seconds
- Read timeout: 120 seconds (2 minutes)
- Async thread pool execution to prevent event loop blocking
- Graceful error handling for network issues

---

## 🧩 Extending the Project

### Adding New LLM Provider

1. Implement `LLMPort` interface in `app/domain/ports/llm_port.py`
2. Create adapter in `app/infrastructure/llm/your_provider.py`
3. Create model config in `app/infrastructure/llm/model_configs/your_model_config.py`
4. Update `container.py` to instantiate your provider
5. Add configuration in `config.py`

### Adding New Vector Database

1. Implement `VectorStorePort` interface
2. Create repository in `app/infrastructure/vectorstores/`
3. Update dependency injection container
4. Add configuration settings

### Adding New Authentication Method

1. Extend `auth_service.py` with new authentication logic
2. Update JWT utilities if needed
3. Modify API endpoints in `auth.py`

### Adding New External API Integration

1. Add API configuration to `available_apis` in `rag_service.py`
2. Update task generator to handle new API endpoints
3. Implement API client methods in `api_client.py`

---

## 📊 Recommended Configuration

### AWS Bedrock Models

**Best for Production RAG:**
- **LLM**: `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - Pros: Excellent reasoning, very reliable, great instruction following
  - Best for: Production RAG systems requiring high-quality responses

**Cost-Effective Alternative:**
- **LLM**: `anthropic.claude-3-haiku-20240307-v1:0`
  - Pros: Fast, consistent, cost-effective
  - Best for: High-volume, simpler queries

**Embeddings:**
- **Model**: `amazon.titan-embed-text-v2:0`
  - Dimensions: 1024
  - Best performance for multilingual content
  - Good balance of quality and cost

**Agent Orchestrator:**
- **Model**: `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - Lower temperature (0.3) for consistent analysis
  - Smaller max_tokens (2048) for faster query analysis

### Hybrid Search Configuration

- **RAG_HYBRID_ALPHA**: 0.5 (balanced)
  - 0.0: BM25 keyword search only
  - 0.5: Balanced hybrid (recommended)
  - 1.0: Vector similarity only

### Why Claude > Other Models for RAG:
- ✅ Superior instruction following
- ✅ Consistent response quality
- ✅ Better temperature control
- ✅ No repetitive loops
- ✅ Respects length limits
- ✅ Excellent context understanding

---

## 🔧 Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS CLI is configured or use IAM roles
2. **SQL Server Connection**: Verify ODBC driver installation and connection string
3. **Weaviate Access**: Check API key and cluster URL format
4. **JWT Errors**: Verify secret key length (minimum 32 characters)
5. **DynamoDB Access**: Ensure proper IAM permissions for table operations
6. **Timeout Errors**: Check network connectivity and increase timeout values if needed
7. **Pydantic Validation**: Ensure all required environment variables are set

### Health Checks

- **API Health**: `GET /api/v1/health/check`
- **Database**: Automatic connection testing on startup
- **AWS Services**: Validated during dependency injection
- **Weaviate**: Connection tested during repository initialization

### Debug Mode

Enable detailed logging:
```env
LOG_LEVEL=DEBUG
API_DEBUG=true
```

---

## 🚨 Important Notes

### Async Execution
- AWS Bedrock calls use `asyncio.to_thread()` to prevent blocking
- Timeouts configured to handle slow network conditions
- Streaming responses are non-blocking

### Message Storage
- All messages stored in DynamoDB with composite keys
- GSI enables efficient filtering by chat and status
- Messages preserved even when chat is soft-deleted

### Chat Sessions
- Auto-create new chat if `chat_id` not provided
- Title format: "Nueva conversación DD/MM/YYYY HH:MM"
- Soft delete preserves data (ID_ESTADO_REGISTRO = 0)

### Security
- JWT tokens include user ID and company/area information
- Tokens expire after configured hours (default: 8)
- All endpoints require authentication except login and health check

---

## 📝 License

This project is licensed under the MIT License.
