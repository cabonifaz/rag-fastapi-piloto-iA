# Qamaq RAG (Retrieval-Augmented Generation with FastAPI)

Qamaq RAG is a **FastAPI** project implementing **Hexagonal Architecture (Ports & Adapters)** for **Retrieval-Augmented Generation (RAG)** using AWS Bedrock, Weaviate Cloud Service, and SQL Server for authentication.

The system provides streaming RAG responses with JWT authentication and company-specific document filtering.

---

## 🚀 Features

- **Hexagonal Architecture** (Domain Ports, Infrastructure Adapters, Services, API layers)
- **Streaming RAG** with Server-Sent Events (SSE)
- **AWS Bedrock Integration**:
  - LLM: Claude 3 Haiku/Sonnet, Llama 3
  - Embeddings: Amazon Titan Text Embeddings V2
- **Vector Database**: Weaviate Cloud Service (WCS)
- **Authentication**: JWT-based with SQL Server backend
- **Company-specific filtering**: Multi-tenant document access
- **Dependency Injection**: Custom DI container for clean architecture
- **Real-time streaming**: Live RAG responses via SSE
- **Configurable**: Full environment-based configuration

---

## 📂 Current Project Structure

```
rag-fastapi-piloto-iA/
├── main.py                          # FastAPI entry point & middleware
├── app/
│   ├── api/                        # API layer (routes/endpoints)
│   │   ├── chat.py                 # RAG streaming endpoint
│   │   ├── auth.py                 # Authentication endpoints
│   │   └── processing.py           # Document processing (commented)
│   ├── core/                       # Configuration & infrastructure
│   │   ├── config.py               # Environment settings & validation
│   │   ├── database.py             # SQL Server connection
│   │   ├── database_config.py      # Database configuration
│   │   └── container.py            # Dependency injection container
│   ├── domain/ports/               # Hexagonal architecture interfaces
│   │   ├── llm_port.py            # LLM provider interface
│   │   ├── embeddings_port.py     # Embeddings provider interface
│   │   └── vectorstore_port.py    # Vector database interface
│   ├── infrastructure/            # External service adapters
│   │   ├── llm/
│   │   │   ├── aws_provider.py    # AWS Bedrock LLM implementation
│   │   │   └── model_formats.py   # Model format strategies (Claude/Llama)
│   │   ├── embeddings/
│   │   │   └── aws_embeddings.py  # AWS Bedrock Titan embeddings
│   │   └── vectorstores/
│   │       └── weaviate_repository.py # Weaviate Cloud implementation
│   ├── services/                  # Business logic services
│   │   ├── chat_service.py        # RAG orchestration service
│   │   ├── auth_service.py        # Authentication business logic
│   │   └── document_processing_service.py # Document handling
│   ├── models/                    # Pydantic data models
│   │   ├── response_models.py     # Standardized API responses
│   │   └── user_models.py         # User & authentication models
│   └── utils/                     # Utility functions
│       ├── jwt_auth.py           # JWT token management
│       ├── password_utils.py     # Password hashing utilities
│       └── token_counter.py      # LLM token usage tracking
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
LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
LLM_REGION=us-east-1
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9

# Weaviate Configuration
WEAVIATE_URL=https://your-cluster.weaviate.cloud
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_CLASS_NAME=Documents
WEAVIATE_GRPC=your-cluster.weaviate.cloud:443

# RAG Configuration
RAG_TOP_K_RESULTS=5
RAG_SIMILARITY_THRESHOLD=0.7

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-at-least-32-characters-long
JWT_EXPIRATION_HOURS=8

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
  "status": "success",
  "chats": []
}
```

### Streaming RAG Chat

```bash
# Streaming RAG endpoint
POST /api/v1/rag/chat-streaming
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "user_id": "user123",
  "message": "What is quantum computing?",
  "company_id": "company_001",
  "area": "technology",
  "collection": "documents",  // Optional
  "top_k": 5,                // Optional
  "similarity_threshold": 0.7, // Optional
  "temperature": 0.7,        // Optional
  "max_tokens": 4096         // Optional
}

# Response (Server-Sent Events)
data: {"type": "metadata", "user_id": "user123", "message": "...", "llm_model_used": "anthropic.claude-3-haiku-20240307-v1:0"}

data: {"type": "chunk", "content": "Quantum computing is a revolutionary..."}

data: {"type": "chunk", "content": "Quantum computing is a revolutionary paradigm that leverages..."}

data: {"type": "complete", "status": "success"}
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

- **ChatService**: Orchestrates the complete RAG flow (embeddings → search → LLM)
- **Container**: Dependency injection using Service Locator pattern
- **AWS Providers**: Implements Bedrock integration for LLM and embeddings
- **Weaviate Repository**: Vector database operations with company/area filtering
- **JWT Authentication**: Secure token-based authentication with user validation

### RAG Flow

1. **Authentication**: JWT token validation and user extraction
2. **Embedding Generation**: Convert user query to vector using AWS Bedrock Titan
3. **Vector Search**: Query Weaviate with company/area filtering
4. **Context Preparation**: Format retrieved documents with source metadata
5. **LLM Generation**: Stream response using AWS Bedrock (Claude/Llama)
6. **Response Streaming**: Real-time delivery via Server-Sent Events

---

## 🧩 Extending the Project

### Adding New LLM Provider

1. Implement `LLMPort` interface in `app/domain/ports/llm_port.py`
2. Create adapter in `app/infrastructure/llm/your_provider.py`
3. Update `container.py` to instantiate your provider
4. Add configuration in `config.py`

### Adding New Vector Database

1. Implement `VectorStorePort` interface
2. Create repository in `app/infrastructure/vectorstores/`
3. Update dependency injection container

### Adding New Authentication Method

1. Extend `auth_service.py` with new authentication logic
2. Update JWT utilities if needed
3. Modify API endpoints in `auth.py`

---

## 📊 Recommended Configuration

### AWS Bedrock Models

**Best for RAG (Recommended):**
- **LLM**: `anthropic.claude-3-haiku-20240307-v1:0`
  - Pros: Consistent, follows instructions, cost-effective
  - Best for: Production RAG systems

**Most Capable (Higher Cost):**
- **LLM**: `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - Pros: Excellent reasoning, very reliable
  - Use for: Complex queries requiring deep analysis

**Embeddings:**
- **Model**: `amazon.titan-embed-text-v2:0`
  - Dimensions: 1024
  - Best performance for multilingual content

### Why Claude > Llama for RAG:
- ✅ Better instruction following
- ✅ More consistent responses
- ✅ Better temperature control
- ✅ No repetitive loops
- ✅ Respects length limits

---

## 🔧 Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS CLI is configured or use IAM roles
2. **SQL Server Connection**: Verify ODBC driver installation and connection string
3. **Weaviate Access**: Check API key and cluster URL
4. **JWT Errors**: Verify secret key length (minimum 32 characters)

### Health Checks

- **API Health**: `GET /api/v1/health/check`
- **Database**: Automatic connection testing on startup
- **AWS Services**: Validated during dependency injection

---

## 📝 License

This project is licensed under the MIT License.