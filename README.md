# Qamaq RAG (Retrieval-Augmented Generation with FastAPI)

Qamaq RAG is a modular **FastAPI** project designed with **Hexagonal Architecture (Ports & Adapters)** to enable **Retrieval-Augmented Generation (RAG)** using a vector database and different LLM/Embeddings providers (AWS Bedrock, OpenAI, etc.).

The system is flexible and fully configurable via environment variables, allowing you to switch providers without changing the application code.

---

## 🚀 Features

- **Hexagonal Architecture** (Domain, Application, Infrastructure, API layers)
- **LLM Support**:
  - AWS Bedrock (Claude, Llama, etc.)
  - OpenAI (GPT family)
- **Embeddings Support**:
  - AWS Bedrock Titan Embeddings
  - OpenAI embeddings
- **Vector Database Support**:
  - Weaviate (default, extendable to Chroma or others)
- **Configurable** using `.env`
- **Extensible**: add new providers easily by implementing the Ports

---

## 📂 Project Structure

rag-fastapi/
├── app/
│ ├── api/ # FastAPI routes
│ │ └── chat.py
│ ├── application/ # Use cases (business logic)
│ │ └── chat_service.py
│ ├── core/ # Configuration & settings
│ │ └── config.py
│ ├── domain/
│ │ └── ports/ # Ports (interfaces)
│ │ ├── llm_port.py
│ │ ├── embeddings_port.py
│ │ └── vectorstore_port.py
│ └── infrastructure/ # Adapters (providers)
│ ├── llm/
│ │ └── aws_provider.py
│ ├── embeddings/
│ │ ├── openai_embeddings.py
│ │ └── aws_embeddings.py
│ └── vectorstores/
│ └── weaviate_repository.py
├── main.py # FastAPI entry point
├── requirements.txt # Dependencies
├── .env # Environment variables
└── README.md # Documentation


---

## ⚙️ Configuration

All configuration is managed through environment variables in `.env`.

### Example `.env`

```env
# Vector DB
VECTORDB_URL=https://example.weaviate.cloud
VECTORDB_API_KEY=your_vector_db_api_key

# Embeddings (choose provider: openai | aws)
EMBEDDINGS_PROVIDER=aws
EMBEDDINGS_MODEL_ID=amazon.titan-embed-text-v1
EMBEDDINGS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
# For OpenAI:
# EMBEDDINGS_PROVIDER=openai
# EMBEDDINGS_API_KEY=sk-xxxx
# EMBEDDINGS_MODEL=text-embedding-ada-002

# LLM (choose provider: openai | aws)
LLM_PROVIDER=aws
LLM_REGION=us-east-1
LLM_MODEL_ID=anthropic.claude-v2
LLM_API_KEY=your_aws_key
# For OpenAI:
# LLM_PROVIDER=openai
# LLM_MODEL_ID=gpt-4
# LLM_API_KEY=sk-xxxx

🛠️ Installation
# Clone repository
git clone https://github.com/your-org/rag-fastapi.git
cd rag-fastapi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

▶️ Running the API
uvicorn main:app --reload

API will be available at:
👉 http://127.0.0.1:8000

📡 API Usage
POST /chat

Request:
{
  "user_id": "user123",
  "message": "What is quantum computing?"
}

Response:
{
  "answer": "Quantum computing is a new paradigm...",
  "context": ["...vectorstore retrieved docs..."]
}

🧩 Extending the Project

To add a new LLM Provider: implement LLMPort in app/domain/ports/llm_port.py and create a new adapter in app/infrastructure/llm/.

To add a new Embeddings Provider: implement EmbeddingsPort in app/domain/ports/embeddings_port.py and create a new adapter in app/infrastructure/embeddings/.

To use a different Vector Database: implement VectorStorePort and create a new repository.

The llm models are gonna be from AWS
The embeddings model is gonna be Amazon Titan Text Embeddings V2
The LLM model is gonna be meta.llama3-8b-instruct-v1:0
The vectorial database is gonna be Weaviate Cloud Service (WCS)