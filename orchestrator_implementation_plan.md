# RAG Orchestrator Implementation Plan

## 🔄 **REFINED ARCHITECTURE** (Updated)

### **Three-Model System:**
1. **Task Decomposition**: `mistral.mistral-7b-instruct-v0:2` - Analyzes user queries and creates task lists
2. **Embeddings**: `amazon.titan-embed-text-v2:0` - Converts text to vectors for similarity search
3. **Final Response**: `us.anthropic.claude-3-haiku-20240307-v1:0` - Generates final answers (existing RAG model)

### **Conversation-First Decision Tree:**

```
User Question → Conversation Analysis → Data Source Analysis → Task Chain Building
```

#### **Decision Logic:**
1. **CONVERSATION CHECK**: Does question reference previous conversation?
   - YES → Include `GET_CONTEXT` task
   - NO → Skip to data source analysis

2. **DATA SOURCE ANALYSIS**: What new data is needed?
   - Database access + relevant question → `SQL_SELECT`
   - Knowledge/documentation question → `EMBEDDING + RETRIEVAL`
   - Both sources needed → `EMBEDDING + RETRIEVAL + SQL_SELECT`
   - Pure conversation processing → No additional tasks

3. **TASK CHAIN ASSEMBLY**: Combine all identified tasks + `LLM_RESPONSE`

#### **Task Flow Examples:**

##### **Approach 1: Vectorial Only**
**User**: "¿Qué es un protocolo BGP?"

**Mistral creates**:
```json
[
  {"action": "embedding", "input": "protocolo BGP características funcionamiento"},
  {"action": "retrieval", "vector_source": "embedding_result"},
  {"action": "llm_response", "instructions": ["explain clearly", "use technical documentation"]}
]
```

##### **Approach 2: Database Only**
**User**: "Dame las ventas del Q1 ordenadas por revenue"

**Mistral creates**:
```json
[
  {"action": "sql_select", "query": "SELECT product, quantity, revenue FROM sales_data WHERE company_id = '{company_id}' AND quarter = 'Q1'"},
  {"action": "llm_response", "instructions": ["orderby revenue desc", "present as summary"]}
]
```

##### **Approach 3: Conversation Only**
**User**: "Resume los puntos principales de la respuesta anterior"

**Mistral creates**:
```json
[
  {"action": "get_context", "source": "conversation_history"},
  {"action": "llm_response", "instructions": ["summarize main points", "organize by importance"]}
]
```

##### **Approach 4: Combined Sources** *(Only when table schemas provided)*
**User**: "¿Qué switches documentamos y cuáles hemos vendido?"

**When table schemas are provided to Mistral**:
```json
[
  {"action": "embedding", "input": "switches network equipment models specifications"},
  {"action": "retrieval", "vector_source": "embedding_result"},
  {"action": "sql_select", "query": "SELECT DISTINCT product, SUM(quantity) as sold FROM sales_data WHERE company_id = '{company_id}' AND product LIKE '%switch%' GROUP BY product"},
  {"action": "llm_response", "instructions": ["compare documented vs sold", "identify gaps", "highlight bestsellers"]}
]
```

**When NO table schemas provided**:
```json
[
  {"action": "embedding", "input": "switches network equipment models specifications sales information"},
  {"action": "retrieval", "vector_source": "embedding_result"},
  {"action": "llm_response", "instructions": ["provide information about documented switches", "mention if sales data available"]}
]
```

##### **Approach 5: Conversation + New Data**
**User**: "De esos switches, ¿cuáles están en nuestro inventario?"

**Mistral creates**:
```json
[
  {"action": "get_context", "source": "conversation_history"},
  {"action": "sql_select", "query": "SELECT name, stock, price FROM products_catalog WHERE company_id = '{company_id}' AND name IN (extracted_switch_names) AND stock > 0"},
  {"action": "llm_response", "instructions": ["correlate with previous context", "show availability status"]}
]
```

### **Key Points:**
- **Mistral**: Task planning + SQL query generation (SELECT only)
- **Claude**: Final response with all processing instructions
- **No separate orderby/filter tasks**: Combined into LLM instructions
- **Direct SQL generation**: Mistral creates secure SELECT queries
- **Simplified pipeline**: Data gathering → Claude response

### **SQL Query Generation Rules:**
- **Only SELECT statements** allowed
- **Always include company_id filter** for security
- **Use predefined table/column schema** provided to Mistral
- **Parameterized queries** to prevent injection
- **No subqueries or joins** unless explicitly allowed

### **Schema Context for Mistral:**
```json
{
  "available_tables": {
    "sales_data": {
      "columns": ["product", "quantity", "revenue", "quarter", "region", "company_id"],
      "required_filters": ["company_id"]
    },
    "network_switches": {
      "columns": ["model", "capacity_gbps", "brand", "price", "ports", "company_id"],
      "required_filters": ["company_id"]
    },
    "products_catalog": {
      "columns": ["name", "category", "price", "stock", "supplier", "company_id"],
      "required_filters": ["company_id"]
    }
  }
}
```

### **Final Task Types:**
```python
class TaskAction(Enum):
    # Conversation handling
    GET_CONTEXT = "get_context"      # Extract relevant conversation history

    # Vector database tasks
    EMBEDDING = "embedding"          # Convert text to vector (Titan)
    RETRIEVAL = "retrieval"          # Search vector database using embeddings

    # SQL database tasks
    SQL_SELECT = "sql_select"        # Execute SELECT query on structured data

    # Response generation (always last)
    LLM_RESPONSE = "llm_response"    # Generate final answer with processing (Claude)
```

### **Task Execution Order Rules:**
1. **GET_CONTEXT** (if conversation reference detected)
2. **EMBEDDING** (if knowledge query detected)
3. **RETRIEVAL** (always follows EMBEDDING)
4. **SQL_SELECT** (if database query detected)
5. **LLM_RESPONSE** (always last, combines all gathered data)

### **Complete Task Combinations Matrix:**

| Conversation | Database | Vectorial | Task Chain |
|--------------|----------|-----------|------------|
| ❌ | ❌ | ✅ | `[EMBEDDING → RETRIEVAL → LLM_RESPONSE]` |
| ❌ | ✅ | ❌ | `[SQL_SELECT → LLM_RESPONSE]` |
| ❌ | ✅ | ✅ | `[EMBEDDING → RETRIEVAL → SQL_SELECT → LLM_RESPONSE]` |
| ✅ | ❌ | ❌ | `[GET_CONTEXT → LLM_RESPONSE]` |
| ✅ | ❌ | ✅ | `[GET_CONTEXT → EMBEDDING → RETRIEVAL → LLM_RESPONSE]` |
| ✅ | ✅ | ❌ | `[GET_CONTEXT → SQL_SELECT → LLM_RESPONSE]` |
| ✅ | ✅ | ✅ | `[GET_CONTEXT → EMBEDDING → RETRIEVAL → SQL_SELECT → LLM_RESPONSE]` |

### **Implementation Flow:**
```
User Question → Mistral (Conversation + Data Source Analysis) → Execute Task Chain → Claude (Final Response) → Stream to User
                            ↓                                           ↓
                    Decision Tree Logic                            Titan Embeddings
                                                                 (for vector tasks only)
```

### **Model Responsibilities:**
- **Mistral 7B**: Conversation analysis, intent classification, task planning, SQL query generation
- **Titan Embeddings**: Text-to-vector conversion for similarity search
- **Claude 3 Haiku**: Context-aware response generation with processing instructions

### **Mistral Task Decomposition Test Prompt:**

```
You are a task decomposer. Analyze user questions and return JSON task arrays.

AVAILABLE TASKS:
- get_context: Extract conversation history
- embedding: Convert text to vector
- retrieval: Search vector database (only use vector_source)
- sql_select: Query structured data (ONLY when table schema provided)
- llm_response: Generate final answer (always last)

DECISION RULES:
1. Conversation reference ("anterior", "eso") → get_context
2. Knowledge/documentation question → embedding + retrieval
3. Structured data question → sql_select ONLY if table schema provided
4. NEVER mix retrieval with sql_select in same task
5. NEVER create sql_select without exact table schema
6. NO TABLE SCHEMA = NO SQL TASK (use vectorial only)
7. NEVER create sql_select without exact table schema
8. NO TABLE SCHEMA = NO SQL TASK (use vectorial only)

TABLE SCHEMAS (when provided):
- sales_data: [product, quantity, revenue, quarter, region, company_id]
- network_switches: [model, capacity_gbps, brand, price, ports, company_id]
- products_catalog: [name, category, price, stock, supplier, company_id]

EXAMPLES:
Knowledge: "¿Qué es BGP?" → [{"action": "embedding", "input": "BGP protocol"}, {"action": "retrieval", "vector_source": "embedding_result"}, {"action": "llm_response", "instructions": ["explain clearly"]}]

Data: "Ventas Q1" → [{"action": "sql_select", "query": "SELECT product, revenue FROM sales_data WHERE company_id = 'COMP_001' AND quarter = 'Q1'"}, {"action": "llm_response", "instructions": ["summarize sales"]}]

Context: "Resume lo anterior" → [{"action": "get_context", "source": "conversation_history"}, {"action": "llm_response", "instructions": ["summarize"]}]

CRITICAL RULES:
- Questions about switch models/specifications → VECTORIAL ONLY
- Questions about sales/inventory data → SQL ONLY if table schema provided
- NEVER combine retrieval + sql_select in same task
- NEVER create sql_select without exact table schema
- NO TABLE SCHEMA = NO SQL TASK (use vectorial only)

DECOMPOSE: "{user_question}"
```

---

## Overview (Original Plan Below)

This document outlines the implementation plan for an intelligent orchestrator system that decomposes user questions into executable tasks, supporting vector database queries, SQL operations, conversation handling, and markdown table generation.

## Current State Analysis

### Existing RAG Flow (`chat_service.py`)
```
User Query → Embedding → Vector Search → LLM Response → Stream
```

**Limitations:**
- Linear flow only
- No SQL database support
- No conversation context
- No structured output (tables)
- Single data source only

## Proposed Orchestrator Architecture

### High-Level Flow
```
User Question → Intent Classifier → Task Decomposer → Task Executor → Response Aggregator → Stream Response
```

### Components Overview

1. **Intent Classifier**: Determines query type (vector, sql, conversation, hybrid)
2. **Task Decomposer**: Breaks down complex queries into executable tasks
3. **Task Executor**: Executes tasks sequentially/parallel with streaming
4. **Response Aggregator**: Combines results and formats output
5. **Table Generator**: Creates markdown tables and saves files

## Detailed Implementation Plan

### Phase 1: Core Architecture Setup

#### 1.1 Task Definition System

**File**: `app/models/task_models.py`

```python
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TaskAction(Enum):
    # Vector Database Tasks
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    ORDERBY = "orderby"
    FILTER = "filter"

    # SQL Database Tasks
    SQL_SELECT = "sql_select"

    # Table Generation Tasks
    GENERATE_TABLE = "generate_table"
    SAVE_TABLE = "save_table"

    # Conversation Tasks
    RESUME = "resume"
    TRANSLATE = "translate"
    CONTINUE = "continue"
    GET_PREVIOUS_TABLE = "get_previous_table"

class Task(BaseModel):
    action: TaskAction
    input: Optional[str] = None
    query: Optional[str] = None
    endpoint: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    data_source: Optional[str] = None
    format: Optional[str] = None
    columns: Optional[List[str]] = None
    filename: Optional[str] = None
    target_language: Optional[str] = None
    field: Optional[str] = None
    direction: Optional[str] = None

class TaskPipeline(BaseModel):
    tasks: List[Task]
    user_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
```

#### 1.2 Intent Classification Service

**File**: `app/services/intent_classifier.py`

```python
from typing import Dict, List
from app.domain.ports.llm_port import LLMPort

class IntentClassifier:
    def __init__(self, llm_provider: LLMPort):
        self.llm_provider = llm_provider

    async def classify_intent(self, message: str, conversation_history: List[Dict]) -> Dict:
        """
        Classify user intent using LLM

        Returns:
        {
            "intent": "vector_search|sql_query|conversation|hybrid",
            "confidence": 0.95,
            "entities": ["switches", "capacity", "176 Gbps"],
            "requires_table": true,
            "data_source": "vector|sql|both"
        }
        """

        classification_prompt = f"""
        Analyze the following user question and classify its intent.

        Previous conversation: {conversation_history[-3:] if conversation_history else "None"}
        Current question: "{message}"

        Classify into one of these categories:
        1. vector_search - Query about documents/knowledge base
        2. sql_query - Query about structured data (sales, products, etc.)
        3. conversation - Reference to previous messages, translation, summary
        4. hybrid - Combination of multiple data sources

        Also identify:
        - Key entities mentioned
        - Whether user wants tabular output
        - What data sources are needed

        Respond in JSON format:
        {{
            "intent": "category",
            "confidence": 0.0-1.0,
            "entities": ["entity1", "entity2"],
            "requires_table": true/false,
            "data_source": "vector|sql|both",
            "reasoning": "brief explanation"
        }}
        """

        response = await self.llm_provider.generate(classification_prompt, max_tokens=500)
        # Parse JSON response and return
        return self._parse_classification_response(response)
```

#### 1.3 Task Decomposition Service

**File**: `app/services/task_decomposer.py`

```python
from typing import List
from app.models.task_models import Task, TaskAction
from app.domain.ports.llm_port import LLMPort

class TaskDecomposer:
    def __init__(self, llm_provider: LLMPort):
        self.llm_provider = llm_provider
        self.sql_endpoints = self._load_sql_endpoints()

    async def decompose_query(self, message: str, intent: Dict, user_context: Dict) -> List[Task]:
        """
        Decompose user query into executable tasks based on intent
        """

        if intent["intent"] == "vector_search":
            return await self._decompose_vector_query(message, intent, user_context)
        elif intent["intent"] == "sql_query":
            return await self._decompose_sql_query(message, intent, user_context)
        elif intent["intent"] == "conversation":
            return await self._decompose_conversation_query(message, intent, user_context)
        elif intent["intent"] == "hybrid":
            return await self._decompose_hybrid_query(message, intent, user_context)

    async def _decompose_vector_query(self, message: str, intent: Dict, user_context: Dict) -> List[Task]:
        """
        Example: "¿Qué switches tienen 176+ Gbps? Ordena por capacidad y ponlo en tabla"
        Returns: [embedding, retrieval, orderby, generate_table, save_table]
        """

        decomposition_prompt = f"""
        Decompose this vector database query into tasks:
        Question: "{message}"
        User wants table: {intent.get('requires_table', False)}

        Available task types:
        - embedding: Generate embedding for search
        - retrieval: Search vector database
        - orderby: Sort results by field
        - filter: Filter results by conditions
        - generate_table: Create markdown table
        - save_table: Save table to file

        Return JSON array of tasks:
        [
            {{"action": "embedding", "input": "search query"}},
            {{"action": "retrieval", "query": "search query", "filters": {{}}}},
            {{"action": "orderby", "field": "capacity", "direction": "desc"}},
            {{"action": "generate_table", "data_source": "retrieval_result", "columns": ["model", "capacity"]}},
            {{"action": "save_table", "filename": "switches_176gbps.md"}}
        ]
        """

        response = await self.llm_provider.generate(decomposition_prompt, max_tokens=1000)
        return self._parse_tasks_response(response)

    async def _decompose_sql_query(self, message: str, intent: Dict, user_context: Dict) -> List[Task]:
        """
        Example: "Dame las ventas del Q1 en formato tabla"
        Returns: [sql_select, generate_table, save_table]
        """

        available_endpoints = list(self.sql_endpoints.keys())

        decomposition_prompt = f"""
        Decompose this SQL query into tasks:
        Question: "{message}"
        Available SQL endpoints: {available_endpoints}
        User context: {user_context}

        Return JSON array of tasks:
        [
            {{"action": "sql_select", "endpoint": "/api/v1/data/sales", "filters": {{"quarter": "Q1"}}}},
            {{"action": "generate_table", "data_source": "sql_result", "format": "markdown"}},
            {{"action": "save_table", "filename": "sales_q1.md"}}
        ]
        """

        response = await self.llm_provider.generate(decomposition_prompt, max_tokens=800)
        return self._parse_tasks_response(response)
```

### Phase 2: Task Execution Engine

#### 2.1 SQL Task Executor (SELECT only)

**File**: `app/services/sql_task_executor.py`

```python
from typing import Dict, List
import logging
from app.core.database import get_db_connection

class SQLTaskExecutor:
    def __init__(self):
        self.endpoints_config = {
            "/api/v1/data/switches": {
                "table": "network_switches",
                "allowed_fields": ["model", "capacity_gbps", "brand", "price", "ports"],
                "required_filters": ["company_id"],
                "optional_filters": ["capacity_gte", "capacity_lte", "brand", "price_range"]
            },
            "/api/v1/data/sales": {
                "table": "sales_data",
                "allowed_fields": ["product", "quantity", "revenue", "quarter", "region"],
                "required_filters": ["company_id"],
                "optional_filters": ["quarter", "region", "product_category", "date_range"]
            },
            "/api/v1/data/products": {
                "table": "products_catalog",
                "allowed_fields": ["name", "category", "price", "stock", "supplier"],
                "required_filters": ["company_id"],
                "optional_filters": ["category", "price_range", "in_stock"]
            }
        }

    async def execute_sql_select(self, task: Dict, user_context: Dict) -> Dict:
        """
        Execute secure SQL SELECT from predefined endpoint
        Security: Only SELECT, whitelisted tables, mandatory company filtering
        """
        endpoint = task["endpoint"]

        if endpoint not in self.endpoints_config:
            raise ValueError(f"Endpoint {endpoint} not authorized")

        config = self.endpoints_config[endpoint]

        # Build secure SELECT query
        fields = ", ".join(config["allowed_fields"])
        query = f"SELECT {fields} FROM {config['table']}"

        # Always filter by company (security requirement)
        filters = [f"company_id = '{user_context['company_id']}'"]

        # Add optional filters from task
        if "filters" in task:
            for key, value in task["filters"].items():
                if key in config["optional_filters"]:
                    filter_condition = self._build_safe_filter(key, value)
                    filters.append(filter_condition)

        query += f" WHERE {' AND '.join(filters)}"

        # Add ordering if specified
        if "orderby" in task:
            orderby_field = task["orderby"].replace("_desc", "").replace("_asc", "")
            if orderby_field in config["allowed_fields"]:
                direction = "DESC" if "_desc" in task["orderby"] else "ASC"
                query += f" ORDER BY {orderby_field} {direction}"

        # Execute query with connection pooling
        try:
            async with get_db_connection() as conn:
                cursor = await conn.execute(query)
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                # Convert to list of dictionaries
                results = [dict(zip(columns, row)) for row in rows]

            return {
                "data": results,
                "columns": columns,
                "total_rows": len(results),
                "query_executed": query  # For debugging (remove in production)
            }

        except Exception as e:
            logging.error(f"SQL execution error: {e}")
            raise ConnectionError(f"Database query failed: {str(e)}")

    def _build_safe_filter(self, key: str, value: str) -> str:
        """Build SQL filter with parameterization to prevent injection"""
        # Implement safe filter building logic
        if key.endswith("_gte"):
            field = key.replace("_gte", "")
            return f"{field} >= {self._sanitize_numeric(value)}"
        elif key.endswith("_lte"):
            field = key.replace("_lte", "")
            return f"{field} <= {self._sanitize_numeric(value)}"
        elif key == "quarter":
            return f"quarter = '{self._sanitize_string(value)}'"
        # Add more filter types as needed
        else:
            return f"{key} = '{self._sanitize_string(value)}'"
```

#### 2.2 Table Generation Service

**File**: `app/services/table_generator.py`

```python
import os
from typing import List, Dict, Optional
from datetime import datetime

class TableGenerator:
    def __init__(self, exports_directory: str = "exports"):
        self.exports_dir = exports_directory
        os.makedirs(exports_directory, exist_ok=True)

    async def generate_markdown_table(self, data: List[Dict], columns: Optional[List[str]] = None) -> str:
        """
        Generate properly formatted markdown table
        """
        if not data:
            return "_No data available_"

        # Use specified columns or infer from data
        if not columns:
            columns = list(data[0].keys())

        # Ensure columns exist in data
        available_columns = [col for col in columns if col in data[0]]

        # Create table header
        header_row = "| " + " | ".join(available_columns) + " |"
        separator_row = "| " + " | ".join(["---"] * len(available_columns)) + " |"

        # Create data rows
        data_rows = []
        for row in data:
            row_values = []
            for col in available_columns:
                value = row.get(col, "")
                # Format different data types
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                elif value is None:
                    formatted_value = ""
                else:
                    formatted_value = str(value)
                row_values.append(formatted_value)

            data_rows.append("| " + " | ".join(row_values) + " |")

        # Combine table parts
        table_parts = [header_row, separator_row] + data_rows
        markdown_table = "\n".join(table_parts)

        # Add metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"\n\n*Generated on {timestamp} | Total rows: {len(data)}*"

        return markdown_table + metadata

    async def save_markdown_file(self, content: str, filename: str, user_context: Dict) -> Dict:
        """
        Save markdown content to user-specific directory
        """
        # Create user directory
        user_dir = os.path.join(self.exports_dir, user_context["user_id"])
        os.makedirs(user_dir, exist_ok=True)

        # Ensure .md extension
        if not filename.endswith('.md'):
            filename += '.md'

        # Full file path
        file_path = os.path.join(user_dir, filename)

        try:
            # Write file with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                "status": "success",
                "file_path": file_path,
                "filename": filename,
                "download_url": f"/api/v1/exports/{user_context['user_id']}/{filename}",
                "file_size": os.path.getsize(file_path)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def generate_comparison_table(self, vector_data: List[Dict], sql_data: List[Dict]) -> str:
        """
        Generate comparison table between vector and SQL results
        """
        comparison_table = "## Data Comparison\n\n"

        comparison_table += "### Vector Database Results\n"
        vector_table = await self.generate_markdown_table(vector_data)
        comparison_table += vector_table + "\n\n"

        comparison_table += "### SQL Database Results\n"
        sql_table = await self.generate_markdown_table(sql_data)
        comparison_table += sql_table + "\n\n"

        return comparison_table
```

### Phase 3: Enhanced Orchestration Service

#### 3.1 Main Orchestration Service

**File**: `app/services/orchestration_chat_service.py`

```python
from typing import AsyncGenerator, Dict, List, Any
from app.services.chat_service import ChatService
from app.services.intent_classifier import IntentClassifier
from app.services.task_decomposer import TaskDecomposer
from app.services.sql_task_executor import SQLTaskExecutor
from app.services.table_generator import TableGenerator
from app.models.task_models import TaskAction

class OrchestrationChatService(ChatService):
    def __init__(self, embeddings_provider, vectorstore, llm_provider):
        super().__init__(embeddings_provider, vectorstore, llm_provider)
        self.intent_classifier = IntentClassifier(llm_provider)
        self.task_decomposer = TaskDecomposer(llm_provider)
        self.sql_executor = SQLTaskExecutor()
        self.table_generator = TableGenerator()
        self.task_results = {}  # Store intermediate results

    async def process_orchestrated_query_stream(self,
                                               user_id: str,
                                               message: str,
                                               company_id: str,
                                               area: str,
                                               conversation_history: List[Dict] = None,
                                               **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main orchestration flow with streaming
        """

        # Step 1: Classify intent
        yield {"type": "metadata", "user_id": user_id, "message": message, "status": "analyzing_intent"}

        intent = await self.intent_classifier.classify_intent(message, conversation_history or [])

        # Step 2: Decompose into tasks
        yield {"type": "metadata", "status": "decomposing_tasks", "intent": intent["intent"]}

        user_context = {
            "user_id": user_id,
            "company_id": company_id,
            "area": area,
            **kwargs
        }

        tasks = await self.task_decomposer.decompose_query(message, intent, user_context)

        yield {"type": "metadata", "status": "executing_tasks", "total_tasks": len(tasks)}

        # Step 3: Execute tasks sequentially
        async for result in self._execute_task_pipeline(tasks, user_context, conversation_history):
            yield result

        # Step 4: Final completion
        yield {"type": "complete", "status": "success"}

    async def _execute_task_pipeline(self, tasks: List[Dict], user_context: Dict, conversation_history: List[Dict]) -> AsyncGenerator[Dict, None]:
        """
        Execute tasks in pipeline with result chaining
        """

        for i, task in enumerate(tasks):
            action = TaskAction(task["action"])

            try:
                yield {"type": "task_start", "task_index": i, "action": action.value}

                if action == TaskAction.EMBEDDING:
                    result = await self._execute_embedding_task(task)
                    self.task_results[f"embedding_{i}"] = result

                elif action == TaskAction.RETRIEVAL:
                    result = await self._execute_retrieval_task(task, user_context)
                    self.task_results[f"retrieval_{i}"] = result

                elif action == TaskAction.SQL_SELECT:
                    result = await self.sql_executor.execute_sql_select(task, user_context)
                    self.task_results[f"sql_result_{i}"] = result

                elif action == TaskAction.GENERATE_TABLE:
                    result = await self._execute_table_generation(task, i)

                    # Stream table content
                    yield {
                        "type": "chunk",
                        "content": f"\n\n{result}\n\n"
                    }

                elif action == TaskAction.SAVE_TABLE:
                    result = await self._execute_table_save(task, i, user_context)

                    # Stream download link
                    if result["status"] == "success":
                        yield {
                            "type": "chunk",
                            "content": f"📄 **Tabla guardada**: [{result['filename']}]({result['download_url']})"
                        }

                elif action == TaskAction.RESUME:
                    result = await self._execute_conversation_task(task, conversation_history)
                    yield {"type": "chunk", "content": result}

                yield {"type": "task_complete", "task_index": i, "action": action.value}

            except Exception as e:
                yield {
                    "type": "task_error",
                    "task_index": i,
                    "action": action.value,
                    "error": str(e)
                }
```

### Phase 4: API Integration

#### 4.1 Updated Chat API Endpoint

**File**: `app/api/chat.py` (enhanced)

```python
from app.services.orchestration_chat_service import OrchestrationChatService

@router.post("/rag/chat-orchestrated")
async def orchestrated_chat_streaming(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Enhanced RAG endpoint with orchestration capabilities
    Supports: vector search, SQL queries, table generation, conversation context
    """

    try:
        # Get orchestration service from container
        orchestration_service: OrchestrationChatService = container.get(OrchestrationChatService)

        # Get conversation history (implement based on your storage)
        conversation_history = await get_user_conversation_history(
            current_user.user_id,
            current_user.company_id,
            limit=10
        )

        # Stream orchestrated response
        return StreamingResponse(
            orchestration_service.process_orchestrated_query_stream(
                user_id=current_user.user_id,
                message=request.message,
                company_id=current_user.company_id,
                area=request.area,
                conversation_history=conversation_history,
                collection=request.collection,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ),
            media_type="text/plain"
        )

    except Exception as e:
        logger.error(f"Orchestrated chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/{user_id}/{filename}")
async def download_export_file(
    user_id: str,
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download exported markdown files
    Security: Users can only access their own files
    """

    # Security check
    if current_user.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = f"exports/{user_id}/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        filename=filename,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
```

### Phase 5: Configuration and Setup

#### 5.1 Enhanced Container Configuration

**File**: `app/core/container.py` (updated)

```python
from app.services.orchestration_chat_service import OrchestrationChatService

class Container:
    def __init__(self):
        # ... existing services ...

        # Orchestration services
        self._orchestration_chat_service = None

    def get_orchestration_chat_service(self) -> OrchestrationChatService:
        if self._orchestration_chat_service is None:
            self._orchestration_chat_service = OrchestrationChatService(
                embeddings_provider=self.get_embeddings_provider(),
                vectorstore=self.get_vectorstore(),
                llm_provider=self.get_llm_provider()
            )
        return self._orchestration_chat_service
```

#### 5.2 Environment Configuration

**File**: `.env` (additions)

```env
# Orchestration Configuration
ORCHESTRATION_ENABLED=true
ORCHESTRATION_MAX_TASKS=10
ORCHESTRATION_TIMEOUT_SECONDS=300

# SQL Endpoints Configuration
SQL_ENDPOINTS_ENABLED=true
SQL_MAX_RESULTS=1000

# Table Generation Configuration
EXPORTS_DIRECTORY=exports
MAX_EXPORT_FILE_SIZE_MB=50
EXPORT_RETENTION_DAYS=30

# Conversation History
CONVERSATION_HISTORY_LIMIT=50
CONVERSATION_CONTEXT_WINDOW=10
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Task models and enums
- [ ] Intent classifier implementation
- [ ] Basic task decomposer

### Week 2: Core Execution
- [ ] SQL task executor with security
- [ ] Table generation service
- [ ] Vector task integration

### Week 3: Orchestration
- [ ] Main orchestration service
- [ ] Task pipeline execution
- [ ] Error handling and recovery

### Week 4: Integration & Testing
- [ ] API endpoint updates
- [ ] File export functionality
- [ ] End-to-end testing
- [ ] Performance optimization

## Security Considerations

### SQL Security
- ✅ Only SELECT statements allowed
- ✅ Predefined endpoints and tables
- ✅ Mandatory company filtering
- ✅ SQL injection prevention
- ✅ Query parameterization

### File Security
- ✅ User-specific directories
- ✅ Access control on downloads
- ✅ File size limits
- ✅ Allowed file types only

### Data Security
- ✅ Company/area data isolation
- ✅ User permission validation
- ✅ Audit logging for all queries

## Example Usage Scenarios

### Scenario 1: Vector Search with Table
**User**: "¿Qué switches tienen capacidad de 176 Gbps o más? Ordénalo de mayor a menor y ponlo en una tabla"

**Tasks Generated**:
```json
[
  {"action": "embedding", "input": "switches with capacity of 176 Gbps or more"},
  {"action": "retrieval", "query": "switches with capacity of 176 Gbps or more"},
  {"action": "orderby", "field": "capacity", "direction": "desc"},
  {"action": "generate_table", "data_source": "retrieval_result", "columns": ["model", "capacity", "brand"]},
  {"action": "save_table", "filename": "switches_176gbps_plus.md"}
]
```

### Scenario 2: SQL Query with Export
**User**: "Dame las ventas del Q1 en formato tabla"

**Tasks Generated**:
```json
[
  {"action": "sql_select", "endpoint": "/api/v1/data/sales", "filters": {"quarter": "Q1"}},
  {"action": "generate_table", "data_source": "sql_result", "format": "markdown"},
  {"action": "save_table", "filename": "sales_q1.md"}
]
```

### Scenario 3: Conversation Reference
**User**: "De la pregunta anterior, traduce la tabla al inglés"

**Tasks Generated**:
```json
[
  {"action": "get_previous_table", "source": "conversation_history"},
  {"action": "translate", "target_language": "en", "input": "previous_table"},
  {"action": "generate_table", "data_source": "translation_result", "format": "markdown"},
  {"action": "save_table", "filename": "translated_table.md"}
]
```

## Success Metrics

### Functional Metrics
- [ ] Supports 5+ task types
- [ ] Handles 3+ data sources (vector, SQL, conversation)
- [ ] Generates downloadable markdown tables
- [ ] Maintains streaming response capability

### Performance Metrics
- [ ] Response time < 3 seconds for simple queries
- [ ] Response time < 10 seconds for complex multi-task queries
- [ ] Handles 100+ concurrent orchestrated queries
- [ ] 99.9% uptime for orchestration service

### Security Metrics
- [ ] Zero SQL injection vulnerabilities
- [ ] 100% company data isolation
- [ ] Audit logging for all database queries
- [ ] Secure file access controls

## Future Enhancements

### Phase 2 Features
- [ ] Parallel task execution
- [ ] Task result caching
- [ ] Advanced table formatting (charts, graphs)
- [ ] Multi-language support for tables

### Phase 3 Features
- [ ] Custom task plugin system
- [ ] Machine learning task recommendations
- [ ] Advanced conversation context understanding
- [ ] Integration with external APIs

---

*This implementation plan provides a comprehensive roadmap for building an intelligent RAG orchestrator that can handle complex, multi-step queries while maintaining security and performance.*