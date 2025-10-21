from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


class TaskAction(Enum):
    """
    Enumeration of available task actions in the orchestration system.
    Based on the refined architecture from the implementation plan.
    """
    # Conversation handling
    GET_CONTEXT = "get_context"      # Extract relevant conversation history

    # Vector database tasks
    EMBEDDING = "embedding"          # Convert text to vector (Titan)
    RETRIEVAL = "retrieval"          # Search vector database using embeddings

    # SQL database tasks
    SQL_SELECT = "sql_select"        # Execute SELECT query on structured data

    # Response generation (always last)
    LLM_RESPONSE = "llm_response"    # Generate final answer with processing (Claude)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """
    Individual task definition in the orchestration pipeline.
    """
    model_config = ConfigDict(use_enum_values=True)

    action: TaskAction = Field(..., description="The action to be performed")
    input: Optional[str] = Field(None, description="Input text for embedding or context tasks")
    query: Optional[str] = Field(None, description="SQL query for database tasks")
    vector_source: Optional[str] = Field(None, description="Source of vector data for retrieval")
    instructions: Optional[List[str]] = Field(default_factory=list, description="Processing instructions for LLM")
    source: Optional[str] = Field(None, description="Data source identifier")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional filters or parameters")

    # Execution metadata
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error: Optional[str] = Field(None, description="Error message if task failed")
    execution_time_ms: Optional[float] = Field(None, description="Task execution time in milliseconds")


class TaskPipeline(BaseModel):
    """
    Complete task pipeline with user context and execution metadata.
    """
    model_config = ConfigDict(use_enum_values=True)

    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    user_id: str = Field(..., description="User who initiated the pipeline")
    company_id: str = Field(..., description="Company context for data filtering")
    area: str = Field(..., description="Area context for document filtering")

    # Task execution
    tasks: List[Task] = Field(..., description="Ordered list of tasks to execute")
    current_task_index: int = Field(default=0, description="Index of currently executing task")

    # Context and metadata
    original_query: str = Field(..., description="Original user question")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous conversation messages")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="Additional user context")

    # Execution tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall pipeline status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Pipeline creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Pipeline start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Pipeline completion timestamp")
    total_execution_time_ms: Optional[float] = Field(None, description="Total pipeline execution time")

    # Results
    final_result: Optional[Dict[str, Any]] = Field(None, description="Final pipeline result")
    intermediate_results: Dict[str, Any] = Field(default_factory=dict, description="Intermediate task results")


class IntentAnalysis(BaseModel):
    """
    Result of intent analysis for user queries.
    """
    intent_type: str = Field(..., description="Primary intent classification")
    requires_conversation: bool = Field(default=False, description="Whether conversation context is needed")
    requires_database: bool = Field(default=False, description="Whether database access is needed")
    requires_vectorial: bool = Field(default=True, description="Whether vector search is needed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for classification")
    entities: List[str] = Field(default_factory=list, description="Extracted entities from query")
    reasoning: Optional[str] = Field(None, description="Explanation of the classification")


class TaskExecutionResult(BaseModel):
    """
    Result of individual task execution.
    """
    model_config = ConfigDict(use_enum_values=True)

    task_action: TaskAction = Field(..., description="The action that was executed")
    status: TaskStatus = Field(..., description="Execution status")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")


class PipelineExecutionSummary(BaseModel):
    """
    Summary of complete pipeline execution.
    """
    pipeline_id: str = Field(..., description="Pipeline identifier")
    user_id: str = Field(..., description="User identifier")
    original_query: str = Field(..., description="Original user question")

    # Execution summary
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(..., description="Number of successfully completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")

    # Timing
    total_execution_time_ms: float = Field(..., description="Total execution time")
    average_task_time_ms: float = Field(..., description="Average time per task")

    # Results
    final_status: TaskStatus = Field(..., description="Final pipeline status")
    task_results: List[TaskExecutionResult] = Field(..., description="Individual task results")
    final_output: Optional[str] = Field(None, description="Final response text")

    # Analytics
    model_used: str = Field(..., description="Primary LLM model used for final response")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")

    created_at: datetime = Field(..., description="Pipeline creation time")
    completed_at: Optional[datetime] = Field(None, description="Pipeline completion time")

    model_config = ConfigDict(use_enum_values=True)


class TaskValidationResult(BaseModel):
    """
    Result of task chain validation.
    """
    valid: bool = Field(..., description="Whether the task chain is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggested_fixes: List[str] = Field(default_factory=list, description="Suggested fixes for issues")


# Task combination matrix as defined in the implementation plan
VALID_TASK_COMBINATIONS = {
    # Conversation, Database, Vectorial -> Task Chain
    (False, False, True): [TaskAction.EMBEDDING, TaskAction.RETRIEVAL, TaskAction.LLM_RESPONSE],
    (False, True, False): [TaskAction.SQL_SELECT, TaskAction.LLM_RESPONSE],
    (False, True, True): [TaskAction.EMBEDDING, TaskAction.RETRIEVAL, TaskAction.SQL_SELECT, TaskAction.LLM_RESPONSE],
    (True, False, False): [TaskAction.GET_CONTEXT, TaskAction.LLM_RESPONSE],
    (True, False, True): [TaskAction.GET_CONTEXT, TaskAction.EMBEDDING, TaskAction.RETRIEVAL, TaskAction.LLM_RESPONSE],
    (True, True, False): [TaskAction.GET_CONTEXT, TaskAction.SQL_SELECT, TaskAction.LLM_RESPONSE],
    (True, True, True): [TaskAction.GET_CONTEXT, TaskAction.EMBEDDING, TaskAction.RETRIEVAL, TaskAction.SQL_SELECT, TaskAction.LLM_RESPONSE],
}


def get_expected_task_chain(requires_conversation: bool, requires_database: bool, requires_vectorial: bool) -> List[TaskAction]:
    """
    Get the expected task chain based on requirements.

    Args:
        requires_conversation: Whether conversation context is needed
        requires_database: Whether database access is needed
        requires_vectorial: Whether vector search is needed

    Returns:
        List of expected task actions in order
    """
    key = (requires_conversation, requires_database, requires_vectorial)
    return VALID_TASK_COMBINATIONS.get(key, [TaskAction.EMBEDDING, TaskAction.RETRIEVAL, TaskAction.LLM_RESPONSE])


def validate_task_order(tasks: List[Task]) -> TaskValidationResult:
    """
    Validate that tasks are in the correct order according to the implementation plan.

    Args:
        tasks: List of tasks to validate

    Returns:
        Validation result with errors and warnings
    """
    errors = []
    warnings = []

    if not tasks:
        errors.append("Task list is empty")
        return TaskValidationResult(valid=False, errors=errors)

    # Check that last task is LLM_RESPONSE
    if tasks[-1].action != TaskAction.LLM_RESPONSE:
        errors.append("Task chain must end with LLM_RESPONSE")

    # Check task order rules
    actions = [task.action for task in tasks]

    # If RETRIEVAL exists, EMBEDDING must come before it
    if TaskAction.RETRIEVAL in actions:
        retrieval_idx = actions.index(TaskAction.RETRIEVAL)
        embedding_indices = [i for i, action in enumerate(actions) if action == TaskAction.EMBEDDING]

        if not embedding_indices or max(embedding_indices) >= retrieval_idx:
            errors.append("EMBEDDING task must come before RETRIEVAL task")

    # GET_CONTEXT should be first if present
    if TaskAction.GET_CONTEXT in actions:
        if actions[0] != TaskAction.GET_CONTEXT:
            warnings.append("GET_CONTEXT should typically be the first task")

    # Check for required fields based on action type
    for i, task in enumerate(tasks):
        if task.action == TaskAction.EMBEDDING and not task.input:
            errors.append(f"EMBEDDING task at index {i} missing required 'input' field")

        if task.action == TaskAction.RETRIEVAL and not task.vector_source:
            errors.append(f"RETRIEVAL task at index {i} missing required 'vector_source' field")

        if task.action == TaskAction.SQL_SELECT and not task.query:
            errors.append(f"SQL_SELECT task at index {i} missing required 'query' field")

    return TaskValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )