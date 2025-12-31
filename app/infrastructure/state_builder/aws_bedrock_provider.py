import aioboto3
import json
import logging
import os
import asyncio
from typing import Optional, Dict, List
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.core.config import settings
from app.domain.ports.state_builder import StateBuilderPort
from app.infrastructure.state_builder.model_factory import ModelFactory

# Configure logging
logger = logging.getLogger(__name__)


class StateBuilder(StateBuilderPort):
    """
    State builder for RAG that builds optimal query state by analyzing conversation history to:
    1. Resolve pronouns and implicit references
    2. Include necessary context from previous messages
    3. Create standalone, self-contained queries for better RAG retrieval

    Uses AWS Bedrock Converse API (non-streaming version) to build query state.
    Supports multiple models: Claude (Anthropic), Llama (Meta), Nova (Amazon), GPT (OpenAI).
    Model configuration is automatically selected based on model_id.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize AWS Bedrock Converse client using aioboto3 (async) for RAG state building.

        Args:
            region: AWS region (defaults to settings.aws_region)
            model_id: Bedrock model ID (defaults to settings.state_builder_model_id)
            profile_name: AWS profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.state_builder_model_id

        session_params = {"region_name": self.region}

        # Use profile only in development, not in production with IAM roles
        profile = profile_name or settings.aws_profile
        if profile and os.getenv('ENVIRONMENT', settings.environment).lower() != 'production':
            session_params["profile_name"] = profile
        # If no profile, use direct credentials if available
        else:
            access_key = aws_access_key_id or settings.aws_access_key_id
            secret_key = aws_secret_access_key or settings.aws_secret_access_key
            if access_key and secret_key:
                session_params["aws_access_key_id"] = access_key
                session_params["aws_secret_access_key"] = secret_key

        # Configure botocore with connection and read timeouts
        self.boto_config = Config(
            connect_timeout=30,  # 30 seconds to establish connection
            read_timeout=120,    # 2 minutes max for reading response
            retries={'max_attempts': 2, 'mode': 'standard'}  # Retry failed requests
        )

        try:
            # Create aioboto3 session (don't create client yet)
            self.session = aioboto3.Session(**session_params)

            # Log session creation
            session_info = {k: '***' if 'key' in k.lower() or 'secret' in k.lower() else v
                           for k, v in session_params.items()}
            logger.info(f"✨ Created NEW aioboto3.Session (id: {id(self.session)}) [RAG State Builder] | Config: {session_info}")

            # Get model-specific configuration based on model_id
            self.model_config = ModelFactory.get_model_config(self.model_id)

            logger.info(f"RAG State Builder initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG State Builder: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    def _build_system_config(self) -> list:
        """Build system configuration for Converse API."""
        system_prompt = self.model_config.get_system_prompt()
        return [{"text": system_prompt}]

    async def build_query_state(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Builds conversation state by analyzing conversation history with aioboto3 (truly async).
        Extracts topic, entities, and goal from previous user messages.

        Args:
            user_query: The user's query text (not included in state extraction).
            conversation_history: Optional list of recent message dicts with 'role' and 'content'.

        Returns:
            Dictionary with:
                - topic: str (the conversation topic)
                - entities: list[str] (entities mentioned)
                - goal: str (user's goal)
            Returns empty state if no conversation history or if state building fails.
        """
        # Default response if no conversation history
        if not conversation_history or len(conversation_history) == 0:
            logger.info("No conversation history provided, returning empty state")
            return {
                "topic": "",
                "entities": [],
                "goal": ""
            }

        try:
            # Convert conversation history to Converse API messages format
            # (includes user messages with content and assistant messages with empty content)
            converse_messages = []
            for msg in conversation_history:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })

            # Add instruction as the final user message
            instruction = "Analyze the previous user messages and extract the conversation state as JSON."
            converse_messages.append({
                "role": "user",
                "content": [{"text": instruction}]
            })

            # Build request parameters
            request_params = {
                "modelId": self.model_id,
                "messages": converse_messages,
                "system": self._build_system_config(),
                "inferenceConfig": {
                    "maxTokens": 2048,  # Sufficient for state-built queries
                    "temperature": 0.0,  # Low temperature for consistent state building
                    "topP": 0.1
                }
            }

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            logger.info(
                f"♻️ Reusing session (id: {id(self.session)}) [RAG State Builder] | "
                f"Request params: model={self.model_id}, max_tokens=2048, temp=0.0, top_p=0.1"
            )
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse(**request_params)

                # Extract the state-built query result
                result = self._extract_result(response)

            if result:
                # Safely access the result with logging
                logger.info(f"Raw result from parser (type: {type(result)}): {result}")
                logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

                topic = result.get('topic', '') if isinstance(result, dict) else ''
                entities = result.get('entities', []) if isinstance(result, dict) else []
                goal = result.get('goal', '') if isinstance(result, dict) else ''

                logger.info(
                    f"Conversation state extracted:\n"
                    f"  Topic: {topic}\n"
                    f"  Entities: {entities}\n"
                    f"  Goal: {goal}\n"
                    f"  Complete result: {result}"
                )
                return result
            else:
                logger.warning("Failed to build query state, returning empty state")
                return {
                    "topic": "",
                    "entities": [],
                    "goal": ""
                }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in RAG State Builder: {error_code} - {e}")

            if error_code == 'ValidationException':
                logger.error(f"Invalid parameters for model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                logger.error(f"Rate limit exceeded for model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                logger.error(f"Service quota exceeded for model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                logger.error(f"Model {self.model_id} is not ready")
            elif error_code == 'ResourceNotFoundException':
                logger.error(f"Model {self.model_id} not found or not accessible")

            return {"topic": "", "entities": [], "goal": ""}

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in RAG State Builder: {e}")
            return {"topic": "", "entities": [], "goal": ""}

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in RAG State Builder: {e}")
            return {"topic": "", "entities": [], "goal": ""}

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in RAG State Builder: {e}")
            return {"topic": "", "entities": [], "goal": ""}

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in RAG State Builder: {e}")
            else:
                logger.error(f"Unexpected error in RAG State Builder: {e}")

            return {"topic": "", "entities": [], "goal": ""}

    def _extract_result(self, response) -> Optional[Dict[str, any]]:
        """
        Extract the conversation state from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with topic, entities, and goal, or None if extraction fails.
        """
        # Delegate to the model config's extract_response method
        return self.model_config.extract_response(response)
