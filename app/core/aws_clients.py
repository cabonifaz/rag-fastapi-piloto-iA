"""Centralized async AWS client management using aioboto3."""

import aioboto3
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from botocore.config import Config
from app.core.config import settings

logger = logging.getLogger(__name__)


class AsyncAWSClientManager:
    """Singleton manager for async AWS clients using aioboto3."""

    _instance: Optional['AsyncAWSClientManager'] = None
    _session: Optional[aioboto3.Session] = None

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize async AWS session if not already initialized."""
        if self._session is None:
            self._initialize_session()

    def _initialize_session(self):
        """Create aioboto3 session with configured credentials."""
        try:
            session_params = {"region_name": settings.aws_region}

            # Use profile in local development, IAM roles in production
            if settings.aws_profile:
                session_params["profile_name"] = settings.aws_profile
            elif settings.aws_access_key_id and settings.aws_secret_access_key:
                session_params["aws_access_key_id"] = settings.aws_access_key_id
                session_params["aws_secret_access_key"] = settings.aws_secret_access_key

            self._session = aioboto3.Session(**session_params)
            logger.info("Async AWS Session initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize async AWS session: {e}")
            raise

    @property
    def session(self) -> aioboto3.Session:
        """Get the aioboto3 session."""
        if self._session is None:
            self._initialize_session()
        return self._session

    @asynccontextmanager
    async def get_s3_client(self):
        """Get S3 client as async context manager."""
        # Use Signature Version 4 for presigned URLs
        config = Config(signature_version='s3v4')
        async with self.session.client('s3', config=config) as client:
            logger.debug("S3 client created with s3v4 signature")
            yield client

    @asynccontextmanager
    async def get_dynamodb_resource(self):
        """Get DynamoDB resource as async context manager."""
        async with self.session.resource('dynamodb') as resource:
            logger.debug("DynamoDB resource created")
            yield resource

    @asynccontextmanager
    async def get_bedrock_client(self):
        """Get Bedrock runtime client as async context manager."""
        async with self.session.client('bedrock-runtime') as client:
            logger.debug("Bedrock runtime client created")
            yield client

    async def get_dynamodb_table(self, table_name: str):
        """Get a specific DynamoDB table resource.

        WARNING: Returns a table object that must be used within the
        resource context. Use get_dynamodb_resource() and access
        table from there in async code.

        Args:
            table_name: Name of the DynamoDB table

        Returns:
            DynamoDB table resource
        """
        # This is a helper that assumes resource context exists
        # Better pattern: use get_dynamodb_resource() and access table from it
        async with self.get_dynamodb_resource() as dynamodb:
            return await dynamodb.Table(table_name)


# Create singleton instance
@lru_cache(maxsize=1)
def get_aws_client_manager() -> AsyncAWSClientManager:
    """Get the singleton async AWS client manager instance."""
    return AsyncAWSClientManager()


# Async convenience functions for direct access
@asynccontextmanager
async def get_s3_client():
    """Get S3 client as async context manager.

    Usage:
        async with get_s3_client() as client:
            response = await client.list_objects(Bucket='...')
    """
    manager = get_aws_client_manager()
    async with manager.get_s3_client() as client:
        yield client


@asynccontextmanager
async def get_dynamodb_resource():
    """Get DynamoDB resource as async context manager.

    Usage:
        async with get_dynamodb_resource() as dynamodb:
            table = dynamodb.Table('table_name')
            response = await table.scan()
    """
    manager = get_aws_client_manager()
    async with manager.get_dynamodb_resource() as resource:
        yield resource


@asynccontextmanager
async def get_dynamodb_table(table_name: str):
    """Get a specific DynamoDB table as async context manager.

    Usage:
        async with get_dynamodb_table('table_name') as table:
            response = await table.get_item(Key={'id': '123'})

    Args:
        table_name: Name of the DynamoDB table
    """
    manager = get_aws_client_manager()
    async with manager.get_dynamodb_resource() as dynamodb:
        table = await dynamodb.Table(table_name)
        yield table


@asynccontextmanager
async def get_bedrock_client():
    """Get Bedrock runtime client as async context manager.

    Usage:
        async with get_bedrock_client() as client:
            response = await client.invoke_model(...)
    """
    manager = get_aws_client_manager()
    async with manager.get_bedrock_client() as client:
        yield client
