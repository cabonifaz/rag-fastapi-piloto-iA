"""Centralized AWS client management - Singleton pattern for boto3 resources."""

import boto3
import logging
from typing import Optional
from functools import lru_cache
from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSClientManager:
    """Singleton manager for AWS boto3 clients and resources."""

    _instance: Optional['AWSClientManager'] = None
    _session: Optional[boto3.Session] = None
    _dynamodb_resource = None
    _s3_client = None

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize AWS session if not already initialized."""
        if self._session is None:
            self._initialize_session()

    def _initialize_session(self):
        """Create boto3 session with configured credentials."""
        try:
            session_params = {"region_name": settings.aws_region}

            # Use profile in local development, IAM roles in production
            if settings.aws_profile:
                session_params["profile_name"] = settings.aws_profile
            elif settings.aws_access_key_id and settings.aws_secret_access_key:
                session_params["aws_access_key_id"] = settings.aws_access_key_id
                session_params["aws_secret_access_key"] = settings.aws_secret_access_key

            self._session = boto3.Session(**session_params)
            logger.info("AWS Session initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AWS session: {e}")
            raise

    @property
    def session(self) -> boto3.Session:
        """Get the boto3 session."""
        if self._session is None:
            self._initialize_session()
        return self._session

    @property
    def dynamodb(self):
        """Get or create DynamoDB resource (cached)."""
        if self._dynamodb_resource is None:
            self._dynamodb_resource = self.session.resource('dynamodb')
            logger.info("DynamoDB resource created")
        return self._dynamodb_resource

    @property
    def s3_client(self):
        """Get or create S3 client (cached)."""
        if self._s3_client is None:
            self._s3_client = self.session.client('s3')
            logger.info("S3 client created")
        return self._s3_client

    def get_dynamodb_table(self, table_name: str):
        """Get a specific DynamoDB table resource."""
        return self.dynamodb.Table(table_name)


# Create singleton instance
@lru_cache(maxsize=1)
def get_aws_client_manager() -> AWSClientManager:
    """Get the singleton AWS client manager instance."""
    return AWSClientManager()


# Convenience functions for direct access
def get_dynamodb_resource():
    """Get DynamoDB resource."""
    return get_aws_client_manager().dynamodb


def get_s3_client():
    """Get S3 client."""
    return get_aws_client_manager().s3_client


def get_dynamodb_table(table_name: str):
    """Get a specific DynamoDB table."""
    return get_aws_client_manager().get_dynamodb_table(table_name)
