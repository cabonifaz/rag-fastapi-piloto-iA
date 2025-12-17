"""AWS S3 implementation of blob storage port."""

import aioboto3
import logging
from typing import List, Dict, Any, Optional
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.blob_storage_port import BlobStoragePort

logger = logging.getLogger(__name__)


class S3BlobStorage(BlobStoragePort):
    """
    AWS S3 implementation of blob storage operations.
    Uses aioboto3 for async S3 operations.
    """

    def __init__(
        self,
        region: str,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize S3 blob storage client using aioboto3 (async).

        Args:
            region: AWS region (e.g., "us-east-1")
            profile_name: AWS profile name (optional if using IAM Role)
            aws_access_key_id: AWS access key ID (optional, used if no profile)
            aws_secret_access_key: AWS secret access key (optional, used if no profile)
        """
        session_params = {"region_name": region}

        # Use profile in development, IAM roles in production
        if profile_name:
            session_params["profile_name"] = profile_name
        elif aws_access_key_id and aws_secret_access_key:
            session_params["aws_access_key_id"] = aws_access_key_id
            session_params["aws_secret_access_key"] = aws_secret_access_key

        # Create aioboto3 session (client created per operation)
        self.session = aioboto3.Session(**session_params)
        self.region = region
        # Use Signature Version 4 for presigned URLs
        self.s3_config = Config(signature_version='s3v4')

    async def generate_presigned_upload_url(
        self,
        bucket_name: str,
        object_key: str,
        expiration_seconds: int = 300
    ) -> str:
        """
        Generate a presigned URL for uploading an object to S3.

        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key/path
            expiration_seconds: URL expiration time in seconds (default: 300 = 5 minutes)

        Returns:
            Presigned upload URL (PUT method)

        Raises:
            ConnectionError: If unable to connect to S3
            ValueError: If parameters are invalid
        """
        try:
            if not bucket_name or not bucket_name.strip():
                raise ValueError("Bucket name cannot be empty")
            if not object_key or not object_key.strip():
                raise ValueError("Object key cannot be empty")
            if expiration_seconds <= 0:
                raise ValueError("Expiration seconds must be positive")

            async with self.session.client('s3', config=self.s3_config) as s3_client:
                presigned_url = await s3_client.generate_presigned_url(
                    'put_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': object_key,
                    },
                    ExpiresIn=expiration_seconds,
                    HttpMethod='PUT'
                )

            logger.debug(f"Generated presigned URL for s3://{bucket_name}/{object_key}")
            return presigned_url

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError generating presigned URL: {error_code} - {e}")
            raise ConnectionError(f"S3 error generating presigned URL: {error_code}")

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error: {e}")
            raise ConnectionError("Unable to connect to AWS S3 service")

        except ValueError:
            raise  # Re-raise validation errors

        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {e}")
            raise ConnectionError(f"S3 presigned URL generation error: {str(e)}")

    async def move_to_deleted_prefix(
        self,
        bucket_name: str,
        object_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Logically delete S3 objects by copying them to 'deleted/' prefix and deleting originals.

        This is a soft delete operation that preserves the objects by moving them to a
        'deleted/' prefix before removing the original.

        Args:
            bucket_name: S3 bucket name
            object_keys: List of S3 object keys to move to deleted prefix

        Returns:
            Dictionary with:
            - success: bool - True if all operations succeeded
            - moved: List of keys that were successfully moved
            - errors: List of error messages for failed operations
            - skipped: List of keys that were skipped (already in deleted/ or don't exist)
        """
        moved = []
        errors = []
        skipped = []

        if not object_keys:
            return {
                "success": True,
                "moved": moved,
                "errors": errors,
                "skipped": skipped
            }

        try:
            async with self.session.client('s3', config=self.s3_config) as s3_client:
                for s3_key in object_keys:
                    if not s3_key or s3_key.strip() == "":
                        skipped.append("Empty key")
                        continue

                    # Skip if already in deleted/ prefix
                    if s3_key.startswith("deleted/"):
                        skipped.append(s3_key)
                        logger.debug(f"Key {s3_key} already in deleted/ prefix, skipping")
                        continue

                    try:
                        # Check if object exists first
                        try:
                            await s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        except ClientError as e:
                            error_code = e.response['Error']['Code']
                            if error_code == '404':
                                logger.warning(f"Object {s3_key} not found in bucket {bucket_name}, skipping")
                                skipped.append(f"{s3_key} (not found)")
                                continue
                            else:
                                raise

                        # Define destination key with 'deleted/' prefix
                        destination_key = f"deleted/{s3_key}"

                        # Copy object to new location with deleted/ prefix
                        copy_source = {'Bucket': bucket_name, 'Key': s3_key}
                        await s3_client.copy_object(
                            CopySource=copy_source,
                            Bucket=bucket_name,
                            Key=destination_key
                        )

                        logger.debug(f"Copied {s3_key} to {destination_key}")

                        # Delete original object
                        await s3_client.delete_object(Bucket=bucket_name, Key=s3_key)

                        logger.info(f"Moved {s3_key} to {destination_key} in bucket {bucket_name}")
                        moved.append(s3_key)

                    except Exception as e:
                        error_msg = f"Failed to move {s3_key}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error: {e}")
            raise ConnectionError("Unable to connect to AWS S3 service")

        except Exception as e:
            logger.error(f"Unexpected error in move_to_deleted_prefix: {e}")
            raise ConnectionError(f"S3 deletion error: {str(e)}")

        success = len(errors) == 0
        return {
            "success": success,
            "moved": moved,
            "errors": errors,
            "skipped": skipped
        }

    async def check_objects_exist(
        self,
        bucket_name: str,
        object_keys: List[str]
    ) -> Dict[str, bool]:
        """
        Check if multiple S3 objects exist in a bucket.

        Args:
            bucket_name: S3 bucket name
            object_keys: List of S3 object keys to check

        Returns:
            Dictionary mapping each key to boolean (True if exists, False otherwise)
        """
        existence_map = {}

        if not object_keys:
            return existence_map

        try:
            async with self.session.client('s3', config=self.s3_config) as s3_client:
                for s3_key in object_keys:
                    if not s3_key or s3_key.strip() == "":
                        existence_map[s3_key] = False
                        continue

                    try:
                        await s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        existence_map[s3_key] = True
                    except ClientError as e:
                        error_code = e.response['Error']['Code']
                        if error_code == '404':
                            existence_map[s3_key] = False
                        else:
                            logger.error(f"Error checking existence of {s3_key}: {e}")
                            existence_map[s3_key] = False
                    except Exception as e:
                        logger.error(f"Unexpected error checking existence of {s3_key}: {e}")
                        existence_map[s3_key] = False

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error: {e}")
            raise ConnectionError("Unable to connect to AWS S3 service")

        except Exception as e:
            logger.error(f"Unexpected error in check_objects_exist: {e}")
            raise ConnectionError(f"S3 existence check error: {str(e)}")

        return existence_map

    async def delete_objects_for_records(
        self,
        records: List[Dict[str, Any]],
        key_fields: List[str],
        bucket_name: str
    ) -> Dict[str, Any]:
        """
        Delete S3 objects for given records based on specified key fields.

        This function extracts S3 keys from the specified fields in the records
        and moves them to the 'deleted/' prefix.

        Args:
            records: List of record dictionaries
            key_fields: List of field names containing S3 keys (e.g., ['RUTA_DOCUMENTO', 'RUTA_EXTRACCION'])
            bucket_name: S3 bucket name

        Returns:
            Dictionary with deletion results (same format as move_to_deleted_prefix)
        """
        s3_keys_to_delete = []

        # Extract S3 keys from specified fields
        for record in records:
            for field in key_fields:
                s3_key = record.get(field)
                if s3_key and s3_key.strip() != "":
                    s3_keys_to_delete.append(s3_key)

        # Remove duplicates
        s3_keys_to_delete = list(set(s3_keys_to_delete))

        if not s3_keys_to_delete:
            return {
                "success": True,
                "moved": [],
                "errors": [],
                "skipped": []
            }

        # Perform logical deletion
        result = await self.move_to_deleted_prefix(bucket_name, s3_keys_to_delete)

        return result
