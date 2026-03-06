"""Port (interface) for blob storage operations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional



class BlobStoragePort(ABC):
    """
    Port (interface) for blob storage services.
    Allows implementing different providers (AWS S3, Azure Blob, GCS, etc.)
    """

    @abstractmethod
    async def generate_presigned_upload_url(
        self,
        bucket_name: str,
        object_key: str,
        expiration_seconds: int = 300
    ) -> str:
        """
        Generate a presigned URL for uploading an object to blob storage.

        Args:
            bucket_name: Name of the storage bucket/container
            object_key: Key/path for the object in storage
            expiration_seconds: URL expiration time in seconds (default: 300 = 5 minutes)

        Returns:
            Presigned upload URL (PUT method)

        Raises:
            ConnectionError: If unable to connect to storage service
            ValueError: If parameters are invalid
        """
        pass
    @abstractmethod
    async def generate_presigned_upload_urls_batch(
        self,
        bucket_name: str,
        object_keys: List[str],
        expiration_seconds: int = 300
    ) -> List[str]:
        """
        Generate presigned PUT URLs for multiple objects using a single client session.

        Args:
            bucket_name: Name of the storage bucket/container
            object_keys: List of object keys to generate URLs for
            expiration_seconds: URL expiration time in seconds (default: 300 = 5 minutes)

        Returns:
            List of presigned upload URLs in the same order as object_keys

        Raises:
            ConnectionError: If unable to connect to storage service
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    async def generate_presigned_download_urls_batch(
        self,
        bucket_name: str,
        object_keys: List[str],
        expiration_seconds: int = 300,
        as_attachment: bool = False
    ) -> List[str]:
        """
        Generate presigned GET URLs for multiple objects using a single client session.

        Args:
            bucket_name: Name of the storage bucket/container
            object_keys: List of object keys to generate URLs for
            expiration_seconds: URL expiration time in seconds (default: 300 = 5 minutes)
            as_attachment: Force download if True

        Returns:
            List of presigned download URLs in the same order as object_keys

        Raises:
            ConnectionError: If unable to connect to storage service
        """
        pass

    @abstractmethod
    async def generate_presigned_download_url(
        self,
        bucket_name: str,
        object_key: str,
        expiration_seconds: int = 300,
        as_attachment: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a presigned URL for downloading or viewing an object.

        Args:
            bucket_name: Name of the storage bucket
            object_key: Object key/path in storage
            expiration_seconds: URL expiration time
            as_attachment: Force download if True
            filename: Optional filename for download

        Returns:
            Presigned GET URL
        """
        pass
    @abstractmethod
    async def move_to_deleted_prefix(
        self,
        bucket_name: str,
        object_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Logically delete objects by copying them to 'deleted/' prefix and deleting originals.

        This is a soft delete operation that preserves the objects by moving them to a
        'deleted/' prefix before removing the original.

        Args:
            bucket_name: Name of the storage bucket/container
            object_keys: List of object keys to move to deleted prefix

        Returns:
            Dictionary with:
            - success: bool - True if all operations succeeded
            - moved: List of keys that were successfully moved
            - errors: List of error messages for failed operations
            - skipped: List of keys that were skipped (already in deleted/ or don't exist)

        Raises:
            ConnectionError: If unable to connect to storage service
        """
        pass

    @abstractmethod
    async def check_objects_exist(
        self,
        bucket_name: str,
        object_keys: List[str]
    ) -> Dict[str, bool]:
        """
        Check if multiple objects exist in storage.

        Args:
            bucket_name: Name of the storage bucket/container
            object_keys: List of object keys to check

        Returns:
            Dictionary mapping each key to boolean (True if exists, False otherwise)

        Raises:
            ConnectionError: If unable to connect to storage service
        """
        pass

    @abstractmethod
    async def delete_objects_for_records(
        self,
        records: List[Dict[str, Any]],
        key_fields: List[str],
        bucket_name: str
    ) -> Dict[str, Any]:
        """
        Delete storage objects for given records based on specified key fields.

        This function extracts object keys from the specified fields in the records
        and moves them to the 'deleted/' prefix.

        Args:
            records: List of record dictionaries
            key_fields: List of field names containing object keys (e.g., ['RUTA_DOCUMENTO', 'RUTA_EXTRACCION'])
            bucket_name: Name of the storage bucket/container

        Returns:
            Dictionary with deletion results (same format as move_to_deleted_prefix)

        Raises:
            ConnectionError: If unable to connect to storage service
        """
        pass
