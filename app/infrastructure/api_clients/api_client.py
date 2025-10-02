"""Simple curl-like functions for API calls."""

import subprocess
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)


# Base64 signatures for common file types
BASE64_SIGNATURES = [
    # Imágenes
    r'^iVBORw0KGgo',         # PNG
    r'^/9j/',                # JPEG
    r'^R0lGOD',              # GIF
    r'^Qk0',                 # BMP (empieza con "BM" → Qk base64)
    r'^SUkq',                # TIFF (empieza con "II*" → SUkq base64)

    # Documentos
    r'^JVBERi0',             # PDF
    r'^UEsDB',               # ZIP/DOCX/XLSX/PPTX/ODT/ODS/ODP
    r'^0M8R4KGx',            # Microsoft Compound (DOC, XLS antiguos)

    # Audio
    r'^SUQz',                # MP3 (ID3)
    r'^f0VMRg',              # ELF (binarios ejecutables, raro pero posible)
    r'^T2dnUw',              # OGG
    r'^UklGR',               # WAV

    # Video
    r'^AAAAIGZ0eXBtcDQy',    # MP4
    r'^AAAAGGZ0eXBtcDQy',    # MP4 variante
    r'^Z01pY3Jv',            # WMV/ASF

    # Otros binarios comunes
    r'^UmFyIRo',             # RAR
    r'^7QAAAC4',             # GZIP
    r'^H4sIA',               # GZIP (otra variante base64)
    r'^MThkYXRh',            # Base64 genérico "8data" (a veces usado)
]


def is_base64_content(value: str) -> bool:
    """Check if string matches common base64 file signatures."""
    import re

    # Check for data URI scheme
    if value.startswith('data:'):
        return True

    # Check against known base64 signatures
    for signature in BASE64_SIGNATURES:
        if re.match(signature, value):
            return True

    return False


def filter_response_data(data: Any, max_field_length: int = 250) -> Any:
    """
    Filter response data to remove columns with very long values (like base64 images).

    For lists of objects, removes entire columns if ANY item has a long value in that column.

    Args:
        data: Response data (can be dict, list, or primitive)
        max_field_length: Maximum allowed length for string fields

    Returns:
        Filtered data with long fields removed
    """
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # For list of dicts, identify columns to remove across all items
        columns_to_remove = set()

        # Scan all items to find columns with long values
        for item in data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str):
                        # Detect base64 content or very long strings
                        if len(value) > max_field_length or (len(value) > 100 and is_base64_content(value)):
                            columns_to_remove.add(key)
                            logger.debug(f"Marking column '{key}' for removal (length: {len(value)})")

        # Remove identified columns from all items
        filtered_list = []
        for item in data:
            if isinstance(item, dict):
                filtered_item = {k: v for k, v in item.items() if k not in columns_to_remove}
                filtered_list.append(filtered_item)
            else:
                filtered_list.append(item)

        return filtered_list

    elif isinstance(data, dict):
        filtered = {}
        for key, value in data.items():
            # Check if value is a string and too long (likely base64 or large content)
            if isinstance(value, str):
                # Detect base64 content or very long strings
                if len(value) > max_field_length or (len(value) > 100 and is_base64_content(value)):
                    logger.debug(f"Filtering out field '{key}' (length: {len(value)})")
                    continue
            filtered[key] = filter_response_data(value, max_field_length)
        return filtered

    elif isinstance(data, list):
        return [filter_response_data(item, max_field_length) for item in data]

    else:
        return data


async def httpx_get(url: str, token: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """GET request using httpx with async support."""
    try:
        BASE_URL = "https://bancotalentobackendpreprod-awdecbbsgrh4d8bn.canadacentral-01.azurewebsites.net"

        # Add base URL if not already present
        if not url.startswith("http"):
            full_url = f"{BASE_URL}/{url.lstrip('/')}"
        else:
            full_url = url

        # Build headers
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # Add custom headers
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                full_url,
                params=params,
                headers=request_headers
            )

            response.raise_for_status()

            # Parse JSON response
            parsed_data = response.json()

            # Filter out columns with long values (like base64 images)
            filtered_data = filter_response_data(parsed_data)

            return {"success": True, "data": filtered_data}

    except httpx.HTTPStatusError as e:
        logger.error(f"httpx_get HTTP error: {e.response.status_code} - {e.response.text}")
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except httpx.TimeoutException as e:
        logger.error(f"httpx_get timeout error: {e}")
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        logger.error(f"httpx_get error: {e}")
        return {"success": False, "error": str(e)}


async def httpx_post(url: str, token: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """POST request using httpx with async support."""
    try:
        BASE_URL = "https://bancotalentobackendpreprod-awdecbbsgrh4d8bn.canadacentral-01.azurewebsites.net"

        # Add base URL if not already present
        if not url.startswith("http"):
            full_url = f"{BASE_URL}/{url.lstrip('/')}"
        else:
            full_url = url

        # Build headers
        request_headers = {
            "accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # Add custom headers
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                full_url,
                json=data,
                headers=request_headers
            )

            response.raise_for_status()

            # Parse JSON response
            parsed_data = response.json()

            # Filter out columns with long values (like base64 images)
            filtered_data = filter_response_data(parsed_data)

            return {"success": True, "data": filtered_data}

    except httpx.HTTPStatusError as e:
        logger.error(f"httpx_post HTTP error: {e.response.status_code} - {e.response.text}")
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except httpx.TimeoutException as e:
        logger.error(f"httpx_post timeout error: {e}")
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        logger.error(f"httpx_post error: {e}")
        return {"success": False, "error": str(e)}


async def httpx_login(url: str, username: str, password: str) -> Dict[str, Any]:
    """Login request using httpx."""
    try:
        login_data = {"username": username, "password": password}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=login_data,
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()

            return {"success": True, "data": response.json()}

    except httpx.HTTPStatusError as e:
        logger.error(f"httpx_login HTTP error: {e.response.status_code} - {e.response.text}")
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except httpx.TimeoutException as e:
        logger.error(f"httpx_login timeout error: {e}")
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        logger.error(f"httpx_login error: {e}")
        return {"success": False, "error": str(e)}


async def httpx_test_login(username: str, password: str) -> Dict[str, Any]:
    """Test login function using httpx for specific endpoint."""
    try:
        URL = "https://bancotalentobackendpreprod-awdecbbsgrh4d8bn.canadacentral-01.azurewebsites.net/bdt/auth/login"
        login_data = {"username": username, "password": password}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                URL,
                json=login_data,
                headers={
                    "accept": "*/*",
                    "Content-Type": "application/json"
                }
            )

            response.raise_for_status()

            return {"success": True, "data": response.json()}

    except httpx.HTTPStatusError as e:
        logger.error(f"httpx_test_login HTTP error: {e.response.status_code} - {e.response.text}")
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except httpx.TimeoutException as e:
        logger.error(f"httpx_test_login timeout error: {e}")
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        logger.error(f"httpx_test_login error: {e}")
        return {"success": False, "error": str(e)}