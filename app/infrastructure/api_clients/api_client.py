"""Simple curl-like functions for API calls."""

import subprocess
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List

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


async def curl_get(url: str, token: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Simple GET request using curl with -X GET and --data-urlencode."""
    try:
        # Define variables like in bash
        TOKEN = token
        BASE_URL = "https://bancotalentobackendpreprod-awdecbbsgrh4d8bn.canadacentral-01.azurewebsites.net"
        # Add base URL if not already present
        if not url.startswith("http"):
            URL = f"{BASE_URL}/{url.lstrip('/')}"
        else:
            URL = url

        cmd = ["curl", "-s", "-X", "GET"]

        # Add default headers
        cmd.extend(["-H", "Accept: application/json"])

        # Add Authorization header with token
        cmd.extend(["-H", f"Authorization: Bearer {TOKEN}"])

        # Add custom headers
        if headers:
            for key, value in headers.items():
                cmd.extend(["-H", f"{key}: {value}"])

        # Add URL with query parameters for GET requests
        if params:
            from urllib.parse import urlencode
            query_string = urlencode(params)
            full_url = f"{URL}?{query_string}"
            cmd.append(full_url)
        else:
            cmd.append(URL)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        result_stdout = stdout.decode('utf-8') if stdout else ""
        result_stderr = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            try:
                parsed_data = json.loads(result_stdout)
                # Filter out columns with long values (like base64 images)
                filtered_data = filter_response_data(parsed_data)
                return {"success": True, "data": filtered_data}
            except json.JSONDecodeError:
                return {"success": True, "data": result_stdout}
        else:
            return {"success": False, "error": result_stderr}

    except Exception as e:
        logger.error(f"curl_get error: {e}")
        return {"success": False, "error": str(e)}


async def curl_post(url: str, token: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Simple POST request using curl with -d for JSON data."""
    try:
        # Define variables like in bash
        TOKEN = token
        BASE_URL = "https://bancotalentobackendpreprod-awdecbbsgrh4d8bn.canadacentral-01.azurewebsites.net"
        # Add base URL if not already present
        if not url.startswith("http"):
            URL = f"{BASE_URL}/{url.lstrip('/')}"
        else:
            URL = url

        cmd = ["curl", "-s", "-X", "POST"]

        # Add URL first (before headers)
        cmd.append(URL)

        # Add default headers
        cmd.extend(["-H", "accept: */*"])
        cmd.extend(["-H", "Content-Type: application/json"])

        # Add Authorization header with token
        cmd.extend(["-H", f"Authorization: Bearer {TOKEN}"])

        # Add custom headers
        if headers:
            for key, value in headers.items():
                cmd.extend(["-H", f"{key}: {value}"])

        # Add data with -d at the end
        if data:
            cmd.extend(["-d", json.dumps(data)])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        result_stdout = stdout.decode('utf-8') if stdout else ""
        result_stderr = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            try:
                parsed_data = json.loads(result_stdout)
                # Filter out columns with long values (like base64 images)
                filtered_data = filter_response_data(parsed_data)
                return {"success": True, "data": filtered_data}
            except json.JSONDecodeError:
                return {"success": True, "data": result_stdout}
        else:
            return {"success": False, "error": result_stderr}

    except Exception as e:
        logger.error(f"curl_post error: {e}")
        return {"success": False, "error": str(e)}


async def curl_login(url: str, username: str, password: str) -> Dict[str, Any]:
    """Simple login request using curl."""
    try:
        login_data = {"username": username, "password": password}

        cmd = [
            "curl", "-s", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(login_data),
            url
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        result_stdout = stdout.decode('utf-8') if stdout else ""
        result_stderr = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            try:
                return {"success": True, "data": json.loads(result_stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": result_stdout}
        else:
            return {"success": False, "error": result_stderr}

    except Exception as e:
        logger.error(f"curl_login error: {e}")
        return {"success": False, "error": str(e)}


async def curl_test_login(username: str, password: str) -> Dict[str, Any]:
    """Test login function for specific endpoint."""
    try:
        # Define variables like in bash
        URL = "https://bancotalentobackendpreprod-awdecbbsgrh4d8bn.canadacentral-01.azurewebsites.net/bdt/auth/login"

        cmd = ["curl", "-s", "-X", "POST"]

        # Add headers exactly as specified
        cmd.extend(["-H", "accept: */*"])
        cmd.extend(["-H", "Content-Type: application/json"])

        # Add URL
        cmd.append(URL)

        # Add login data with -d
        login_data = {"username": username, "password": password}
        cmd.extend(["-d", json.dumps(login_data)])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        result_stdout = stdout.decode('utf-8') if stdout else ""
        result_stderr = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            try:
                return {"success": True, "data": json.loads(result_stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": result_stdout}
        else:
            return {"success": False, "error": result_stderr}

    except Exception as e:
        logger.error(f"curl_test_login error: {e}")
        return {"success": False, "error": str(e)}