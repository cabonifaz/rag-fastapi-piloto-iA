"""Simple curl-like functions for API calls."""

import subprocess
import asyncio
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


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
                return {"success": True, "data": json.loads(result_stdout)}
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

        # Add default headers
        cmd.extend(["-H", "Content-Type: application/json"])
        cmd.extend(["-H", "Accept: application/json"])

        # Add Authorization header with token
        cmd.extend(["-H", f"Authorization: Bearer {TOKEN}"])

        # Add custom headers
        if headers:
            for key, value in headers.items():
                cmd.extend(["-H", f"{key}: {value}"])

        # Add data with -d
        if data:
            cmd.extend(["-d", json.dumps(data)])

        # Add URL
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
                return {"success": True, "data": json.loads(result_stdout)}
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