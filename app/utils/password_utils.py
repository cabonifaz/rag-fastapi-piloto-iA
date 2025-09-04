import hashlib
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class PasswordUtils:
    """
    Utility class for password hashing and verification.
    Centralized to allow easy changes to encryption methods.
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using SHA256.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Hashed password as hexadecimal string
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def hash_password_binary(password: str) -> bytes:
        """
        Hash password using SHA256, return as binary.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Hashed password as binary data
        """
        return hashlib.sha256(password.encode()).digest()
    
    
    @staticmethod
    def verify_password(stored_password: str, provided_password: str) -> bool:
        """
        Verify password against SHA256 hash stored as binary data in SQL Server.
        
        Args:
            stored_password: Password hash stored in database (binary data as Unicode string)
            provided_password: Password provided by user
            
        Returns:
            True if passwords match, False otherwise
        """
        try:
            # Convert Unicode characters back to bytes (SQL Server VARBINARY -> Unicode conversion)
            byte_values = []
            for char in stored_password:
                byte_values.append(ord(char) & 0xFF)
            stored_bytes = bytes(byte_values)
            
            # Only support SHA256 (32 bytes)
            if len(stored_bytes) == 32:
                # Generate SHA256 hash of provided password
                hashed_provided_binary = PasswordUtils.hash_password_binary(provided_password)
                
                # Compare binary hashes
                return stored_bytes == hashed_provided_binary
            
            # If not 32 bytes, it's not our expected SHA256 format
            return False
                
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def get_password_strength(password: str) -> Tuple[bool, str]:
        """
        Basic password strength validation.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_strong, message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        if not (has_upper and has_lower and has_digit):
            return False, "Password must contain uppercase, lowercase, and numbers"
        
        if not has_special:
            return False, "Password should contain special characters"
        
        return True, "Password is strong"