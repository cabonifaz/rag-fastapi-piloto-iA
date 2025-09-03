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
    def verify_password(stored_password: str, provided_password: str) -> bool:
        """
        Verify password against stored hash with backward compatibility.
        
        Supports:
        - SHA256 hashed passwords (64 hex characters)
        - Plain text passwords (legacy support)
        
        Args:
            stored_password: Password stored in database
            provided_password: Password provided by user
            
        Returns:
            True if passwords match, False otherwise
        """
        try:
            # Check if stored password looks like SHA256 hash (64 hex characters)
            if PasswordUtils._is_sha256_hash(stored_password):
                # Compare hashed versions
                hashed_provided = PasswordUtils.hash_password(provided_password)
                return stored_password.lower() == hashed_provided.lower()
            else:
                # Legacy plain text comparison
                logger.warning("Plain text password detected - consider migrating to hashed passwords")
                return stored_password == provided_password
                
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def _is_sha256_hash(password: str) -> bool:
        """
        Check if a string looks like a SHA256 hash.
        
        Args:
            password: String to check
            
        Returns:
            True if string appears to be SHA256 hash
        """
        return (
            len(password) == 64 and 
            all(c in '0123456789abcdefABCDEF' for c in password)
        )
    
    @staticmethod
    def needs_rehashing(stored_password: str) -> bool:
        """
        Check if password needs to be rehashed (e.g., plain text to hash).
        
        Args:
            stored_password: Password stored in database
            
        Returns:
            True if password should be rehashed
        """
        return not PasswordUtils._is_sha256_hash(stored_password)
    
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