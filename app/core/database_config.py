from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # SQL Server Database Configuration
    db_server: str
    db_database: str
    db_username: str
    db_password: str
    db_driver: str = "ODBC Driver 17 for SQL Server"
    db_port: int = 1433
    db_trusted_connection: bool = False
    
    @field_validator('db_server')
    @classmethod
    def validate_db_server(cls, v):
        if not v:
            raise ValueError("Database server is required")
        return v

    @field_validator('db_database')
    @classmethod
    def validate_db_database(cls, v):
        if not v:
            raise ValueError("Database name is required")
        return v

    @field_validator('db_username')
    @classmethod
    def validate_db_username(cls, v):
        if not v:
            raise ValueError("Database username is required")
        return v

    @field_validator('db_password')
    @classmethod
    def validate_db_password(cls, v):
        if not v:
            raise ValueError("Database password is required")
        return v
    
    @property
    def database_url(self) -> str:
        """Generate SQL Server connection string"""
        if self.db_trusted_connection:
            return f"mssql+pyodbc://@{self.db_server}:{self.db_port}/{self.db_database}?driver={self.db_driver.replace(' ', '+')}&trusted_connection=yes"
        else:
            return f"mssql+pyodbc://{self.db_username}:{self.db_password}@{self.db_server}:{self.db_port}/{self.db_database}?driver={self.db_driver.replace(' ', '+')}"

# Create database configuration instance
try:
    database_config = DatabaseConfig()
    logger.info("Database configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load database configuration: {e}")
    raise RuntimeError(f"Database configuration validation failed: {str(e)}")