"""
Structured Exception Hierarchy for Lark Suite API Operations.
"""
from typing import Optional


class LarkError(Exception):
    """Base exception for Lark operations."""
    pass


class LarkConfigError(LarkError, RuntimeError):
    """Raised when required configuration (app_token, table_id, credentials) is missing or invalid."""
    pass


class LarkAuthError(LarkError, RuntimeError):
    """Raised when authentication/authorization fails (401, 403, invalid token)."""
    pass


class LarkRateLimitError(LarkError):
    """Raised when Lark API rate limit (429) is encountered."""
    pass


class LarkAPIError(LarkError):
    """Raised when Lark API call fails or returns error code."""
    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
