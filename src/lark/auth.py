import os
import time
import requests
from threading import Lock
from typing import Optional

from src.lark.exceptions import LarkConfigError, LarkAuthError


class TenantAccessTokenManager:
    """
    Thread-safe manager for fetching and auto-refreshing Lark Tenant Access Tokens.
    Uses app_id and app_secret from configuration or environment.
    Provides both get_token() and get_tenant_access_token() interfaces.
    """
    def __init__(self, app_id: Optional[str] = None, app_secret: Optional[str] = None, refresh_margin: int = 60):
        self.app_id = app_id if app_id is not None else os.getenv("LARK_APP_ID", "")
        self.app_secret = app_secret if app_secret is not None else os.getenv("LARK_APP_SECRET", "")
        self.refresh_margin = refresh_margin
        self._token: Optional[str] = None
        self._expiry: float = 0.0
        self._lock = Lock()

    def _refresh_token(self) -> None:
        if not self.app_id or not self.app_secret:
            raise LarkConfigError("LARK_APP_ID and LARK_APP_SECRET must be configured for Lark authentication")

        url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise LarkAuthError(f"Network error while requesting Lark tenant access token: {e}") from e

        if data.get("code", 0) != 0:
            raise LarkAuthError(f"Lark tenant token request failed: code={data.get('code')}, msg={data.get('msg')}")

        self._token = data.get("tenant_access_token")
        expires_in = int(data.get("expire", 7200))
        self._expiry = time.time() + expires_in - self.refresh_margin
        if not self._token:
            raise LarkAuthError("Lark tenant token response did not include tenant_access_token")

    def get_token(self) -> str:
        with self._lock:
            if not self._token or time.time() >= self._expiry:
                self._refresh_token()
            return self._token

    def get_tenant_access_token(self) -> str:
        """
        Retrieves the valid tenant access token (auto-refreshing if expired).
        Interface method expected by LarkBaseWriteClient and external callers.
        """
        return self.get_token()

