import os
import time
import logging
from typing import Any, Dict, Optional
import requests

from src.lark.auth import TenantAccessTokenManager
from src.lark.lark_base import (
    LarkError,
    LarkConfigError,
    LarkAuthError,
    LarkRateLimitError,
    LarkAPIError,
)

logger = logging.getLogger(__name__)

ALLOWED_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM"}


class LarkBaseWriteClient:
    """
    Dedicated CONTROLLED WRITE CLIENT for Lark Base Bitable API.
    Separated from read-only LarkBaseClient.
    STRICTLY CONSTRAINED TO UPDATE ONLY Daily_Task.Priority.
    """

    def __init__(
        self,
        app_token: Optional[str] = None,
        table_id: Optional[str] = None,
        token_manager: Optional[TenantAccessTokenManager] = None
    ):
        self.app_token = app_token or os.getenv("LARK_BASE_APP_TOKEN") or os.getenv("LARK_APP_TOKEN", "")
        self.table_id = table_id or os.getenv("LARK_DAILY_TASK_TABLE_ID") or os.getenv("LARK_TABLE_ID", "")
        self.token_manager = token_manager or TenantAccessTokenManager()

    def _get_headers(self) -> Dict[str, str]:
        # Retrieve token, supporting both get_tenant_access_token() and get_token().
        try:
            # Prefer explicit tenant token method if available.
            token = self.token_manager.get_tenant_access_token()
        except AttributeError:
            # Fall back to generic token method.
            try:
                token = self.token_manager.get_token()
            except Exception as e:
                raise LarkAuthError(f"Failed to obtain Lark access token via get_token(): {e}") from e
        except LarkConfigError as e:
            # Propagate configuration errors directly.
            raise e
        except Exception as e:
            raise LarkAuthError(f"Failed to obtain Lark access token via get_tenant_access_token(): {e}") from e
        if not token:
            raise LarkAuthError("Lark tenant access token is empty or null.")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }


    def update_ticket_priority(self, record_id: str, priority: str) -> Dict[str, Any]:
        """
        Updates ONLY the Priority field of a Daily_Task Bitable record in Lark Base.
        STRICTLY SCOPED: Refuses to update any other field.
        """
        if not self.app_token or not self.table_id:
            raise LarkConfigError("Missing Lark Base app_token or table_id configuration.")

        if not record_id or not record_id.strip():
            raise LarkAPIError("Cannot update ticket priority without a valid record_id.")

        normalized_priority = priority.strip().upper() if priority else ""
        if normalized_priority not in ALLOWED_PRIORITIES:
            raise LarkAPIError(f"Invalid priority '{priority}'. Allowed values: {sorted(list(ALLOWED_PRIORITIES))}")

        # STRICT PAYLOAD ISOLATION: Payload contains ONLY the Priority field
        payload = {
            "fields": {
                "Priority": normalized_priority
            }
        }

        # DEMO MODE / DRY-RUN CHECK
        demo_mode = os.getenv("DEMO_MODE", "").lower() in ("true", "1") or os.getenv("LARK_WRITE_MODE", "").lower() == "mock"
        if demo_mode:
            logger.info(f"[DEMO MODE / MOCK WRITE] Priority update SIMULATED for record {record_id} Priority -> {normalized_priority}. Payload: {payload}")
            return {
                "success": True,
                "record_id": record_id,
                "updated_priority": normalized_priority,
                "simulated": True,
                "message": "Priority update simulated in DEMO MODE. Lark Base remains UNCHANGED."
            }

        url = f"https://open.larksuite.com/open-apis/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records/{record_id}"
        
        logger.info(f"[LarkBaseWriteClient] Updating record {record_id} Priority -> {normalized_priority}. Payload: {payload}")

        max_retries = 3
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                headers = self._get_headers()
                resp = requests.put(url, json=payload, headers=headers, timeout=10)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == 0:
                        logger.info(f"[LarkBaseWriteClient] Priority update SUCCESS for record {record_id}.")
                        return {
                            "success": True,
                            "record_id": record_id,
                            "updated_priority": normalized_priority,
                            "raw_response": data
                        }
                    else:
                        err_code = data.get("code")
                        err_msg = data.get("msg", "Unknown API error")
                        logger.error(f"[LarkBaseWriteClient] Lark API Error ({err_code}): {err_msg}")
                        raise LarkAPIError(f"Lark API error ({err_code}): {err_msg}", status_code=200, error_code=err_code)

                elif resp.status_code in (401, 403):
                    logger.error(f"[LarkBaseWriteClient] Auth failure HTTP {resp.status_code}.")
                    raise LarkAuthError(f"Authentication failure (HTTP {resp.status_code}).")
                elif resp.status_code == 404:
                    logger.error(f"[LarkBaseWriteClient] Record or table not found HTTP 404.")
                    raise LarkAPIError(f"Record '{record_id}' or table not found (HTTP 404).", status_code=404)
                elif resp.status_code == 429:
                    if attempt < max_retries:
                        logger.warning(f"[LarkBaseWriteClient] Rate limit 429 encountered. Retrying attempt {attempt+1}/{max_retries}...")
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LarkRateLimitError("Rate limit exceeded (HTTP 429).")
                elif resp.status_code >= 500:
                    if attempt < max_retries:
                        logger.warning(f"[LarkBaseWriteClient] Server error HTTP {resp.status_code}. Retrying attempt {attempt+1}/{max_retries}...")
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LarkAPIError(f"Lark server error (HTTP {resp.status_code}).", status_code=resp.status_code)
                else:
                    raise LarkAPIError(f"Lark HTTP error {resp.status_code}: {resp.text}", status_code=resp.status_code)

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    logger.warning(f"[LarkBaseWriteClient] Network timeout/connection error: {e}. Retrying {attempt+1}/{max_retries}...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise LarkAPIError(f"Network error during Lark update: {e}")

        raise LarkAPIError(f"Failed priority update for record {record_id} after {max_retries} attempts.")
