import os
import re
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.lark.auth import TenantAccessTokenManager

logger = logging.getLogger(__name__)


# Custom Exception Hierarchy for Structured Error Handling
from src.lark.exceptions import (
    LarkError,
    LarkConfigError,
    LarkAuthError,
    LarkRateLimitError,
    LarkAPIError,
)


@dataclass
class Ticket:
    """Normalized representation of a Lark Daily_Task ticket."""
    ticket_id: str
    name: str
    description: str = ""
    start_date: str = ""
    end_date: str = ""
    status: str = "UNKNOWN"
    attachment: List[Any] = field(default_factory=list)
    sub_ticket_ids: List[str] = field(default_factory=list)
    parent_ticket_id: str = ""
    assignee: str = ""
    creator: str = ""
    reporter: str = ""
    size: str = ""
    story_points: Optional[float] = None
    release_note: str = ""
    priority: str = "UNKNOWN"
    team: str = ""
    labels: List[str] = field(default_factory=list)
    sprint: str = ""
    record_id: str = ""
    raw_record: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TicketNormalizer:
    """Normalizes raw Lark Base record into standard internal Ticket object."""

    @staticmethod
    def _extract_string(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            return str(val.get("name") or val.get("text") or val.get("id") or "").strip()
        if isinstance(val, list):
            items = [TicketNormalizer._extract_string(item) for item in val if item is not None]
            return ", ".join([i for i in items if i])
        return str(val).strip()

    @staticmethod
    def _extract_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, list):
            res = []
            for item in val:
                extracted = TicketNormalizer._extract_string(item)
                if extracted:
                    res.append(extracted)
            return res
        if isinstance(val, str):
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return parts
        return [str(val).strip()]

    @staticmethod
    def _normalize_date(val: Any) -> str:
        if not val:
            return ""
        if isinstance(val, (int, float)):
            try:
                # Timestamps in Lark are often milliseconds
                ts = val / 1000.0 if val > 1e11 else val
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return str(val)
        if isinstance(val, str):
            val_str = val.strip()
            # If string is numeric digits
            if val_str.isdigit():
                try:
                    ts = float(val_str) / 1000.0 if float(val_str) > 1e11 else float(val_str)
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    return val_str
            return val_str
        return str(val)

    @staticmethod
    def _normalize_story_points(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val.strip())
            except ValueError:
                return None
        return None

    @classmethod
    def normalize(cls, record: Dict[str, Any]) -> Ticket:
        """Converts raw Lark item into normalized Ticket object."""
        if not isinstance(record, dict):
            record = {}
        
        record_id = record.get("record_id", "")
        fields = record.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}

        ticket_id = cls._extract_string(
            fields.get("ID Ticket") or fields.get("Ticket ID") or fields.get("ticket_id") or fields.get("ID") or record_id or "UNKNOWN"
        )
        if not ticket_id:
            ticket_id = record_id or "UNKNOWN"

        name = cls._extract_string(fields.get("Name") or fields.get("name") or fields.get("Title") or "")
        description = cls._extract_string(fields.get("Description") or fields.get("description") or "")
        
        start_date = cls._normalize_date(fields.get("Start Date") or fields.get("start_date"))
        end_date = cls._normalize_date(fields.get("End Date") or fields.get("end_date"))
        
        raw_status = cls._extract_string(fields.get("Status") or fields.get("status"))
        status = raw_status if raw_status else "UNKNOWN"
        
        attachment_val = fields.get("Attachment") or fields.get("attachment")
        attachment = attachment_val if isinstance(attachment_val, list) else ([attachment_val] if attachment_val else [])
        
        sub_ticket_ids = cls._extract_list(fields.get("Sub-Ticket") or fields.get("Sub Ticket") or fields.get("sub_ticket"))
        parent_ticket_id = cls._extract_string(fields.get("Parent Ticket") or fields.get("parent_ticket"))
        
        assignee = cls._extract_string(fields.get("Assignee") or fields.get("assignee"))
        creator = cls._extract_string(fields.get("Creator/ Reporter") or fields.get("Creator / Reporter") or fields.get("Creator") or fields.get("creator") or fields.get("Reporter"))
        reporter = cls._extract_string(fields.get("Reporter") or fields.get("reporter") or creator)
        
        size = cls._extract_string(fields.get("Size") or fields.get("size"))
        story_points = cls._normalize_story_points(fields.get("Story Point") or fields.get("story_points") or fields.get("Story Points"))
        
        release_note = cls._extract_string(fields.get("Release Note") or fields.get("release_note"))
        
        raw_priority = cls._extract_string(fields.get("Priority") or fields.get("priority"))
        priority = raw_priority if raw_priority else "UNKNOWN"
        
        team = cls._extract_string(fields.get("Team") or fields.get("team"))
        labels = cls._extract_list(fields.get("Label") or fields.get("labels") or fields.get("Tag"))
        sprint = cls._extract_string(fields.get("Sprint") or fields.get("sprint"))

        provenance = {
            "source": "Lark",
            "source_table": "Daily_Task",
            "record_id": record_id,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

        return Ticket(
            ticket_id=ticket_id,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            status=status,
            attachment=attachment,
            sub_ticket_ids=sub_ticket_ids,
            parent_ticket_id=parent_ticket_id,
            assignee=assignee,
            creator=creator,
            reporter=reporter,
            size=size,
            story_points=story_points,
            release_note=release_note,
            priority=priority,
            team=team,
            labels=labels,
            sprint=sprint,
            record_id=record_id,
            raw_record=record,
            provenance=provenance
        )


class LarkBaseClient:
    """Read-only client for Lark Base Bitable API (Daily_Task table)."""

    def __init__(
        self,
        token_manager: Optional[TenantAccessTokenManager] = None,
        app_token: Optional[str] = None,
        table_id: Optional[str] = None,
        timeout: int = 10
    ):
        self.token_manager = token_manager or TenantAccessTokenManager()
        self.app_token = app_token or os.getenv("LARK_BASE_APP_TOKEN") or os.getenv("LARK_APP_TOKEN")
        self.table_id = table_id or os.getenv("LARK_DAILY_TASK_TABLE_ID") or os.getenv("LARK_TABLE_ID")
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        try:
            token = self.token_manager.get_token()
        except Exception as e:
            raise LarkAuthError(f"Failed to obtain Lark access token: {e}") from e

        if not token:
            raise LarkAuthError("Lark tenant access token is empty or null")

        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }

    def get_records(
        self,
        page_token: Optional[str] = None,
        page_size: int = 100
    ) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """
        Fetch records from Daily_Task bitable endpoint.
        Returns: (items, next_page_token, has_more)
        """
        if not self.app_token or not self.table_id:
            raise LarkConfigError(
                "Lark Base app_token and table_id must be configured. "
                "Set LARK_BASE_APP_TOKEN and LARK_DAILY_TASK_TABLE_ID in environment."
            )

        url = f"https://open.larksuite.com/open-apis/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records"
        params: Dict[str, Any] = {"page_size": min(page_size, 100)}
        if page_token:
            params["page_token"] = page_token

        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            raise LarkAPIError(f"Lark API request timed out after {self.timeout}s: {e}") from e
        except requests.exceptions.RequestException as e:
            raise LarkAPIError(f"Network error during Lark Base API request: {e}") from e

        if response.status_code == 401 or response.status_code == 403:
            raise LarkAuthError(f"Lark API authentication/authorization failed (status {response.status_code}): {response.text}")

        if response.status_code == 429:
            raise LarkRateLimitError(f"Lark API rate limit exceeded (status 429): {response.text}")

        if response.status_code != 200:
            raise LarkAPIError(
                f"Lark Base API returned unexpected status code {response.status_code}: {response.text}",
                status_code=response.status_code
            )

        try:
            data = response.json()
        except Exception as e:
            raise LarkAPIError(f"Failed to parse Lark API response JSON: {e}") from e

        code = data.get("code", -1)
        if code != 0:
            msg = data.get("msg", "Unknown error")
            if code in (99991663, 99991664, 99991668):
                raise LarkAuthError(f"Lark API token error: code={code}, msg={msg}")
            raise LarkAPIError(f"Lark Base API error: code={code}, msg={msg}", error_code=code)

        data_obj = data.get("data", {})
        if not isinstance(data_obj, dict):
            raise LarkAPIError("Malformed Lark API response: 'data' is not an object")

        items = data_obj.get("items", [])
        if not isinstance(items, list):
            items = []

        has_more = bool(data_obj.get("has_more", False))
        next_page_token = data_obj.get("page_token") if has_more else None

        return items, next_page_token, has_more


class TicketService:
    """High-level read-only service for querying and managing Daily_Task tickets."""

    def __init__(self, client: Optional[LarkBaseClient] = None):
        self.client = client or LarkBaseClient()

    def get_tickets(
        self,
        page_token: Optional[str] = None,
        page_size: int = 100
    ) -> Tuple[List[Ticket], Optional[str], bool]:
        """Retrieves a single page of normalized tickets."""
        items, next_token, has_more = self.client.get_records(page_token=page_token, page_size=page_size)
        tickets = [TicketNormalizer.normalize(item) for item in items]
        return tickets, next_token, has_more

    def get_all_tickets(self, max_pages: int = 20) -> List[Ticket]:
        """Retrieves all tickets across pages up to max_pages safeguard."""
        all_tickets: List[Ticket] = []
        page_token: Optional[str] = None
        for _ in range(max_pages):
            tickets, next_token, has_more = self.get_tickets(page_token=page_token, page_size=100)
            all_tickets.extend(tickets)
            if not has_more or not next_token:
                break
            page_token = next_token
        return all_tickets

    def get_ticket_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """Finds a ticket by its Ticket ID or record_id."""
        if not ticket_id:
            return None
        target_clean = ticket_id.strip().lower()
        all_tickets = self.get_all_tickets()
        for t in all_tickets:
            if t.ticket_id.lower() == target_clean or t.raw_record.get("record_id", "").lower() == target_clean:
                return t
        return None

    def get_tickets_by_sprint(self, sprint: str) -> List[Ticket]:
        """Filters tickets matching specified sprint, normalizing optional 'Sprint ' prefix."""
        if not sprint:
            return []
        target = sprint.strip().lower()
        clean_target = re.sub(r'^sprint\s+', '', target, flags=re.IGNORECASE).strip()
        all_tickets = self.get_all_tickets()
        return [
            t for t in all_tickets
            if t.sprint.strip().lower() in (target, clean_target)
            or re.sub(r'^sprint\s+', '', t.sprint.strip().lower(), flags=re.IGNORECASE).strip() in (target, clean_target)
        ]

    def get_tickets_by_project(self, project: str) -> List[Ticket]:
        """Filters tickets matching specified project name or identifier."""
        if not project:
            return []
        target = project.strip().lower()
        all_tickets = self.get_all_tickets()
        matching = []
        for t in all_tickets:
            sprint_lower = t.sprint.lower()
            name_lower = t.name.lower()
            tid_lower = t.ticket_id.lower()
            team_lower = t.team.lower()
            if (
                target in sprint_lower
                or target in name_lower
                or target in team_lower
                or any(target in lbl.lower() for lbl in t.labels)
                or (target in ("crm", "australian retirement fund crm", "retirement fund crm", "retirement fund") and tid_lower.startswith("crm-"))
            ):
                matching.append(t)
        return matching

    def get_tickets_by_team(self, team: str) -> List[Ticket]:
        """Filters tickets matching specified team."""
        if not team:
            return []
        target = team.strip().lower()
        all_tickets = self.get_all_tickets()
        return [t for t in all_tickets if t.team.lower() == target]

    def get_tickets_by_status(self, status: str) -> List[Ticket]:
        """Filters tickets matching specified status."""
        if not status:
            return []
        target = status.strip().lower()
        all_tickets = self.get_all_tickets()
        return [t for t in all_tickets if t.status.lower() == target]

    def get_related_tickets(self, ticket_id: str) -> List[Ticket]:
        """Retrieves sub-tickets and parent tickets related to target ticket."""
        target = self.get_ticket_by_id(ticket_id)
        if not target:
            return []

        all_tickets = self.get_all_tickets()
        related: List[Ticket] = []
        
        sub_ids = {s.lower() for s in target.sub_ticket_ids if s}
        parent_id = target.parent_ticket_id.lower() if target.parent_ticket_id else ""

        for t in all_tickets:
            if t.ticket_id == target.ticket_id:
                continue
            t_id = t.ticket_id.lower()
            rec_id = t.raw_record.get("record_id", "").lower()

            # Matches sub-ticket list or parent ticket
            if (t_id in sub_ids) or (rec_id in sub_ids) or (parent_id and (t_id == parent_id or rec_id == parent_id)):
                if t not in related:
                    related.append(t)
            # Reverse check: if t's parent_ticket_id matches target
            elif target.ticket_id and t.parent_ticket_id.lower() == target.ticket_id.lower():
                if t not in related:
                    related.append(t)

        return related

    def validate_ticket(self, ticket: Ticket) -> Dict[str, Any]:
        """
        Performs data quality checks on normalized ticket.
        Returns dict containing 'is_valid', 'errors', and 'warnings'.
        Does NOT invalidate tickets for missing optional fields.
        """
        errors = []
        warnings = []

        if not ticket.ticket_id or ticket.ticket_id == "UNKNOWN":
            errors.append("ticket_id is missing or UNKNOWN")

        if not ticket.name:
            warnings.append("name is empty")

        if ticket.status == "UNKNOWN":
            warnings.append("status is UNKNOWN")

        if ticket.priority == "UNKNOWN":
            warnings.append("priority is UNKNOWN")

        if ticket.story_points is None:
            warnings.append("story_points is missing or non-numeric")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
