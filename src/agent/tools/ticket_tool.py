import logging
from typing import Any, Dict, List, Optional
from src.lark.lark_base import TicketService, Ticket, LarkError

logger = logging.getLogger(__name__)


class TicketToolAdapter:
    """
    Adapter exposing TicketService operations as LangGraph-compatible tools.
    Provides structured ticket data retrieval from Lark Base Daily_Task table.
    """

    def __init__(self, ticket_service: Optional[TicketService] = None):
        self.service = ticket_service or TicketService()

    def get_tickets(self, page_token: Optional[str] = None, page_size: int = 100) -> Dict[str, Any]:
        try:
            tickets, next_token, has_more = self.service.get_tickets(page_token=page_token, page_size=page_size)
            return {
                "success": True,
                "tickets": [t.to_dict() for t in tickets],
                "next_page_token": next_token,
                "has_more": has_more,
                "count": len(tickets)
            }
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_tickets failed: {e}")
            return {"success": False, "error": str(e), "tickets": []}

    def get_all_tickets(self, max_pages: int = 10) -> Dict[str, Any]:
        try:
            tickets = self.service.get_all_tickets(max_pages=max_pages)
            return {
                "success": True,
                "tickets": [t.to_dict() for t in tickets],
                "count": len(tickets)
            }
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_all_tickets failed: {e}")
            return {"success": False, "error": str(e), "tickets": []}

    def get_ticket_by_id(self, ticket_id: str) -> Dict[str, Any]:
        try:
            ticket = self.service.get_ticket_by_id(ticket_id)
            if ticket:
                return {"success": True, "ticket": ticket.to_dict(), "found": True}
            return {"success": True, "ticket": None, "found": False, "message": f"Ticket '{ticket_id}' not found"}
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_ticket_by_id failed for '{ticket_id}': {e}")
            return {"success": False, "error": str(e), "ticket": None, "found": False}

    def get_tickets_by_sprint(self, sprint: str) -> Dict[str, Any]:
        try:
            tickets = self.service.get_tickets_by_sprint(sprint)
            return {"success": True, "tickets": [t.to_dict() for t in tickets], "sprint": sprint, "count": len(tickets)}
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_tickets_by_sprint failed for '{sprint}': {e}")
            return {"success": False, "error": str(e), "tickets": []}

    def get_tickets_by_project(self, project: str) -> Dict[str, Any]:
        try:
            tickets = self.service.get_tickets_by_project(project)
            return {"success": True, "tickets": [t.to_dict() for t in tickets], "project": project, "count": len(tickets)}
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_tickets_by_project failed for '{project}': {e}")
            return {"success": False, "error": str(e), "tickets": []}

    def get_tickets_by_team(self, team: str) -> Dict[str, Any]:
        try:
            tickets = self.service.get_tickets_by_team(team)
            return {"success": True, "tickets": [t.to_dict() for t in tickets], "team": team, "count": len(tickets)}
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_tickets_by_team failed for '{team}': {e}")
            return {"success": False, "error": str(e), "tickets": []}

    def get_tickets_by_status(self, status: str) -> Dict[str, Any]:
        try:
            tickets = self.service.get_tickets_by_status(status)
            return {"success": True, "tickets": [t.to_dict() for t in tickets], "status": status, "count": len(tickets)}
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_tickets_by_status failed for '{status}': {e}")
            return {"success": False, "error": str(e), "tickets": []}

    def get_related_tickets(self, ticket_id: str) -> Dict[str, Any]:
        try:
            related = self.service.get_related_tickets(ticket_id)
            return {"success": True, "related_tickets": [t.to_dict() for t in related], "count": len(related)}
        except LarkError as e:
            logger.error(f"TicketToolAdapter.get_related_tickets failed for '{ticket_id}': {e}")
            return {"success": False, "error": str(e), "related_tickets": []}
