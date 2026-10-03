import logging
from typing import Any, Dict
from src.agent.update.update_model import UpdateResult, UpdateResultStatus
from src.agent.approval.approval_model import ActorType

logger = logging.getLogger(__name__)


class UpdateVerifier:
    """
    Verification checks for controlled priority update execution, payload isolation, and post-write verification.
    """

    @classmethod
    def verify_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies that the update payload contains ONLY the Priority field and zero other Daily_Task fields.
        """
        fields = payload.get("fields", {}) if isinstance(payload, dict) else {}
        keys = set(fields.keys())

        # STRICT PAYLOAD ISOLATION: Must contain ONLY 'Priority'
        is_isolated = (keys == {"Priority"})
        forbidden = keys - {"Priority"}

        return {
            "verified": is_isolated,
            "keys": list(keys),
            "forbidden_fields": list(forbidden),
            "reason": "Payload contains ONLY 'Priority' field." if is_isolated else f"Payload isolation violation! Forbidden fields: {forbidden}"
        }

    @classmethod
    def verify_authorization(cls, actor_id: str, actor_type: ActorType) -> Dict[str, Any]:
        """
        Verifies that update authorization originates from a human approver.
        """
        is_human = (actor_type != ActorType.AGENT and actor_id.upper() != "AGENT")
        is_identified = bool(actor_id and actor_id.strip())
        all_ok = is_human and is_identified

        return {
            "verified": all_ok,
            "is_human": is_human,
            "is_identified": is_identified,
            "reason": "Human update authorization verified." if all_ok else "Unauthorized update: Agent cannot execute update without human authorization."
        }

    @classmethod
    def verify_post_write(cls, expected_priority: str, actual_priority: str) -> Dict[str, Any]:
        """
        Verifies post-write equality between approved priority and actual priority read from Lark.
        """
        match = (expected_priority.upper() == actual_priority.upper())
        return {
            "verified": match,
            "expected": expected_priority,
            "actual": actual_priority,
            "reason": "Post-write verification SUCCESS." if match else f"Post-write verification FAILED: Expected {expected_priority}, Observed {actual_priority}."
        }
