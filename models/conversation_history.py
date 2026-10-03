from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, desc
    from database import Base, SessionLocal, engine

    class ConversationHistory(Base):
        __tablename__ = "conversation_history"

        id = Column(Integer, primary_key=True, index=True)
        chat_id = Column(String(255), nullable=False, index=True)
        message_id = Column(String(255), nullable=False, index=True)
        sender_id = Column(String(255), nullable=True, index=True)
        sender_type = Column(String(50), default="user", nullable=False)
        content = Column(Text, nullable=False)
        event_type = Column(String(50), default="message", nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        is_bot_reply = Column(Boolean, default=False, nullable=False)
except ImportError:
    ConversationHistory = None
    SessionLocal = None
    engine = None


def save_message(
    chat_id: str,
    message_id: str,
    content: str,
    sender_id: Optional[str] = None,
    sender_type: str = "user",
    event_type: str = "message",
) -> Optional[ConversationHistory]:
    try:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            existing = (
                db.query(ConversationHistory)
                .filter(ConversationHistory.chat_id == chat_id)
                .filter(ConversationHistory.message_id == message_id)
                .first()
            )
            if existing is not None:
                return existing

            record = ConversationHistory(
                chat_id=chat_id,
                message_id=message_id,
                sender_id=sender_id,
                sender_type=sender_type,
                content=content,
                event_type=event_type,
                is_bot_reply=(sender_type == "bot"),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        finally:
            db.close()
    except Exception as err:
        import logging
        logging.getLogger("smartlogi.db").warning("Database save_message skipped due to connection error: %s", err)
        return None


def get_recent_history(chat_id: str, limit: int = 8, exclude_message_id: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        db = SessionLocal()
        try:
            query = db.query(ConversationHistory).filter(ConversationHistory.chat_id == chat_id)
            if exclude_message_id:
                query = query.filter(ConversationHistory.message_id != exclude_message_id)

            rows = query.order_by(desc(ConversationHistory.created_at)).limit(limit).all()
            rows = list(reversed(rows))
            return [
                {
                    "sender_type": row.sender_type,
                    "content": row.content,
                    "message_id": row.message_id,
                }
                for row in rows
            ]
        finally:
            db.close()
    except Exception as err:
        import logging
        logging.getLogger("smartlogi.db").warning("Database get_recent_history skipped due to connection error: %s", err)
        return []


def build_context_prompt(chat_id: str, current_question: str, limit: int = 8, exclude_message_id: Optional[str] = None) -> str:
    history = get_recent_history(chat_id=chat_id, limit=limit, exclude_message_id=exclude_message_id)

    if not history:
        return current_question

    context_lines = []
    for item in history:
        role = "User" if item["sender_type"] == "user" else "Assistant"
        context_lines.append(f"{role}: {item['content']}")

    return (
        "Lịch sử hội thoại gần đây:\n"
        + "\n".join(context_lines)
        + f"\n\nCâu hỏi hiện tại: {current_question}"
    )


def save_bot_reply(chat_id: str, message_id: str, reply_text: str) -> ConversationHistory:
    return save_message(
        chat_id=chat_id,
        message_id=message_id,
        content=reply_text,
        sender_id="bot",
        sender_type="bot",
        event_type="bot_reply",
    )
