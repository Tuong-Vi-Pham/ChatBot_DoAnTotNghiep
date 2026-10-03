import os
from dotenv import load_dotenv

load_dotenv()

try:
    db_url = os.getenv("DB_URL", "sqlite:///./chatbot.db")
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import declarative_base, sessionmaker

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception:
    engine = None
    SessionLocal = None
    Base = object


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from models.conversation_history import ConversationHistory  # noqa: F401
    from models.user import User  # noqa: F401

    Base.metadata.create_all(bind=engine)


def check_db_connection():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
