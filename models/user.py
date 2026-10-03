try:
    from sqlalchemy import Column, Integer, String
    from database import Base

    class User(Base):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(100), unique=True, nullable=False)
        email = Column(String(255), nullable=False)
except ImportError:
    User = None
