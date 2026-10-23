from collections.abc import Generator
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import DATABASE_URL

logger = logging.getLogger(__name__)

db_url = DATABASE_URL
if not db_url:
    logger.warning("DATABASE_URL not set; falling back to local SQLite database (sqlite:///./rorak_dev.db)")
    db_url = "sqlite:///./rorak_dev.db"

connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True if not db_url.startswith("sqlite") else False
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
)

class Base(DeclarativeBase):
    pass

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Alias for backwards compatibility
getdb = get_db