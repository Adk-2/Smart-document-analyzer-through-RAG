from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.db import Base


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    sources = relationship("Source", back_populates="workspace")
    ingestion_history = relationship("IngestionHistory", back_populates="workspace")


class Source(Base):
    __tablename__ = "sources"

    id = Column(String(36), primary_key=True, index=True)
    workspace_name = Column(
        String(255),
        ForeignKey("workspaces.name"),
        nullable=False,
        index=True,
    )
    source_type = Column(String(50), nullable=False)
    source_name = Column(String(500), nullable=False)
    source_url = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default="indexed")
    indexed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    workspace = relationship("Workspace", back_populates="sources")
    ingestion_history = relationship("IngestionHistory", back_populates="source")


class IngestionHistory(Base):
    __tablename__ = "ingestion_history"

    id = Column(Integer, primary_key=True, index=True)
    workspace_name = Column(
        String(255),
        ForeignKey("workspaces.name"),
        nullable=False,
        index=True,
    )
    source_id = Column(String(36), ForeignKey("sources.id"), nullable=True, index=True)
    source_type = Column(String(50), nullable=False)
    source_name = Column(String(500), nullable=True)
    source_url = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    workspace = relationship("Workspace", back_populates="ingestion_history")
    source = relationship("Source", back_populates="ingestion_history")
