"""RL Interaction model for storing reinforcement learning interactions."""

from sqlalchemy import (
    Column, String, DateTime, Text, Integer, Float, ForeignKey, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime, timezone

from .database import Base


class RLInteraction(Base):
    """Model for storing RL agent interactions with the environment."""

    __tablename__ = "rl_interactions"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )

    # Foreign key to task
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Associated task ID"
    )

    # Step sequence in the RL episode
    step_number = Column(
        Integer,
        nullable=False,
        comment="Sequential step number in the RL episode"
    )

    # Input shell command
    shell_command = Column(
        Text,
        nullable=False,
        comment="Shell command executed by the RL agent"
    )

    # Environment response (metrics, logs, system state)
    env_response = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Environment response including metrics, logs, and system state"
    )

    # LLM judge evaluation
    judge_score = Column(
        Float,
        nullable=True,
        comment="Judge evaluation score (0.0-1.0)"
    )

    judge_feedback = Column(
        Text,
        nullable=True,
        comment="Detailed feedback from the LLM judge"
    )

    # Execution metadata
    execution_duration = Column(
        Float,
        nullable=True,
        comment="Command execution duration in seconds"
    )

    exit_code = Column(
        Integer,
        nullable=True,
        comment="Shell command exit code"
    )

    stdout = Column(
        Text,
        nullable=True,
        comment="Standard output from command execution"
    )

    stderr = Column(
        Text,
        nullable=True,
        comment="Standard error from command execution"
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        comment="When the interaction was recorded"
    )

    executed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the command was executed"
    )

    # Relationship to task
    task = relationship("Task", back_populates="rl_interactions")

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_rl_task_step", "task_id", "step_number"),
        Index("idx_rl_created_at", "created_at"),
        Index("idx_rl_judge_score", "judge_score"),
    )

    def __repr__(self):
        return f"<RLInteraction(id={self.id}, task_id={self.task_id}, step={self.step_number})>"
