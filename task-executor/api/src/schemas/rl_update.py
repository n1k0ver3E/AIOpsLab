"""Schemas for RL update endpoints."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime


class RLUpdateRequest(BaseModel):
    """Request schema for RL update endpoint."""
    
    task_id: UUID = Field(..., description="Task ID for the RL episode")
    shell_command: str = Field(..., min_length=1, max_length=10000, description="Shell command to execute")
    
    @validator('shell_command')
    def validate_shell_command(cls, v):
        """Validate shell command for security."""
        # Basic validation - in production, you'd want more sophisticated filtering
        dangerous_commands = ['rm -rf', 'sudo rm', 'mkfs', 'dd if=', 'format', ':(){ :|:& };:']
        v_lower = v.lower()
        for cmd in dangerous_commands:
            if cmd in v_lower:
                raise ValueError(f"Potentially dangerous command detected: {cmd}")
        return v.strip()


class EnvironmentResponse(BaseModel):
    """Environment response data structure."""
    
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="System and application metrics")
    logs: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Recent log entries")
    system_state: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Current system state")
    kubernetes_status: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Kubernetes cluster status")
    execution_output: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Command execution results")


class JudgeEvaluation(BaseModel):
    """LLM judge evaluation result."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Appropriateness score (0.0-1.0)")
    feedback: str = Field(..., min_length=1, max_length=5000, description="Detailed feedback from judge")
    reasoning: Optional[str] = Field(None, max_length=5000, description="Judge's reasoning process")
    category: Optional[str] = Field(None, description="Category of the interaction (e.g., exploration, analysis, mitigation)")


class RLUpdateResponse(BaseModel):
    """Response schema for RL update endpoint."""
    
    id: UUID = Field(..., description="Unique ID for this interaction")
    step_number: int = Field(..., ge=1, description="Step number in the RL episode")
    env_response: Optional[str] = Field(..., description="Environment response data")
    judge_evaluation: Optional[JudgeEvaluation] = Field(None, description="LLM judge evaluation")
    execution_duration: Optional[float] = Field(None, ge=0.0, description="Command execution time in seconds")
    is_finish: Optional[bool] = Field(None, description="if is the finish of the task")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interaction timestamp")


class RLInteractionHistory(BaseModel):
    """Schema for retrieving RL interaction history."""
    
    id: UUID = Field(..., description="Interaction ID")
    task_id: UUID = Field(..., description="Associated task ID")
    step_number: int = Field(..., description="Step number in episode")
    shell_command: str = Field(..., description="Executed command")
    env_response: Dict[str, Any] = Field(..., description="Environment response")
    judge_score: Optional[float] = Field(None, description="Judge evaluation score")
    judge_feedback: Optional[str] = Field(None, description="Judge feedback")
    execution_duration: Optional[float] = Field(None, description="Execution time")
    exit_code: Optional[int] = Field(None, description="Command exit code")
    created_at: datetime = Field(..., description="Creation timestamp")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")

    class Config:
        from_attributes = True


class RLInteractionListResponse(BaseModel):
    """Response schema for listing RL interactions."""
    
    interactions: List[RLInteractionHistory] = Field(..., description="List of RL interactions")
    total: int = Field(..., ge=0, description="Total number of interactions")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class RLEpisodeStats(BaseModel):
    """Statistics for an RL episode (task)."""
    
    task_id: UUID = Field(..., description="Task ID")
    total_interactions: int = Field(..., ge=0, description="Total number of interactions")
    average_judge_score: Optional[float] = Field(None, description="Average judge score")
    episode_duration: Optional[float] = Field(None, description="Total episode duration in seconds")
    success_rate: Optional[float] = Field(None, description="Success rate of commands")
    start_time: Optional[datetime] = Field(None, description="Episode start time")
    end_time: Optional[datetime] = Field(None, description="Episode end time")


class RLJudgeRequest(BaseModel):
    """Request schema for standalone judge evaluation."""
    
    shell_command: str = Field(..., description="Command to evaluate")
    env_response: EnvironmentResponse = Field(..., description="Environment response")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for evaluation")


class RLJudgeResponse(BaseModel):
    """Response schema for standalone judge evaluation."""
    
    evaluation: JudgeEvaluation = Field(..., description="Judge evaluation result")
    model_used: Optional[str] = Field(None, description="LLM model used for evaluation")
    evaluation_time: Optional[float] = Field(None, description="Time taken for evaluation")