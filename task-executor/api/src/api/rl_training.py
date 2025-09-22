"""RL Training API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_db, TaskStatus, TaskType
from ..services.rl_service import RLService
from ..schemas.rl_update import (
    RLUpdateRequest,
    RLUpdateResponse,
    RLInteractionHistory,
    RLInteractionListResponse,
    RLEpisodeStats,
    RLJudgeRequest,
    RLJudgeResponse
)
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/rl/update", response_model=RLUpdateResponse, status_code=200)
async def rl_update(
    request: RLUpdateRequest,
    session: AsyncSession = Depends(get_db)
) -> RLUpdateResponse:
    """
    Execute a shell command in RL training environment and get environment response.
    
    This endpoint:
    1. Validates the task exists and is an RL training task
    2. Executes the shell command safely in the task's environment
    3. Collects comprehensive environment response (metrics, logs, system state)
    4. Evaluates the interaction with an LLM judge
    5. Returns structured response for RL agent learning
    """
    try:
        service = RLService(session)
        response = await service.process_rl_update(request)

        logger.info(
            "api.rl.update.success",
            task_id=str(request.task_id),
            interaction_id=str(response.interaction_id),
            step_number=response.step_number,
            exit_code=response.exit_code
        )

        return response

    except ValueError as e:
        logger.warning(
            "api.rl.update.validation_error",
            task_id=str(request.task_id),
            error=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "api.rl.update.error",
            task_id=str(request.task_id),
            command=request.shell_command[:100],
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl/tasks/{task_id}/history", response_model=RLInteractionListResponse)
async def get_rl_historys(
    task_id: UUID = Path(..., description="RL training task ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_db)
) -> RLInteractionListResponse:
    """Get interaction history for an RL training task."""
    try:
        service = RLService(session)
        interactions = await service.get_interaction_history(task_id, page, page_size)
        
        # Get total count for pagination
        total_interactions = await service.count_interactions(task_id)
        has_next = page * page_size < total_interactions

        return RLInteractionListResponse(
            interactions=interactions,
            total=total_interactions,
            page=page,
            page_size=page_size,
            has_next=has_next
        )

    except Exception as e:
        logger.error(
            "api.rl.interactions.error",
            task_id=str(task_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl/tasks/{task_id}/stats", response_model=RLEpisodeStats)
async def get_rl_episode_stats(
    task_id: UUID = Path(..., description="RL training task ID"),
    session: AsyncSession = Depends(get_db)
) -> RLEpisodeStats:
    """Get statistics for an RL training episode."""
    try:
        service = RLService(session)
        stats = await service.get_episode_stats(task_id)

        return stats

    except Exception as e:
        logger.error(
            "api.rl.stats.error",
            task_id=str(task_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rl/judge", response_model=RLJudgeResponse, status_code=200)
async def evaluate_with_judge(
    request: RLJudgeRequest,
    session: AsyncSession = Depends(get_db)
) -> RLJudgeResponse:
    """
    Standalone endpoint to evaluate an RL interaction with LLM judge.
    
    This can be used for testing judge evaluation logic or getting
    evaluations without executing commands.
    """
    try:
        service = RLService(session)
        
        # Create a dummy task context for evaluation
        evaluation = await service.evaluate_standalone(
            request.shell_command,
            request.env_response,
            request.context
        )

        return RLJudgeResponse(
            evaluation=evaluation,
            model_used="gpt-4",  # This would be configurable
            evaluation_time=0.5  # This would be measured
        )

    except Exception as e:
        logger.error(
            "api.rl.judge.error",
            command=request.shell_command[:100],
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/rl/reward-stats", response_model=Dict[str, Any], status_code=200)
async def get_reward_statistics(
    task_id: UUID = Path(..., description="Task ID"),
    session: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get heuristic reward function statistics for a task.
    
    Returns detailed statistics about the task's interaction history,
    including critical path hits, discoveries made, and exploration metrics.
    """
    try:
        service = RLService(session)
        stats = await service.get_task_reward_statistics(task_id)
        
        logger.info(
            "api.rl.stats.success",
            task_id=str(task_id)
        )
        
        return stats

    except Exception as e:
        logger.error(
            "api.rl.stats.error",
            task_id=str(task_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}/rl/history", status_code=204)
async def cleanup_task_history(
    task_id: UUID = Path(..., description="Task ID"),
    session: AsyncSession = Depends(get_db)
):
    """
    Clean up RL interaction history for a completed task.
    
    This should be called when a task completes to free up memory
    and reset the heuristic reward function's internal state.
    """
    try:
        service = RLService(session)
        await service.cleanup_task_history(task_id)
        
        logger.info(
            "api.rl.cleanup.success",
            task_id=str(task_id)
        )
        
        return  # 204 No Content

    except Exception as e:
        logger.error(
            "api.rl.cleanup.error",
            task_id=str(task_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))
