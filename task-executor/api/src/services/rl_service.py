"""Service layer for RL update operations."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from ..models import Task, TaskStatus, TaskType, RLInteraction
from ..schemas.rl_update import (
    RLUpdateRequest, RLUpdateResponse, EnvironmentResponse, 
    JudgeEvaluation, RLInteractionHistory, RLEpisodeStats
)
from ..config.logging import get_logger
from ..config.settings import settings
from .heuristic_reward import HeuristicRewardFunction

logger = get_logger(__name__)


class RLService:
    """Service for handling RL update operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.heuristic_reward = HeuristicRewardFunction()
    
    async def process_rl_update(self, request: RLUpdateRequest) -> RLUpdateResponse:
        """Process an RL update request."""
        logger.info(
            "rl.update.start",
            task_id=str(request.task_id),
            command=request.shell_command[:100] + "..." if len(request.shell_command) > 100 else request.shell_command
        )
        
        # Verify task exists and is running
        task = await self._get_task(request.task_id)
        if not task:
            raise ValueError(f"Task {request.task_id} not found")
        
        if task.status not in [TaskStatus.RUNNING, TaskStatus.PENDING]:
            raise ValueError(f"Task {request.task_id} is not in a runnable state (status: {task.status})")
        
        if task.task_type != TaskType.RL_TRAINING:
            raise ValueError(f"Task {request.task_id} is not an RL training task (type: {task.task_type})")
        
        # Get next step number
        step_number = await self._get_next_step_number(request.task_id)
        
        # Execute shell command
        start_time = time.time()
        execution_result = await self._execute_shell_command(request.shell_command)
        execution_duration = time.time() - start_time
        
        # Collect environment response
        env_response = await self._collect_environment_response(task, execution_result)
        
        # Create RL interaction record
        interaction = RLInteraction(
            task_id=request.task_id,
            step_number=step_number,
            shell_command=request.shell_command,
            env_response=env_response.model_dump(),
            execution_duration=execution_duration,
            exit_code=execution_result.get('exit_code'),
            stdout=execution_result.get('stdout'),
            stderr=execution_result.get('stderr'),
            executed_at=datetime.utcnow()
        )
        
        self.session.add(interaction)
        await self.session.commit()
        await self.session.refresh(interaction)
        
        # Evaluate with both LLM judge and heuristic reward
        judge_evaluation = None
        if settings.ENABLE_RL_JUDGE:
            try:
                # Try LLM judge first
                judge_evaluation = await self._evaluate_with_judge(
                    request.shell_command, env_response, task
                )
                
            except Exception as e:
                logger.warning("rl.judge.error", error=str(e), task_id=str(request.task_id))
        
        # Always use heuristic reward as primary or fallback
        if judge_evaluation is None:
            # Use heuristic reward as primary evaluation
            judge_evaluation = self.heuristic_reward.calculate_reward(
                task_id=request.task_id,
                problem_id=task.problem_id,
                command=request.shell_command,
                env_response=env_response,
                step_number=step_number,
                context={}
            )
            logger.info("rl.heuristic.primary", task_id=str(request.task_id))
        else:
            # Blend LLM judge with heuristic reward for enhanced evaluation
            heuristic_eval = self.heuristic_reward.calculate_reward(
                task_id=request.task_id,
                problem_id=task.problem_id,
                command=request.shell_command,
                env_response=env_response,
                step_number=step_number,
                context={}
            )
            
            # Weighted combination: 60% heuristic (ground truth), 40% LLM judge
            blended_score = 0.6 * heuristic_eval.score + 0.4 * judge_evaluation.score
            judge_evaluation = JudgeEvaluation(
                score=blended_score,
                feedback=f"Heuristic: {heuristic_eval.feedback} | LLM: {judge_evaluation.feedback}",
                reasoning=f"Blended evaluation - Heuristic: {heuristic_eval.reasoning} | LLM: {judge_evaluation.reasoning}",
                category=heuristic_eval.category  # Use heuristic category as primary
            )
            logger.info("rl.blended.evaluation", task_id=str(request.task_id), 
                       heuristic_score=heuristic_eval.score, llm_score=judge_evaluation.score)
        
        # Update interaction with evaluation results
        interaction.judge_score = judge_evaluation.score
        interaction.judge_feedback = judge_evaluation.feedback
        await self.session.commit()
        
        logger.info(
            "rl.update.complete",
            task_id=str(request.task_id),
            interaction_id=str(interaction.id),
            step_number=step_number,
            execution_duration=execution_duration
        )
        
        return RLUpdateResponse(
            interaction_id=interaction.id,
            step_number=step_number,
            env_response=env_response,
            judge_evaluation=judge_evaluation,
            execution_duration=execution_duration,
            exit_code=execution_result.get('exit_code'),
            timestamp=interaction.created_at
        )
    
    async def get_interaction_history(
        self, 
        task_id: UUID,
        page: int = 1,
        page_size: int = 20
    ) -> List[RLInteractionHistory]:
        """Get RL interaction history for a task."""
        offset = (page - 1) * page_size
        
        query = (
            select(RLInteraction)
            .where(RLInteraction.task_id == task_id)
            .order_by(RLInteraction.step_number.asc())
            .offset(offset)
            .limit(page_size)
        )
        
        result = await self.session.execute(query)
        interactions = result.scalars().all()
        
        return [
            RLInteractionHistory.model_validate(interaction)
            for interaction in interactions
        ]
    
    async def get_episode_stats(self, task_id: UUID) -> RLEpisodeStats:
        """Get statistics for an RL episode."""
        # Basic stats query
        stats_query = (
            select(
                func.count(RLInteraction.id).label('total_interactions'),
                func.avg(RLInteraction.judge_score).label('avg_score'),
                func.min(RLInteraction.created_at).label('start_time'),
                func.max(RLInteraction.created_at).label('end_time'),
                func.avg(
                    func.case(
                        (RLInteraction.exit_code == 0, 1.0),
                        else_=0.0
                    )
                ).label('success_rate')
            )
            .where(RLInteraction.task_id == task_id)
        )
        
        result = await self.session.execute(stats_query)
        stats = result.first()
        
        # Calculate episode duration
        episode_duration = None
        if stats.start_time and stats.end_time:
            episode_duration = (stats.end_time - stats.start_time).total_seconds()
        
        return RLEpisodeStats(
            task_id=task_id,
            total_interactions=stats.total_interactions or 0,
            average_judge_score=float(stats.avg_score) if stats.avg_score else None,
            episode_duration=episode_duration,
            success_rate=float(stats.success_rate) if stats.success_rate else None,
            start_time=stats.start_time,
            end_time=stats.end_time
        )
    
    async def _get_task(self, task_id: UUID) -> Optional[Task]:
        """Get task by ID."""
        query = select(Task).where(Task.id == task_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_next_step_number(self, task_id: UUID) -> int:
        """Get the next step number for a task."""
        query = (
            select(func.coalesce(func.max(RLInteraction.step_number), 0) + 1)
            .where(RLInteraction.task_id == task_id)
        )
        result = await self.session.execute(query)
        return result.scalar()
    
    async def _execute_shell_command(self, command: str) -> Dict[str, Any]:
        """Execute shell command safely using the task's orchestrator."""
        try:
            # Try to get the orchestrator executor for this task
            executor = await self._get_rl_executor()
            if executor and executor.rl_mode:
                # Use the orchestrator's RL command execution
                result = await executor.execute_rl_command(command)
                return result
            else:
                # Fallback to direct execution (less secure/integrated)
                return await self._execute_command_direct(command)
                
        except Exception as e:
            logger.error("rl.command.execution.failed", command=command[:100], error=str(e))
            return {
                'exit_code': -2,
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'command': command
            }

    async def _execute_command_direct(self, command: str) -> Dict[str, Any]:
        """Direct command execution fallback."""
        try:
            # Use asyncio.subprocess for non-blocking execution
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024  # 1MB limit for output
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=settings.RL_COMMAND_TIMEOUT
            )
            
            return {
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace')[:10000],  # Limit output size
                'stderr': stderr.decode('utf-8', errors='replace')[:10000],
                'command': command
            }
            
        except asyncio.TimeoutError:
            logger.warning("rl.command.timeout", command=command[:100])
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': f'Command timed out after {settings.RL_COMMAND_TIMEOUT} seconds',
                'command': command
            }
        except Exception as e:
            logger.error("rl.command.error", command=command[:100], error=str(e))
            return {
                'exit_code': -2,
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'command': command
            }

    async def _get_rl_executor(self):
        """Get the RL orchestrator executor for command execution."""
        # This would need to be implemented based on your worker management
        # For now, return None to use fallback execution
        # In a full implementation, you'd:
        # 1. Find which worker is handling the task
        # 2. Get the OrchestratorExecutor instance
        # 3. Return it for RL command execution
        return None
    
    async def _collect_environment_response(
        self, 
        task: Task, 
        execution_result: Dict[str, Any]
    ) -> EnvironmentResponse:
        """Collect comprehensive environment response."""
        try:
            # Try to use orchestrator's environment collection if available
            executor = await self._get_rl_executor()
            if executor and executor.rl_mode:
                env_state = await executor.get_environment_state()
                env_data = {
                    'metrics': env_state.get('metrics', {}),
                    'logs': env_state.get('logs', []),
                    'system_state': env_state.get('system_state', {}),
                    'kubernetes_status': env_state.get('kubernetes_status', {}),
                    'execution_output': execution_result
                }
            else:
                # Fallback to basic collection
                env_data = {
                    'metrics': await self._collect_metrics(task),
                    'logs': await self._collect_recent_logs(task),
                    'system_state': await self._collect_system_state(task),
                    'kubernetes_status': await self._collect_k8s_status(task),
                    'execution_output': execution_result
                }
            
            return EnvironmentResponse(**env_data)
            
        except Exception as e:
            logger.error("rl.env_collection.failed", task_id=str(task.id), error=str(e))
            # Return basic response with execution result
            return EnvironmentResponse(
                execution_output=execution_result,
                metrics={"error": str(e)},
                logs=[{"error": "Failed to collect logs", "details": str(e)}],
                system_state={"error": str(e)},
                kubernetes_status={"error": str(e)}
            )
    
    async def _collect_metrics(self, task: Task) -> Dict[str, Any]:
        """Collect system and application metrics."""
        # Placeholder - integrate with Prometheus/metrics system
        return {
            'cpu_usage': 0.5,
            'memory_usage': 0.6,
            'disk_usage': 0.3,
            'network_io': {'bytes_in': 1024, 'bytes_out': 2048},
            'application_metrics': {}
        }
    
    async def _collect_recent_logs(self, task: Task) -> List[Dict[str, Any]]:
        """Collect recent log entries."""
        # Placeholder - integrate with log aggregation system
        return [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'INFO',
                'message': 'Application running normally',
                'source': 'application'
            }
        ]
    
    async def _collect_system_state(self, task: Task) -> Dict[str, Any]:
        """Collect current system state."""
        # Placeholder - collect system state information
        return {
            'uptime': 3600,
            'load_average': [0.5, 0.6, 0.7],
            'processes': 150,
            'connections': 25
        }
    
    async def _collect_k8s_status(self, task: Task) -> Dict[str, Any]:
        """Collect Kubernetes cluster status."""
        # Placeholder - integrate with Kubernetes API
        return {
            'pods_running': 10,
            'pods_failed': 0,
            'services_count': 5,
            'nodes_ready': 3
        }
    
    async def _evaluate_with_judge(
        self, 
        command: str, 
        env_response: EnvironmentResponse,
        task: Task
    ) -> JudgeEvaluation:
        """Evaluate interaction with LLM judge using existing agent infrastructure."""
        try:
            # Import the GPTAgent from existing infrastructure
            from ..workers.simple_gpt_agent import GPTAgent
            import os
            
            # Create prompt for judge evaluation
            prompt = self._build_judge_prompt(command, env_response, task)
            
            # Try to create a judge agent using existing infrastructure
            judge_agent = await self._create_judge_agent()
            if judge_agent:
                try:
                    # Use the agent's get_action method to get evaluation
                    response = await judge_agent.get_action(prompt)
                    return self._parse_judge_response(response, command, env_response, task)
                except Exception as e:
                    logger.warning("rl.judge.agent.failed", error=str(e))
                    # Fall back to rule-based evaluation
                    return self._rule_based_evaluation(command, env_response, task)
            else:
                # No judge agent available, use rule-based evaluation
                return self._rule_based_evaluation(command, env_response, task)
            
        except Exception as e:
            logger.error("rl.judge.evaluation.error", error=str(e))
            # Return safe fallback evaluation
            return JudgeEvaluation(
                score=0.5,
                feedback=f"Unable to evaluate command '{command}' due to error: {str(e)}",
                reasoning="Evaluation failed, using neutral score",
                category="error"
            )

    def _build_judge_prompt(self, command: str, env_response: EnvironmentResponse, task: Task) -> str:
        """Build prompt for LLM judge evaluation."""
        return f"""You are an expert AIOps judge evaluating the quality of diagnostic commands.

Task Context:
- Problem ID: {task.problem_id}
- Task Type: RL Training for AIOps

Command Executed: {command}

Execution Results:
- Exit Code: {env_response.execution_output.get('exit_code', 'N/A')}
- Output: {env_response.execution_output.get('stdout', '')[:500]}
- Error: {env_response.execution_output.get('stderr', '')[:500]}

Environment State:
- Metrics: {json.dumps(env_response.metrics, indent=2)[:500]}
- Kubernetes Status: {json.dumps(env_response.kubernetes_status, indent=2)[:200]}

Please evaluate this command on a scale of 0.0-1.0 considering:
1. Safety (no harmful operations)
2. Relevance to AIOps/debugging tasks
3. Diagnostic value (does it provide useful information)
4. Appropriateness for the given problem context

Respond in this exact JSON format:
{{
  "score": 0.0-1.0,
  "feedback": "Brief explanation of the score",
  "reasoning": "Detailed reasoning for the evaluation",
  "category": "one of: exploration, analysis, monitoring, mitigation, error"
}}"""

    async def _create_judge_agent(self):
        """Create a judge agent using existing agent infrastructure."""
        try:
            from ..workers.simple_gpt_agent import GPTAgent
            import os
            
            # Try OpenAI first
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return GPTAgent(
                    model="gpt-4o-mini",  # Use cheaper model for judge
                    api_key=api_key,
                    temperature=0.1  # Lower temperature for consistent evaluation
                )
            
            # Try OpenRouter as fallback
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                return GPTAgent(
                    model="openai/gpt-4o-mini",
                    api_key=openrouter_key,
                    temperature=0.1,
                    base_url=base_url
                )
            
            logger.warning("rl.judge.no_api_key", message="No API key found for judge agent")
            return None
            
        except Exception as e:
            logger.error("rl.judge.agent.creation.failed", error=str(e))
            return None

    def _parse_judge_response(self, response: str, command: str, env_response: EnvironmentResponse, task: Task = None) -> JudgeEvaluation:
        """Parse LLM judge response into structured evaluation."""
        try:
            # Try to parse as JSON
            import json
            # Clean up response (remove markdown formatting if present)
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            data = json.loads(clean_response)
            
            return JudgeEvaluation(
                score=float(data.get("score", 0.5)),
                feedback=data.get("feedback", f"Evaluation for command: {command}"),
                reasoning=data.get("reasoning", "No reasoning provided"),
                category=data.get("category", "exploration")
            )
            
        except Exception as e:
            logger.warning("rl.judge.parse.failed", error=str(e), response=response[:200])
            # Fallback to simple parsing or rule-based evaluation
            return self._rule_based_evaluation(command, env_response, task)

    def _rule_based_evaluation(self, command: str, env_response: EnvironmentResponse, task: Task = None) -> JudgeEvaluation:
        """Enhanced rule-based evaluation using heuristic reward function."""
        if task:
            # Use full heuristic reward function if task context is available
            return self.heuristic_reward.calculate_reward(
                task_id=task.id,
                problem_id=task.problem_id,
                command=command,
                env_response=env_response,
                step_number=1,  # Default step number for fallback
                context={}
            )
        else:
            # Fallback to basic safety-focused evaluation
            exit_code = env_response.execution_output.get('exit_code', -1)
            stdout = env_response.execution_output.get('stdout', '').lower()
            stderr = env_response.execution_output.get('stderr', '').lower()
            
            # Basic scoring based on command characteristics
            score = 0.5  # Default neutral score
            category = "exploration"
            
            # Safety check - dangerous commands get low scores
            dangerous_patterns = ['rm -rf', 'dd if=', 'mkfs', 'format', 'shutdown', 'reboot']
            if any(pattern in command.lower() for pattern in dangerous_patterns):
                score = 0.1
                category = "error"
                feedback = "Potentially dangerous command detected"
            
            # Positive scoring for useful diagnostic commands
            elif command.startswith(('kubectl', 'docker', 'ps', 'top', 'netstat', 'lsof', 'df', 'free')):
                if exit_code == 0:
                    score = 0.8
                    category = "monitoring"
                    feedback = f"Good diagnostic command executed successfully"
                else:
                    score = 0.4
                    category = "exploration"
                    feedback = f"Diagnostic command failed but was appropriate to try"
            
            # Medium score for other system commands
            elif exit_code == 0 and len(stdout) > 10:
                score = 0.6
                feedback = f"Command executed successfully and produced output"
            elif exit_code != 0:
                score = 0.3
                feedback = f"Command failed with exit code {exit_code}"
            else:
                feedback = f"Command executed but produced limited output"
                
            return JudgeEvaluation(
                score=score,
                feedback=feedback,
                reasoning=f"Basic rule-based evaluation: exit_code={exit_code}, command_type='{category}'",
                category=category
            )
    
    async def count_interactions(self, task_id: UUID) -> int:
        """Count total interactions for a task."""
        query = (
            select(func.count(RLInteraction.id))
            .where(RLInteraction.task_id == task_id)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def evaluate_standalone(
        self, 
        shell_command: str, 
        env_response: EnvironmentResponse,
        context: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate interaction with LLM judge without a specific task context."""
        try:
            # Similar to _evaluate_with_judge but without task context
            prompt = f"""
            Evaluate the following RL interaction for an AIOps task:
            
            Command: {shell_command}
            Exit Code: {env_response.execution_output.get('exit_code', 'N/A')}
            Output: {env_response.execution_output.get('stdout', '')[:500]}
            Error: {env_response.execution_output.get('stderr', '')[:500]}
            
            Metrics: {json.dumps(env_response.metrics, indent=2)[:500]}
            Context: {json.dumps(context, indent=2)[:500]}
            
            Rate the appropriateness of this command (0.0-1.0) and provide feedback.
            Consider: relevance to AIOps tasks, safety, diagnostic value, and potential impact.
            """
            
            # Placeholder evaluation - replace with actual LLM call
            score = 0.8 if env_response.execution_output.get('exit_code') == 0 else 0.3
            feedback = f"Command '{shell_command}' executed with exit code {env_response.execution_output.get('exit_code')}"
            
            return JudgeEvaluation(
                score=score,
                feedback=feedback,
                reasoning="Standalone evaluation - integrate with actual LLM",
                category="exploration"
            )
            
        except Exception as e:
            logger.error("rl.judge.standalone.error", error=str(e))
            raise
    
    async def get_task_reward_statistics(self, task_id: UUID) -> Dict[str, Any]:
        """Get reward function statistics for a task."""
        try:
            stats = self.heuristic_reward.get_task_statistics(task_id)
            return {
                "task_id": str(task_id),
                "heuristic_stats": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("rl.stats.error", task_id=str(task_id), error=str(e))
            return {"error": str(e)}
    
    async def cleanup_task_history(self, task_id: UUID):
        """Clean up task history when task completes."""
        try:
            self.heuristic_reward.reset_task_history(task_id)
            logger.info("rl.cleanup.success", task_id=str(task_id))
        except Exception as e:
            logger.error("rl.cleanup.error", task_id=str(task_id), error=str(e))