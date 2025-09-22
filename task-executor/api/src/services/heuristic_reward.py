"""
Ground Truth-Based Heuristic Reward Function for AIOps RL Training.

This module implements sophisticated reward functions that leverage ground truth
information about problems to provide meaningful feedback to RL agents.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import UUID

from ..schemas.rl_update import EnvironmentResponse, JudgeEvaluation
from ..config.logging import get_logger

logger = get_logger(__name__)


class ProblemGroundTruth:
    """Encapsulates ground truth information about a specific problem."""
    
    def __init__(self, problem_id: str):
        self.problem_id = problem_id
        self.parse_problem_info()
    
    def parse_problem_info(self):
        """Parse problem ID to extract ground truth information."""
        # Parse problem format: "k8s_target_port-misconfig-detection-1"
        parts = self.problem_id.split('-')
        
        self.app_type = None
        self.fault_type = None
        self.task_type = None
        self.variant = None
        
        if len(parts) >= 3:
            self.fault_type = '-'.join(parts[:-2])  # e.g., "k8s_target_port-misconfig"
            self.task_type = parts[-2]              # e.g., "detection"
            self.variant = parts[-1]                # e.g., "1"
        
        # Extract application context
        if 'hotel' in self.problem_id.lower():
            self.app_type = 'hotel'
        elif 'social' in self.problem_id.lower():
            self.app_type = 'social'
        elif 'astronomy' in self.problem_id.lower():
            self.app_type = 'astronomy'
        
        # Define ground truth based on problem type
        self._define_ground_truth()
    
    def _define_ground_truth(self):
        """Define ground truth critical paths and expected behaviors."""
        self.critical_commands = []
        self.critical_resources = []
        self.expected_discoveries = []
        self.fault_indicators = []
        
        # K8s Target Port Misconfiguration
        if 'k8s_target_port' in self.fault_type and 'misconfig' in self.fault_type:
            self.critical_resources = [
                'service', 'svc', 'user-service', 'text-service', 'post-storage-service'
            ]
            self.critical_commands = [
                'kubectl get svc',
                'kubectl describe svc',
                'kubectl get service',
                'kubectl describe service',
                'kubectl get pods',
                'kubectl logs'
            ]
            self.expected_discoveries = [
                'targetPort', 'port', '9090', '8080', 'endpoint', 'connection refused'
            ]
            self.fault_indicators = [
                'CrashLoopBackOff', 'connection refused', 'target port', 'port mismatch'
            ]
        
        # Pod Kill/Failure
        elif 'pod_kill' in self.fault_type or 'pod_failure' in self.fault_type:
            self.critical_commands = [
                'kubectl get pods',
                'kubectl describe pod',
                'kubectl logs',
                'kubectl get events'
            ]
            self.expected_discoveries = [
                'Terminated', 'Killed', 'CrashLoopBackOff', 'RestartCount'
            ]
            self.fault_indicators = [
                'pod killed', 'container terminated', 'exit code'
            ]
        
        # Network Issues
        elif 'network' in self.fault_type:
            self.critical_commands = [
                'kubectl get svc',
                'kubectl get endpoints',
                'kubectl describe svc',
                'netstat', 'ss', 'ping', 'telnet'
            ]
            self.expected_discoveries = [
                'network', 'connection', 'timeout', 'unreachable'
            ]
        
        # Resource/Scaling Issues
        elif 'scale' in self.fault_type or 'cpu' in self.fault_type:
            self.critical_commands = [
                'kubectl top pods',
                'kubectl top nodes',
                'kubectl describe pod',
                'kubectl get hpa'
            ]
            self.expected_discoveries = [
                'cpu', 'memory', 'resource', 'limit', 'request'
            ]
        
        # Storage Issues
        elif 'storage' in self.fault_type or 'pv' in self.fault_type:
            self.critical_commands = [
                'kubectl get pv',
                'kubectl get pvc',
                'kubectl describe pv',
                'kubectl describe pvc'
            ]
            self.expected_discoveries = [
                'persistent volume', 'storage', 'bound', 'available'
            ]
        
        # Hotel Reservation App Misconfiguration
        elif 'misconfig_app_hotel_res' in self.fault_type:
            self.critical_resources = [
                'mongodb-geo', 'geo', 'hotel-reservation', 'svc', 'service'
            ]
            self.critical_commands = [
                'kubectl get pods',
                'kubectl describe pod',
                'kubectl logs',
                'kubectl get svc',
                'kubectl describe svc',
                'kubectl patch svc',
                'kubectl get deploy'
            ]
            self.expected_discoveries = [
                'no reachable servers', 'connection', 'mongodb-geo:27777', 
                'port', '27017', '27777', 'database', 'panic', 'Error', 'CrashLoopBackOff'
            ]
            self.fault_indicators = [
                'no reachable servers', 'connection refused', 'database connection',
                'panic', 'mongodb-geo', 'port mismatch'
            ]
        
        # Default for unknown problems
        else:
            self.critical_commands = [
                'kubectl get pods',
                'kubectl get svc',
                'kubectl logs',
                'kubectl describe'
            ]


class HeuristicRewardFunction:
    """
    Ground truth-based heuristic reward function for AIOps RL training.
    
    This reward function provides sophisticated scoring based on:
    1. Critical path alignment with ground truth
    2. Exploration quality and relevance  
    3. Progress toward problem resolution
    4. Safety and efficiency considerations
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.interaction_history = {}  # Track per-task interaction history
        
        # Reward weights
        self.weights = {
            'critical_path': 0.4,      # Matching critical path to solution
            'exploration': 0.25,       # Quality exploration commands
            'discovery': 0.2,          # Finding relevant information
            'safety': 0.1,             # Safety considerations
            'efficiency': 0.05         # Avoiding redundancy
        }
    
    def calculate_reward(
        self,
        task_id: UUID,
        problem_id: str,
        command: str,
        env_response: EnvironmentResponse,
        step_number: int,
        context: Dict[str, Any] = None
    ) -> JudgeEvaluation:
        """
        Calculate comprehensive heuristic reward for an RL interaction.
        
        Args:
            task_id: Unique task identifier
            problem_id: Problem identifier (e.g., "k8s_target_port-misconfig-detection-1")
            command: Executed shell command
            env_response: Environment response with execution results
            step_number: Current step in the episode
            context: Additional context information
            
        Returns:
            JudgeEvaluation with detailed scoring and feedback
        """
        try:
            # Initialize task history if needed
            if task_id not in self.interaction_history:
                self.interaction_history[task_id] = {
                    'commands': [],
                    'discoveries': set(),
                    'resources_explored': set(),
                    'critical_hits': 0,
                    'redundant_commands': 0
                }
            
            history = self.interaction_history[task_id]
            ground_truth = ProblemGroundTruth(problem_id)
            
            # Calculate individual reward components
            critical_path_score = self._calculate_critical_path_reward(
                command, env_response, ground_truth, history
            )
            
            exploration_score = self._calculate_exploration_reward(
                command, env_response, ground_truth, history
            )
            
            discovery_score = self._calculate_discovery_reward(
                command, env_response, ground_truth, history
            )
            
            safety_score = self._calculate_safety_reward(command, env_response)
            
            efficiency_score = self._calculate_efficiency_reward(
                command, history, step_number
            )
            
            # Combine weighted scores
            total_score = (
                self.weights['critical_path'] * critical_path_score +
                self.weights['exploration'] * exploration_score +
                self.weights['discovery'] * discovery_score +
                self.weights['safety'] * safety_score +
                self.weights['efficiency'] * efficiency_score
            )
            
            # Ensure score is in [0, 1] range
            total_score = max(0.0, min(1.0, total_score))
            
            # Update history
            history['commands'].append(command)
            
            # Generate detailed feedback
            feedback = self._generate_feedback(
                command, total_score, critical_path_score, exploration_score,
                discovery_score, safety_score, efficiency_score, ground_truth
            )
            
            category = self._determine_category(
                command, critical_path_score, exploration_score
            )
            
            reasoning = self._generate_reasoning(
                critical_path_score, exploration_score, discovery_score,
                safety_score, efficiency_score, ground_truth
            )
            
            self.logger.info(
                "heuristic.reward.calculated",
                task_id=str(task_id),
                command=command[:100],
                total_score=total_score,
                critical_path=critical_path_score,
                exploration=exploration_score,
                discovery=discovery_score
            )
            
            return JudgeEvaluation(
                score=total_score,
                feedback=feedback,
                reasoning=reasoning,
                category=category
            )
            
        except Exception as e:
            self.logger.error("heuristic.reward.error", error=str(e))
            # Return safe fallback
            return JudgeEvaluation(
                score=0.3,
                feedback=f"Error calculating heuristic reward: {str(e)}",
                reasoning="Fallback due to calculation error",
                category="error"
            )
    
    def _calculate_critical_path_reward(
        self,
        command: str,
        env_response: EnvironmentResponse,
        ground_truth: ProblemGroundTruth,
        history: Dict
    ) -> float:
        """Calculate reward for commands on the critical path to solution."""
        score = 0.0
        exit_code = env_response.execution_output.get('exit_code', -1)
        
        # Check if command matches critical commands for this problem
        command_lower = command.lower()
        for critical_cmd in ground_truth.critical_commands:
            if critical_cmd.lower() in command_lower:
                # Higher score for mitigation commands (patch, apply, etc.)
                if any(mitigation_word in command_lower for mitigation_word in ['patch', 'apply', 'edit', 'replace']):
                    score += 0.6  # Higher reward for actual mitigation actions
                    # Extra bonus for successful mitigation commands
                    if exit_code == 0:
                        score += 0.2
                else:
                    score += 0.3
                history['critical_hits'] += 1
                break
        
        # Check if command targets critical resources
        for resource in ground_truth.critical_resources:
            if resource.lower() in command_lower:
                score += 0.2
                history['resources_explored'].add(resource)
                break
        
        # Bonus for successful execution of critical commands
        if score > 0 and exit_code == 0:
            score += 0.2
        
        # Extra bonus for commands that likely reveal the fault
        stdout = env_response.execution_output.get('stdout', '').lower()
        stderr = env_response.execution_output.get('stderr', '').lower()
        
        for indicator in ground_truth.fault_indicators:
            if indicator.lower() in stdout or indicator.lower() in stderr:
                score += 0.3  # High reward for finding fault indicators
                break
        
        return min(1.0, score)
    
    def _calculate_exploration_reward(
        self,
        command: str,
        env_response: EnvironmentResponse,
        ground_truth: ProblemGroundTruth,
        history: Dict
    ) -> float:
        """Calculate reward for meaningful exploration activities."""
        score = 0.0
        command_lower = command.lower()
        
        # Reward different types of exploration commands
        exploration_patterns = {
            'kubectl_get': r'kubectl\s+get\s+\w+',
            'kubectl_describe': r'kubectl\s+describe\s+\w+',
            'kubectl_logs': r'kubectl\s+logs\s+',
            'system_monitoring': r'(top|ps|netstat|ss|lsof|df|free)',
            'network_debug': r'(ping|telnet|curl|wget)',
            'file_inspection': r'(cat|less|head|tail|grep)\s+',
        }
        
        for pattern_name, pattern in exploration_patterns.items():
            if re.search(pattern, command_lower):
                if pattern_name.startswith('kubectl'):
                    score += 0.4  # High reward for k8s exploration
                elif pattern_name == 'system_monitoring':
                    score += 0.3  # Good reward for system monitoring
                else:
                    score += 0.2  # Moderate reward for other exploration
                break
        
        # Bonus for successful exploration
        exit_code = env_response.execution_output.get('exit_code', 0)
        stdout_len = len(env_response.execution_output.get('stdout', ''))
        
        if exit_code == 0 and stdout_len > 50:  # Successful with meaningful output
            score += 0.2
        elif exit_code == 0:  # Successful but limited output
            score += 0.1
        
        # Penalty for failed exploration (but still some reward for trying)
        if exit_code != 0 and score > 0:
            score *= 0.7
        
        return min(1.0, score)
    
    def _calculate_discovery_reward(
        self,
        command: str,
        env_response: EnvironmentResponse,
        ground_truth: ProblemGroundTruth,
        history: Dict
    ) -> float:
        """Calculate reward for discovering relevant information."""
        score = 0.0
        
        stdout = env_response.execution_output.get('stdout', '').lower()
        stderr = env_response.execution_output.get('stderr', '').lower()
        combined_output = stdout + ' ' + stderr
        
        # Reward for discovering expected information
        discoveries_found = 0
        for discovery in ground_truth.expected_discoveries:
            if discovery.lower() in combined_output:
                discoveries_found += 1
                history['discoveries'].add(discovery)
        
        if discoveries_found > 0:
            score = min(1.0, discoveries_found * 0.3)
        
        # Additional reward for discovering new information
        new_discoveries = len(history['discoveries'])
        if new_discoveries > 0:
            score += min(0.3, new_discoveries * 0.05)
        
        # Extra reward for successful mitigation command output
        if 'patched' in combined_output or 'applied' in combined_output or 'updated' in combined_output:
            score += 0.4  # High reward for successful mitigation actions
        
        # Reward for finding error messages that could indicate the problem
        error_indicators = ['error', 'failed', 'timeout', 'refused', 'unreachable']
        for indicator in error_indicators:
            if indicator in combined_output:
                score += 0.1
                break
        
        return min(1.0, score)
    
    def _calculate_safety_reward(
        self,
        command: str,
        env_response: EnvironmentResponse
    ) -> float:
        """Calculate safety reward (penalties for dangerous commands)."""
        command_lower = command.lower()
        
        # Severe penalties for dangerous commands
        dangerous_patterns = [
            r'rm\s+-rf',
            r'sudo\s+rm',
            r'mkfs',
            r'dd\s+if=',
            r'format',
            r'shutdown',
            r'reboot',
            r'kill\s+-9',
            r'pkill',
            r':\(\)\{\s*:\|\:&\s*\};\:'  # Fork bomb
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return 0.0  # Zero reward for dangerous commands
        
        # Moderate penalties for potentially disruptive commands
        disruptive_patterns = [
            r'kubectl\s+delete',
            r'docker\s+rm',
            r'systemctl\s+stop',
            r'service\s+\w+\s+stop'
        ]
        
        for pattern in disruptive_patterns:
            if re.search(pattern, command_lower):
                return 0.3  # Low reward for disruptive commands
        
        # Full reward for safe commands
        return 1.0
    
    def _calculate_efficiency_reward(
        self,
        command: str,
        history: Dict,
        step_number: int
    ) -> float:
        """Calculate efficiency reward (penalties for redundancy)."""
        score = 1.0
        
        # Count similar commands
        similar_commands = 0
        for prev_command in history['commands']:
            if self._commands_similar(command, prev_command):
                similar_commands += 1
        
        # Penalty for redundant commands
        if similar_commands > 0:
            score -= min(0.5, similar_commands * 0.15)
        
        # Slight penalty for very long episodes (encourage efficiency)
        if step_number > 20:
            score -= min(0.3, (step_number - 20) * 0.02)
        
        return max(0.0, score)
    
    def _commands_similar(self, cmd1: str, cmd2: str) -> bool:
        """Check if two commands are similar (for redundancy detection)."""
        # Simple similarity check - could be enhanced
        cmd1_tokens = set(cmd1.lower().split())
        cmd2_tokens = set(cmd2.lower().split())
        
        # If commands have significant overlap, consider them similar
        intersection = cmd1_tokens & cmd2_tokens
        union = cmd1_tokens | cmd2_tokens
        
        if len(union) == 0:
            return False
        
        similarity = len(intersection) / len(union)
        return similarity > 0.7
    
    def _generate_feedback(
        self,
        command: str,
        total_score: float,
        critical_path_score: float,
        exploration_score: float,
        discovery_score: float,
        safety_score: float,
        efficiency_score: float,
        ground_truth: ProblemGroundTruth
    ) -> str:
        """Generate human-readable feedback for the command."""
        feedback_parts = []
        
        if total_score >= 0.8:
            feedback_parts.append("Excellent command choice!")
        elif total_score >= 0.6:
            feedback_parts.append("Good command.")
        elif total_score >= 0.4:
            feedback_parts.append("Reasonable command.")
        else:
            feedback_parts.append("Command could be improved.")
        
        if critical_path_score >= 0.5:
            feedback_parts.append("This command aligns well with the critical path to solving this problem.")
        
        if exploration_score >= 0.5:
            feedback_parts.append("Good exploration command for gathering information.")
        
        if discovery_score >= 0.3:
            feedback_parts.append("This command revealed relevant information.")
        
        if safety_score < 1.0:
            feedback_parts.append("⚠️ Safety concern: Be careful with potentially disruptive commands.")
        
        if efficiency_score < 0.7:
            feedback_parts.append("Consider avoiding redundant commands for better efficiency.")
        
        return " ".join(feedback_parts)
    
    def _determine_category(
        self,
        command: str,
        critical_path_score: float,
        exploration_score: float
    ) -> str:
        """Determine the category of the interaction."""
        if critical_path_score >= 0.5:
            return "critical_path"
        elif exploration_score >= 0.5:
            return "exploration"
        elif "kubectl" in command.lower():
            return "monitoring"
        else:
            return "other"
    
    def _generate_reasoning(
        self,
        critical_path_score: float,
        exploration_score: float,
        discovery_score: float,
        safety_score: float,
        efficiency_score: float,
        ground_truth: ProblemGroundTruth
    ) -> str:
        """Generate detailed reasoning for the evaluation."""
        reasoning = f"Heuristic evaluation for {ground_truth.problem_id}: "
        reasoning += f"Critical path alignment: {critical_path_score:.2f}, "
        reasoning += f"Exploration quality: {exploration_score:.2f}, "
        reasoning += f"Discovery value: {discovery_score:.2f}, "
        reasoning += f"Safety: {safety_score:.2f}, "
        reasoning += f"Efficiency: {efficiency_score:.2f}."
        
        return reasoning
    
    def reset_task_history(self, task_id: UUID):
        """Reset interaction history for a task (e.g., when task completes)."""
        if task_id in self.interaction_history:
            del self.interaction_history[task_id]
    
    def get_task_statistics(self, task_id: UUID) -> Dict[str, Any]:
        """Get statistics for a task's interaction history."""
        if task_id not in self.interaction_history:
            return {}
        
        history = self.interaction_history[task_id]
        return {
            'total_commands': len(history['commands']),
            'critical_hits': history['critical_hits'],
            'discoveries_made': len(history['discoveries']),
            'resources_explored': len(history['resources_explored']),
            'redundant_commands': history['redundant_commands']
        }