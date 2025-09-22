"""
Test examples and validation for the heuristic reward function.

This module provides comprehensive test cases to demonstrate and validate
the ground truth-based heuristic reward system for different AIOps scenarios.
"""

import json
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any

from .heuristic_reward import HeuristicRewardFunction, ProblemGroundTruth

# Simple mock classes for testing
class EnvironmentResponse:
    def __init__(self, execution_output=None, metrics=None, logs=None, system_state=None, kubernetes_status=None):
        self.execution_output = execution_output or {}
        self.metrics = metrics or {}
        self.logs = logs or []
        self.system_state = system_state or {}
        self.kubernetes_status = kubernetes_status or {}


class HeuristicRewardTester:
    """Test suite for validating the heuristic reward function."""
    
    def __init__(self):
        self.reward_function = HeuristicRewardFunction()
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite and return results."""
        print("🧪 Running Heuristic Reward Function Tests...")
        print("=" * 60)
        
        results = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_results": {}
        }
        
        # Test different problem types
        test_cases = [
            self._test_k8s_target_port_misconfig(),
            self._test_pod_failure_scenario(),
            self._test_network_issues(),
            self._test_safety_mechanisms(),
            self._test_exploration_rewards(),
            self._test_efficiency_penalties(),
            self._test_discovery_rewards(),
            self._test_hotel_reservation_mitigation()
        ]
        
        for test_case in test_cases:
            test_name = test_case["test_name"]
            results["test_results"][test_name] = test_case
            
            print(f"\n📋 Test: {test_name}")
            print(f"   Status: {'✅ PASS' if test_case['passed'] else '❌ FAIL'}")
            print(f"   Average Score: {test_case['average_score']:.3f}")
            if test_case.get('issues'):
                print(f"   Issues: {test_case['issues']}")
        
        # Calculate overall results
        total_tests = len(test_cases)
        passed_tests = sum(1 for tc in test_cases if tc['passed'])
        overall_score = sum(tc['average_score'] for tc in test_cases) / total_tests
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "overall_average_score": overall_score
        }
        
        print(f"\n📊 Overall Results:")
        print(f"   Tests Passed: {passed_tests}/{total_tests} ({results['summary']['pass_rate']:.1%})")
        print(f"   Overall Average Score: {overall_score:.3f}")
        
        return results
    
    def _test_k8s_target_port_misconfig(self) -> Dict[str, Any]:
        """Test K8s target port misconfiguration scenario."""
        problem_id = "k8s_target_port-misconfig-detection-1"
        task_id = uuid4()
        
        test_commands = [
            # Critical path commands (should get high scores)
            {
                "command": "kubectl get svc user-service",
                "exit_code": 0,
                "stdout": "NAME           TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE\nuser-service   ClusterIP   10.96.123.45   <none>        8080:9090/TCP   5m",
                "expected_score_range": (0.7, 1.0),
                "description": "Critical path: checking target service"
            },
            {
                "command": "kubectl describe svc user-service", 
                "exit_code": 0,
                "stdout": "Name: user-service\nTargetPort: 8080\nEndpoints: 10.244.1.5:9090",
                "expected_score_range": (0.8, 1.0),
                "description": "Critical path: detailed service inspection"
            },
            {
                "command": "kubectl get pods -l app=user-service",
                "exit_code": 0, 
                "stdout": "NAME                   READY   STATUS             RESTARTS   AGE\nuser-service-abc123    0/1     CrashLoopBackOff   5          10m",
                "expected_score_range": (0.7, 1.0),
                "description": "Critical discovery: finding CrashLoopBackOff"
            },
            # Good exploration commands
            {
                "command": "kubectl logs user-service-abc123",
                "exit_code": 0,
                "stdout": "Error: connection refused on port 8080",
                "expected_score_range": (0.6, 0.9),
                "description": "Good exploration: checking logs"
            },
            # Less relevant but safe commands
            {
                "command": "kubectl get nodes",
                "exit_code": 0,
                "stdout": "NAME     STATUS   ROLES    AGE   VERSION\nnode1    Ready    master   1d    v1.21.0",
                "expected_score_range": (0.3, 0.6),
                "description": "Safe but less relevant exploration"
            }
        ]
        
        return self._run_test_scenario(
            "K8s Target Port Misconfiguration",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_pod_failure_scenario(self) -> Dict[str, Any]:
        """Test pod failure/kill scenario."""
        problem_id = "pod_kill-detection-1"
        task_id = uuid4()
        
        test_commands = [
            {
                "command": "kubectl get pods",
                "exit_code": 0,
                "stdout": "NAME        READY   STATUS      RESTARTS   AGE\napp-pod     0/1     Terminated  0          5m",
                "expected_score_range": (0.8, 1.0),
                "description": "Critical discovery: finding terminated pod"
            },
            {
                "command": "kubectl describe pod app-pod",
                "exit_code": 0,
                "stdout": "State: Terminated\nReason: Killed\nExit Code: 137",
                "expected_score_range": (0.8, 1.0),
                "description": "Critical path: detailed pod inspection"
            },
            {
                "command": "kubectl get events",
                "exit_code": 0,
                "stdout": "LAST SEEN   TYPE      REASON    OBJECT      MESSAGE\n2m          Warning   Killing   Pod/app-pod Container killed",
                "expected_score_range": (0.7, 0.9),
                "description": "Good exploration: checking events"
            }
        ]
        
        return self._run_test_scenario(
            "Pod Failure Detection",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_network_issues(self) -> Dict[str, Any]:
        """Test network-related problems."""
        problem_id = "network_delay-detection-1"
        task_id = uuid4()
        
        test_commands = [
            {
                "command": "kubectl get svc",
                "exit_code": 0,
                "stdout": "NAME        TYPE        CLUSTER-IP     PORT(S)\napi-svc     ClusterIP   10.96.1.100    80/TCP",
                "expected_score_range": (0.6, 0.9),
                "description": "Relevant network exploration"
            },
            {
                "command": "kubectl get endpoints api-svc",
                "exit_code": 0,
                "stdout": "NAME      ENDPOINTS        AGE\napi-svc   10.244.1.5:80    5m",
                "expected_score_range": (0.7, 0.9),
                "description": "Critical path: checking endpoints"
            },
            {
                "command": "ping 10.244.1.5",
                "exit_code": 1,
                "stdout": "",
                "stderr": "ping: connect: Network is unreachable",
                "expected_score_range": (0.5, 0.8),
                "description": "Network debugging attempt"
            }
        ]
        
        return self._run_test_scenario(
            "Network Issues Detection",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_safety_mechanisms(self) -> Dict[str, Any]:
        """Test safety mechanisms for dangerous commands."""
        problem_id = "k8s_target_port-misconfig-mitigation-1"
        task_id = uuid4()
        
        test_commands = [
            # Dangerous commands (should get very low scores)
            {
                "command": "rm -rf /",
                "exit_code": 1,
                "stdout": "",
                "stderr": "Permission denied",
                "expected_score_range": (0.0, 0.1),
                "description": "Extremely dangerous command"
            },
            {
                "command": "sudo rm -rf /var/lib/docker",
                "exit_code": 1,
                "stdout": "",
                "stderr": "Permission denied", 
                "expected_score_range": (0.0, 0.1),
                "description": "Dangerous system deletion"
            },
            # Disruptive but sometimes necessary commands
            {
                "command": "kubectl delete pod app-pod",
                "exit_code": 0,
                "stdout": "pod 'app-pod' deleted",
                "expected_score_range": (0.2, 0.4),
                "description": "Disruptive but potentially necessary"
            },
            # Safe commands
            {
                "command": "kubectl get pods",
                "exit_code": 0,
                "stdout": "No resources found",
                "expected_score_range": (0.6, 1.0),
                "description": "Safe exploration command"
            }
        ]
        
        return self._run_test_scenario(
            "Safety Mechanisms",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_exploration_rewards(self) -> Dict[str, Any]:
        """Test exploration reward mechanisms."""
        problem_id = "misconfig_app_hotel_res-detection-1"
        task_id = uuid4()
        
        test_commands = [
            # Different types of exploration
            {
                "command": "kubectl get pods -o wide",
                "exit_code": 0,
                "stdout": "NAME    READY   STATUS    IP           NODE\napp     1/1     Running   10.244.1.5   node1",
                "expected_score_range": (0.6, 0.9),
                "description": "Kubernetes exploration with details"
            },
            {
                "command": "top",
                "exit_code": 0,
                "stdout": "PID USER %CPU %MEM COMMAND\n1234 root 50.0 10.0 app-server",
                "expected_score_range": (0.4, 0.7),
                "description": "System monitoring exploration"
            },
            {
                "command": "netstat -tulpn",
                "exit_code": 0,
                "stdout": "Proto Local Address State PID/Program\ntcp 0.0.0.0:8080 LISTEN 1234/app-server",
                "expected_score_range": (0.4, 0.7),
                "description": "Network monitoring exploration"
            },
            {
                "command": "cat /var/log/app.log",
                "exit_code": 0,
                "stdout": "2023-01-01 12:00:00 ERROR: Database connection failed",
                "expected_score_range": (0.3, 0.6),
                "description": "Log file inspection"
            }
        ]
        
        return self._run_test_scenario(
            "Exploration Rewards",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_efficiency_penalties(self) -> Dict[str, Any]:
        """Test efficiency penalties for redundant commands."""
        problem_id = "pod_failure-localization-1"
        task_id = uuid4()
        
        # Simulate redundant command sequence
        test_commands = [
            {
                "command": "kubectl get pods",
                "exit_code": 0,
                "stdout": "NAME    READY   STATUS\napp     1/1     Running",
                "expected_score_range": (0.6, 0.9),
                "description": "First execution - good score"
            },
            {
                "command": "kubectl get pods",  # Exact same command
                "exit_code": 0,
                "stdout": "NAME    READY   STATUS\napp     1/1     Running", 
                "expected_score_range": (0.4, 0.7),
                "description": "Second execution - efficiency penalty"
            },
            {
                "command": "kubectl get pod",  # Very similar command
                "exit_code": 0,
                "stdout": "NAME    READY   STATUS\napp     1/1     Running",
                "expected_score_range": (0.3, 0.6),
                "description": "Similar command - more penalty"
            }
        ]
        
        return self._run_test_scenario(
            "Efficiency Penalties",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_discovery_rewards(self) -> Dict[str, Any]:
        """Test discovery rewards for finding relevant information."""
        problem_id = "k8s_target_port-misconfig-analysis-1"
        task_id = uuid4()
        
        test_commands = [
            # Commands that discover expected information
            {
                "command": "kubectl describe svc user-service",
                "exit_code": 0,
                "stdout": "TargetPort: 8080\nPort: 9090\nEndpoints: none",
                "expected_score_range": (0.8, 1.0),
                "description": "Discovery: targetPort mismatch found"
            },
            {
                "command": "kubectl get pods -o yaml",
                "exit_code": 0,
                "stdout": "containerPort: 9090\ntargetPort: 8080\nconnection refused",
                "expected_score_range": (0.7, 1.0),
                "description": "Discovery: multiple relevant findings"
            },
            # Commands with no relevant discoveries
            {
                "command": "kubectl get configmap",
                "exit_code": 0,
                "stdout": "NAME         DATA   AGE\napp-config   1      5m",
                "expected_score_range": (0.3, 0.6),
                "description": "No relevant discoveries"
            }
        ]
        
        return self._run_test_scenario(
            "Discovery Rewards",
            problem_id,
            task_id,
            test_commands
        )
    
    def _test_hotel_reservation_mitigation(self) -> Dict[str, Any]:
        """Test hotel reservation app misconfiguration scenario based on sample data."""
        problem_id = "misconfig_app_hotel_res-mitigation-1"
        task_id = uuid4()
        
        test_commands = [
            # Step 1: Discover failing pod
            {
                "command": "kubectl get pods -n test-hotel-reservation",
                "exit_code": 0,
                "stdout": "geo-c47ff745-nd9q2    0/1     Error     1 (15s ago)   26s",
                "expected_score_range": (0.7, 0.9),
                "description": "Critical discovery: finding failing geo pod"
            },
            # Step 2: Investigate pod details
            {
                "command": "kubectl describe pod geo-c47ff745-nd9q2 -n test-hotel-reservation",
                "exit_code": 0,
                "stdout": "Status: Running\nContainer: hotel-reserv-geo\nImage: geo-service",
                "expected_score_range": (0.7, 0.9),
                "description": "Critical path: detailed pod inspection"
            },
            # Step 3: Check logs - KEY DISCOVERY
            {
                "command": "kubectl logs -n test-hotel-reservation geo-c47ff745-nd9q2 -c hotel-reserv-geo --previous",
                "exit_code": 0,
                "stdout": "Read database URL: mongodb-geo:27777\nno reachable servers\npanic: no reachable servers",
                "expected_score_range": (0.8, 1.0),
                "description": "Critical discovery: database connection error with port 27777"
            },
            # Step 4: Check service configuration
            {
                "command": "kubectl get svc mongodb-geo -n test-hotel-reservation",
                "exit_code": 0,
                "stdout": "mongodb-geo   ClusterIP   10.96.1.100   <none>   27017/TCP   5m",
                "expected_score_range": (0.8, 1.0),
                "description": "Critical path: checking MongoDB service ports"
            },
            # Step 5: THE SOLUTION - Patch service to add port 27777
            {
                "command": "kubectl patch svc mongodb-geo -n test-hotel-reservation --type='merge' -p '{\"spec\":{\"ports\":[{\"name\":\"mongo\",\"port\":27017,\"targetPort\":27017,\"protocol\":\"TCP\"},{\"name\":\"mongo-compat\",\"port\":27777,\"targetPort\":27017,\"protocol\":\"TCP\"}]}}'",
                "exit_code": 0,
                "stdout": "service/mongodb-geo patched",
                "expected_score_range": (0.9, 1.0),
                "description": "SOLUTION: Patching service to add port 27777 mapping"
            },
            # Step 6: Verify fix
            {
                "command": "kubectl get pods -n test-hotel-reservation",
                "exit_code": 0,
                "stdout": "geo-c47ff745-nd9q2    1/1     Running   5 (2m54s ago)   5m5s",
                "expected_score_range": (0.6, 0.8),
                "description": "Verification: checking pod status after fix"
            },
            # Less relevant commands should get lower scores
            {
                "command": "kubectl get configmaps -n test-hotel-reservation",
                "exit_code": 0,
                "stdout": "admin-config   1   5m",
                "expected_score_range": (0.3, 0.5),
                "description": "Less relevant: checking configmaps"
            }
        ]
        
        return self._run_test_scenario(
            "Hotel Reservation Mitigation",
            problem_id,
            task_id,
            test_commands
        )
    
    def _run_test_scenario(
        self,
        test_name: str,
        problem_id: str,
        task_id,
        test_commands: list
    ) -> Dict[str, Any]:
        """Run a test scenario with multiple commands."""
        results = {
            "test_name": test_name,
            "problem_id": problem_id,
            "task_id": str(task_id),
            "command_results": [],
            "passed": True,
            "issues": [],
            "average_score": 0.0
        }
        
        total_score = 0.0
        
        for i, test_cmd in enumerate(test_commands):
            # Create environment response
            env_response = EnvironmentResponse(
                execution_output={
                    "exit_code": test_cmd["exit_code"],
                    "stdout": test_cmd.get("stdout", ""),
                    "stderr": test_cmd.get("stderr", ""),
                    "command": test_cmd["command"]
                },
                metrics={"test": True},
                logs=[{"message": "test log"}],
                system_state={"test": True},
                kubernetes_status={"test": True}
            )
            
            # Calculate reward
            evaluation = self.reward_function.calculate_reward(
                task_id=task_id,
                problem_id=problem_id,
                command=test_cmd["command"],
                env_response=env_response,
                step_number=i + 1,
                context={}
            )
            
            # Check if score is in expected range
            min_expected, max_expected = test_cmd["expected_score_range"]
            score_in_range = min_expected <= evaluation.score <= max_expected
            
            if not score_in_range:
                results["passed"] = False
                results["issues"].append(
                    f"Command '{test_cmd['command']}' scored {evaluation.score:.3f}, "
                    f"expected {min_expected:.3f}-{max_expected:.3f}"
                )
            
            command_result = {
                "command": test_cmd["command"],
                "description": test_cmd["description"],
                "expected_range": test_cmd["expected_score_range"],
                "actual_score": evaluation.score,
                "feedback": evaluation.feedback,
                "category": evaluation.category,
                "score_in_range": score_in_range
            }
            
            results["command_results"].append(command_result)
            total_score += evaluation.score
            
            print(f"    {i+1}. {test_cmd['description']}")
            print(f"       Command: {test_cmd['command']}")
            print(f"       Score: {evaluation.score:.3f} (expected: {min_expected:.3f}-{max_expected:.3f}) {'✅' if score_in_range else '❌'}")
            print(f"       Feedback: {evaluation.feedback}")
        
        results["average_score"] = total_score / len(test_commands)
        return results


def demo_reward_function():
    """Run a demonstration of the heuristic reward function."""
    print("🚀 Heuristic Reward Function Demonstration")
    print("=" * 60)
    
    # Create a simple example
    reward_function = HeuristicRewardFunction()
    task_id = uuid4()
    problem_id = "k8s_target_port-misconfig-detection-1"
    
    # Example commands with different characteristics
    examples = [
        {
            "command": "kubectl get svc user-service",
            "description": "Critical path command for target port issue"
        },
        {
            "command": "rm -rf /tmp/test",
            "description": "Potentially dangerous command"
        },
        {
            "command": "kubectl logs random-pod",
            "description": "Good exploration but not critical path"
        },
        {
            "command": "echo 'hello world'",
            "description": "Safe but not useful command"
        }
    ]
    
    print(f"\nProblem: {problem_id}")
    print(f"Task ID: {task_id}")
    print("\nTesting different command types:")
    print("-" * 40)
    
    for i, example in enumerate(examples, 1):
        # Create mock environment response
        env_response = EnvironmentResponse(
            execution_output={
                "exit_code": 0,
                "stdout": "mock output",
                "stderr": "",
                "command": example["command"]
            }
        )
        
        # Calculate reward
        evaluation = reward_function.calculate_reward(
            task_id=task_id,
            problem_id=problem_id,
            command=example["command"],
            env_response=env_response,
            step_number=i,
            context={}
        )
        
        print(f"\n{i}. {example['description']}")
        print(f"   Command: {example['command']}")
        print(f"   Score: {evaluation.score:.3f}")
        print(f"   Category: {evaluation.category}")
        print(f"   Feedback: {evaluation.feedback}")
    
    # Show task statistics
    stats = reward_function.get_task_statistics(task_id)
    print(f"\n📊 Task Statistics:")
    print(f"   Total Commands: {stats.get('total_commands', 0)}")
    print(f"   Critical Hits: {stats.get('critical_hits', 0)}")
    print(f"   Discoveries Made: {stats.get('discoveries_made', 0)}")
    print(f"   Resources Explored: {stats.get('resources_explored', 0)}")


if __name__ == "__main__":
    # Run demonstration
    demo_reward_function()
    
    print("\n" + "=" * 60)
    
    # Run comprehensive tests
    tester = HeuristicRewardTester()
    results = tester.run_comprehensive_tests()
    
    # Save results to file
    with open("/tmp/heuristic_reward_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Test results saved to: /tmp/heuristic_reward_test_results.json")