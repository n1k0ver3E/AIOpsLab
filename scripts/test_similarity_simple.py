#!/usr/bin/env python3
"""
Simple test script for the similarity-based reward calculation.
Tests the core similarity logic without requiring the full service infrastructure.
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Tuple


def calculate_string_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher."""
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def normalize_command(command: str) -> str:
    """Normalize command for better similarity matching."""
    # Remove exec_shell wrapper if present
    command = re.sub(r'exec_shell\("([^"]+)"\)', r'\1', command)
    
    # Normalize whitespace
    command = ' '.join(command.split())
    
    # Remove specific identifiers that might vary (pod names, timestamps, etc.)
    command = re.sub(r'-[a-f0-9]{8,}', '-<id>', command)  # Pod hash suffixes
    command = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '<timestamp>', command)  # Timestamps
    
    return command.lower().strip()


def find_best_similarity_match(command: str, problem_id: str, training_data: list) -> Tuple[float, str]:
    """Find the best similarity match for a command in the training data."""
    if not training_data:
        return 0.0, "No training data available"
    
    normalized_command = normalize_command(command)
    best_score = 0.0
    best_match_info = "No similar commands found"
    
    # First, try to find matches from the same problem type
    same_problem_matches = [
        item for item in training_data 
        if item.get('problem_id', '').startswith(problem_id.split('-')[0]) if problem_id
    ]
    
    # If no same-problem matches, use all training data
    candidates = same_problem_matches if same_problem_matches else training_data
    
    for item in candidates:
        training_command = normalize_command(item.get('command', ''))
        similarity = calculate_string_similarity(normalized_command, training_command)
        
        if similarity > best_score:
            best_score = similarity
            problem_type = item.get('problem_id', 'unknown').split('-')[0]
            best_match_info = f"Similar to '{item.get('command', '')[:50]}...' from {problem_type} (similarity: {similarity:.2f})"
    
    return best_score, best_match_info


def calculate_heuristic_reward(command: str, exit_code: int, problem_id: str, training_data: list) -> Tuple[float, str]:
    """Calculate heuristic reward based on similarity to successful traces."""
    # Get similarity score
    similarity_score, match_info = find_best_similarity_match(command, problem_id, training_data)
    
    # Base reward calculation
    if similarity_score >= 0.8:
        base_reward = 0.9  # Very high similarity
        category = "high_similarity"
    elif similarity_score >= 0.6:
        base_reward = 0.7  # Good similarity
        category = "good_similarity"
    elif similarity_score >= 0.4:
        base_reward = 0.5  # Moderate similarity
        category = "moderate_similarity"
    elif similarity_score >= 0.2:
        base_reward = 0.3  # Low similarity
        category = "low_similarity"
    else:
        base_reward = 0.1  # Very low similarity
        category = "no_similarity"
    
    # Adjust based on execution success
    if exit_code == 0:
        # Successful execution gets a bonus
        final_reward = min(base_reward + 0.1, 1.0)
        execution_info = "Command executed successfully"
    elif exit_code != 0:
        # Failed execution gets a penalty
        final_reward = max(base_reward - 0.2, 0.0)
        execution_info = f"Command failed with exit code {exit_code}"
    else:
        final_reward = base_reward
        execution_info = "Command execution status unclear"
    
    feedback = f"Heuristic reward: {final_reward:.2f}. {match_info}. {execution_info}"
    
    return final_reward, feedback


def main():
    """Test the heuristic reward function."""
    
    # Load training data
    data_path = Path("ai_sre_data/openai_gpt-5/0911/rl_training_dataset.json")
    
    if not data_path.exists():
        print(f"Training data not found at {data_path}")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training examples")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            "command": "kubectl get pods -n test-hotel-reservation",
            "problem_id": "revoke_auth_mongodb-detection-1",
            "exit_code": 0,
            "description": "Exact match from training data"
        },
        {
            "command": "kubectl describe pod geo-xyz123 -n test-hotel-reservation", 
            "problem_id": "revoke_auth_mongodb-detection-1",
            "exit_code": 0,
            "description": "Similar command, different pod name"
        },
        {
            "command": "kubectl get services -n test-social-network",
            "problem_id": "scale_pod_zero_social_net-detection-1",
            "exit_code": 0,
            "description": "Similar kubectl command, different resource"
        },
        {
            "command": "docker ps -a",
            "problem_id": "container_kill-detection",
            "exit_code": 0,
            "description": "Docker command for container problem"
        },
        {
            "command": "ls -la /tmp",
            "problem_id": "random_problem-detection-1",
            "exit_code": 0,
            "description": "Unrelated command"
        },
        {
            "command": "kubectl get pods -n test-hotel-reservation",
            "problem_id": "revoke_auth_mongodb-detection-1", 
            "exit_code": 1,
            "description": "Same command but failed execution"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Command: {test_case['command']}")
        print(f"Problem ID: {test_case['problem_id']}")
        print(f"Exit Code: {test_case['exit_code']}")
        
        reward, feedback = calculate_heuristic_reward(
            test_case['command'],
            test_case['exit_code'],
            test_case['problem_id'],
            training_data
        )
        
        print(f"Reward: {reward:.3f}")
        print(f"Feedback: {feedback}")
        print("-" * 60)
    
    # Show some training data examples
    print(f"\nSample Training Data (first 5 examples):")
    print("=" * 80)
    for i, example in enumerate(training_data[:5]):
        print(f"{i+1}. Problem: {example['problem_id']}")
        print(f"   Command: {example['command']}")
        print(f"   Response: {example['response'][:100]}...")
        print()


if __name__ == "__main__":
    main()
