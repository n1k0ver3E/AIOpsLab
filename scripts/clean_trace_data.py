#!/usr/bin/env python3
"""
Data cleaning script for extracting successful command-response pairs from JSON traces.
This script processes AI SRE trace data to create a clean dataset for heuristic RL rewards.

Usage: python clean_trace_data.py <json_file_path>
Output: Cleaned command-response pairs from successful traces
"""

import json
import sys
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class TraceDataCleaner:
    """Cleans and extracts meaningful command-response pairs from trace data."""
    
    def __init__(self):
        # Patterns to identify noise in responses
        self.noise_patterns = [
            r"Error parsing response: No API call found!",
            r"^$",  # Empty responses
            r"^\s*$",  # Whitespace-only responses
            r"Error parsing response:",
            r"Invalid API call format",
        ]
        
        # Patterns to identify high-value commands (diagnostic/fixing)
        self.valuable_command_patterns = [
            r"kubectl\s+get\s+",
            r"kubectl\s+describe\s+",
            r"kubectl\s+logs\s+",
            r"kubectl\s+scale\s+",
            r"kubectl\s+delete\s+",
            r"kubectl\s+apply\s+",
            r"kubectl\s+patch\s+",
            r"kubectl\s+restart\s+",
            r"docker\s+",
            r"systemctl\s+",
            r"service\s+",
            r"ps\s+aux",
            r"netstat\s+",
            r"curl\s+",
            r"ping\s+",
            r"nslookup\s+",
            r"dig\s+",
        ]
    
    def is_successful_trace(self, trace_data: Dict) -> bool:
        """Check if the trace represents a successful problem resolution."""
        # Handle case where trace_data might not be a dict
        if not isinstance(trace_data, dict):
            return False
            
        # Check for explicit success indicators
        results = trace_data.get("results", {})
        if isinstance(results, dict):
            if "supervisor_result" in results:
                return results["supervisor_result"] == "Correct"
            
            if "success" in results:
                return results["success"] is True
        
        # Check top-level success field (some files have it there)
        if "success" in trace_data:
            return trace_data["success"] is True
            
        return False
    
    def is_noise_response(self, response: str) -> bool:
        """Check if a response is noise/error that should be filtered out."""
        if not response or not response.strip():
            return True
            
        for pattern in self.noise_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        return False
    
    def is_valuable_command(self, command: str) -> bool:
        """Check if a command is likely to be valuable for problem-solving."""
        if not command or not command.strip():
            return False
            
        # Extract command from exec_shell() wrapper if present
        command_match = re.search(r'exec_shell\("([^"]+)"\)', command)
        if command_match:
            actual_command = command_match.group(1)
        else:
            actual_command = command
        
        # Check against valuable command patterns
        for pattern in self.valuable_command_patterns:
            if re.search(pattern, actual_command, re.IGNORECASE):
                return True
        
        return False
    
    def extract_command_from_content(self, content: str) -> Optional[str]:
        """Extract the actual shell command from assistant content."""
        if not content:
            return None
            
        # Look for exec_shell() calls
        exec_shell_match = re.search(r'exec_shell\("([^"]+)"\)', content)
        if exec_shell_match:
            return exec_shell_match.group(1)
        
        # Look for code blocks with shell commands
        code_block_match = re.search(r'```(?:bash|shell)?\n(.+?)\n```', content, re.DOTALL)
        if code_block_match:
            command = code_block_match.group(1).strip()
            # If it's an exec_shell call, extract the inner command
            exec_match = re.search(r'exec_shell\("([^"]+)"\)', command)
            if exec_match:
                return exec_match.group(1)
            return command
        
        return None
    
    def calculate_response_value_score(self, response: str) -> float:
        """Calculate a value score for a response based on information content."""
        if self.is_noise_response(response):
            return 0.0
        
        score = 0.5  # Base score for non-noise responses
        
        # Bonus for diagnostic information
        diagnostic_indicators = [
            r"STATUS|READY|RESTARTS",  # Pod status info
            r"Error|Failed|CrashLoopBackOff|Pending",  # Error states
            r"Authentication|Authorization|Permission",  # Auth issues
            r"Connection\s+refused|timeout|unreachable",  # Network issues
            r"mongodb|database|connection",  # Database issues
            r"not\s+authorized|access\s+denied",  # Permission issues
            r"Exit\s+Code:\s+[1-9]",  # Non-zero exit codes
            r"Events:|Warning:|Error:",  # Kubernetes events
        ]
        
        for pattern in diagnostic_indicators:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.1
        
        # Bonus for longer, more informative responses
        if len(response) > 500:
            score += 0.1
        elif len(response) > 200:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_command_response_pairs(self, trace: List[Dict]) -> List[Tuple[str, str, float]]:
        """Extract meaningful command-response pairs from trace."""
        pairs = []
        
        for i in range(len(trace) - 1):
            current_item = trace[i]
            next_item = trace[i + 1]
            
            # Look for assistant -> env pairs
            if (current_item.get("role") == "assistant" and 
                next_item.get("role") == "env"):
                
                command = self.extract_command_from_content(current_item.get("content", ""))
                response = next_item.get("content", "")
                
                if command and self.is_valuable_command(command) and not self.is_noise_response(response):
                    value_score = self.calculate_response_value_score(response)
                    pairs.append((command, response, value_score))
        
        return pairs
    
    def clean_trace_file(self, file_path: str) -> Optional[Dict]:
        """Clean a single trace file and extract meaningful pairs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error reading {file_path}: {e}")
            return None
        
        # Skip files that are not valid trace data (like our cleaned output files)
        if not isinstance(data, dict) or "trace" not in data:
            return None
        
        # Check if this is a successful trace
        if not self.is_successful_trace(data):
            print(f"Skipping {file_path}: Not a successful trace")
            return None
        
        # Extract basic information
        problem_id = data.get("problem_id", "unknown")
        trace = data.get("trace", [])
        
        if not trace:
            print(f"Skipping {file_path}: No trace data found")
            return None
        
        # Extract command-response pairs
        pairs = self.extract_command_response_pairs(trace)
        
        if not pairs:
            print(f"Skipping {file_path}: No valuable command-response pairs found")
            return None
        
        # Filter pairs by value score (keep only high-value pairs)
        high_value_pairs = [(cmd, resp) for cmd, resp, score in pairs if score >= 0.6]
        
        if not high_value_pairs:
            # If no high-value pairs, keep the top-scoring ones
            pairs.sort(key=lambda x: x[2], reverse=True)
            high_value_pairs = [(cmd, resp) for cmd, resp, score in pairs[:3]]  # Top 3
        
        return {
            "problem_id": problem_id,
            "file_path": file_path,
            "command_response_pairs": high_value_pairs,
            "total_pairs_extracted": len(pairs),
            "high_value_pairs_count": len(high_value_pairs)
        }


def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_trace_data.py <json_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    cleaner = TraceDataCleaner()
    result = cleaner.clean_trace_file(file_path)
    
    if result:
        print(f"\n=== Cleaned Data for {result['problem_id']} ===")
        print(f"File: {result['file_path']}")
        print(f"Total pairs extracted: {result['total_pairs_extracted']}")
        print(f"High-value pairs: {result['high_value_pairs_count']}")
        print("\n=== Command-Response Pairs ===")
        
        for i, (command, response) in enumerate(result['command_response_pairs'], 1):
            print(f"\n--- Pair {i} ---")
            print(f"Command: {command}")
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            print("-" * 50)
        
        # Save cleaned data to JSON file
        output_file = file_path.replace('.json', '_cleaned.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nCleaned data saved to: {output_file}")
    else:
        print("No meaningful data extracted from the trace file.")


if __name__ == "__main__":
    main()
