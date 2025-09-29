#!/usr/bin/env python3
"""
Batch processing script for cleaning multiple trace files.
Processes all JSON files in a directory and creates a consolidated dataset.

Usage: python batch_clean_traces.py <directory_path>
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from clean_trace_data import TraceDataCleaner


def process_directory(directory_path: str) -> Dict:
    """Process all JSON files in a directory."""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return {}
    
    cleaner = TraceDataCleaner()
    results = {
        "successful_traces": [],
        "failed_traces": [],
        "total_files_processed": 0,
        "successful_files": 0,
        "summary": {}
    }
    
    # Find all JSON files
    json_files = list(directory.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        results["total_files_processed"] += 1
        print(f"Processing: {json_file.name}")
        
        cleaned_data = cleaner.clean_trace_file(str(json_file))
        
        if cleaned_data:
            results["successful_traces"].append(cleaned_data)
            results["successful_files"] += 1
            
            # Update summary statistics
            problem_id = cleaned_data["problem_id"]
            if problem_id not in results["summary"]:
                results["summary"][problem_id] = {
                    "count": 0,
                    "total_pairs": 0,
                    "avg_pairs": 0
                }
            
            results["summary"][problem_id]["count"] += 1
            results["summary"][problem_id]["total_pairs"] += cleaned_data["high_value_pairs_count"]
            results["summary"][problem_id]["avg_pairs"] = (
                results["summary"][problem_id]["total_pairs"] / 
                results["summary"][problem_id]["count"]
            )
        else:
            results["failed_traces"].append(str(json_file))
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_clean_traces.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    results = process_directory(directory_path)
    
    if not results:
        sys.exit(1)
    
    # Print summary
    print(f"\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Total files processed: {results['total_files_processed']}")
    print(f"Successful traces extracted: {results['successful_files']}")
    print(f"Failed/skipped traces: {len(results['failed_traces'])}")
    
    print(f"\n=== PROBLEM TYPE SUMMARY ===")
    for problem_id, stats in results["summary"].items():
        print(f"{problem_id}: {stats['count']} traces, {stats['total_pairs']} total pairs, {stats['avg_pairs']:.1f} avg pairs/trace")
    
    # Save consolidated results
    output_file = Path(directory_path) / "consolidated_cleaned_traces.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nConsolidated results saved to: {output_file}")
    
    # Create a simplified dataset for RL training
    rl_dataset = []
    for trace in results["successful_traces"]:
        for command, response in trace["command_response_pairs"]:
            rl_dataset.append({
                "problem_id": trace["problem_id"],
                "command": command,
                "response": response
            })
    
    rl_output_file = Path(directory_path) / "rl_training_dataset.json"
    with open(rl_output_file, 'w', encoding='utf-8') as f:
        json.dump(rl_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"RL training dataset saved to: {rl_output_file}")
    print(f"Dataset contains {len(rl_dataset)} command-response pairs")


if __name__ == "__main__":
    main()
