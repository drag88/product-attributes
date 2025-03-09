#!/usr/bin/env python3
"""
Script to view and analyze log files with support for enhanced logging features.
"""
import argparse
import os
import sys
import json
from pathlib import Path
import re
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


def list_log_files(log_dir: Path) -> list:
    """List all log files in the directory."""
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return []
    
    log_files = sorted(
        [f for f in log_dir.glob("*.log")],
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    
    return log_files


def print_log_list(log_files: list):
    """Print a list of log files with metadata."""
    if not log_files:
        print("No log files found.")
        return
    
    print(f"\nFound {len(log_files)} log files:\n")
    print(f"{'#':<3} {'Date':<12} {'Time':<10} {'App':<20} {'Size':<10} {'Path'}")
    print("-" * 80)
    
    for i, log_file in enumerate(log_files):
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
        date_str = mtime.strftime("%Y-%m-%d")
        time_str = mtime.strftime("%H:%M:%S")
        
        # Extract app name from filename
        filename = log_file.name
        app_name = filename.split('_')[0] if '_' in filename else filename
        
        size = os.path.getsize(log_file)
        size_str = f"{size/1024:.1f} KB" if size >= 1024 else f"{size} B"
        
        print(f"{i:<3} {date_str:<12} {time_str:<10} {app_name:<20} {size_str:<10} {log_file}")


def is_json_log(line: str) -> bool:
    """Check if a log line is in JSON format."""
    try:
        json.loads(line)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a log line into a structured format."""
    # Try parsing as JSON first
    if is_json_log(line):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fall back to regex parsing for standard format
    # Example: 2023-02-15 14:30:45,123 - root - INFO - Starting product attribute generation
    standard_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
    match = re.match(standard_pattern, line)
    
    if match:
        timestamp, logger_name, level, message = match.groups()
        return {
            "timestamp": timestamp,
            "name": logger_name,
            "level": level,
            "message": message.strip()
        }
    
    return None


def view_log(
    log_file: Path, 
    tail: Optional[int] = None, 
    grep: Optional[str] = None, 
    level: Optional[str] = None,
    json_format: bool = False
):
    """View log file contents with filtering options."""
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"\nViewing log file: {log_file}\n")
    
    # Read the file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Apply tail if specified
    if tail:
        lines = lines[-tail:]
    
    # Process each line
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse the line
        parsed = parse_log_line(line)
        if not parsed:
            # If we can't parse, include the line as-is if no filters
            if not grep and not level:
                filtered_lines.append(line)
            continue
        
        # Apply grep filter
        if grep and grep.lower() not in line.lower():
            continue
        
        # Apply level filter
        if level and parsed.get("level", "").upper() != level.upper():
            continue
        
        # Format the output
        if json_format:
            filtered_lines.append(json.dumps(parsed, indent=2))
        else:
            filtered_lines.append(line)
    
    # Print the filtered lines
    if filtered_lines:
        for line in filtered_lines:
            print(line)
    else:
        print("No matching log entries found.")


def extract_performance_metrics(log_file: Path) -> Dict[str, Any]:
    """Extract performance metrics from a log file."""
    metrics = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as JSON
            if is_json_log(line):
                try:
                    data = json.loads(line)
                    # Look for performance metrics
                    if "extra" in data and "performance" in data["extra"]:
                        metrics = data["extra"]["performance"]
                        break
                except (json.JSONDecodeError, ValueError):
                    continue
            
            # Try regex for standard format
            if "Performance metrics" in line:
                # Try to find the next line with the metrics
                metrics_line = next(f, "").strip()
                if metrics_line:
                    try:
                        # Extract JSON-like part
                        metrics_match = re.search(r'performance=(\{.+\})', metrics_line)
                        if metrics_match:
                            metrics_str = metrics_match.group(1)
                            # Convert to proper JSON and parse
                            metrics_str = metrics_str.replace("'", '"')
                            metrics = json.loads(metrics_str)
                    except Exception:
                        continue
    
    return metrics


def visualize_performance(log_file: Path):
    """Visualize performance metrics from a log file."""
    metrics = extract_performance_metrics(log_file)
    
    if not metrics:
        print("No performance metrics found in the log file.")
        return
    
    # Create a DataFrame for visualization
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Duration'])
    df = df.sort_values('Duration', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.barh(df['Metric'], df['Duration'], color='skyblue')
    plt.xlabel('Duration (seconds)')
    plt.title(f'Performance Metrics - {log_file.name}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add duration values at the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}s', va='center')
    
    # Save the plot
    output_file = log_file.parent / f"{log_file.stem}_performance.png"
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Performance visualization saved to: {output_file}")
    
    # Print summary
    print("\nPerformance Summary:")
    print("-" * 40)
    for metric, duration in metrics.items():
        print(f"{metric:<30} {duration:.2f}s")


def search_all_logs(
    log_files: list, 
    pattern: str, 
    level: Optional[str] = None,
    json_format: bool = False
):
    """Search for a pattern across all log files."""
    if not log_files:
        print("No log files to search.")
        return
    
    print(f"\nSearching for '{pattern}' across {len(log_files)} log files:\n")
    
    results = []
    
    for log_file in log_files:
        file_results = []
        
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse the line
                parsed = parse_log_line(line)
                
                # Apply filters
                if pattern.lower() not in line.lower():
                    continue
                
                if level and parsed and parsed.get("level", "").upper() != level.upper():
                    continue
                
                file_results.append((line_num, line, parsed))
        
        if file_results:
            results.append((log_file, file_results))
    
    # Print results
    if not results:
        print("No matches found.")
        return
    
    for log_file, file_results in results:
        print(f"\n{log_file} ({len(file_results)} matches):")
        print("-" * 80)
        
        for line_num, line, parsed in file_results:
            if json_format and parsed:
                print(f"Line {line_num}: {json.dumps(parsed, indent=2)}")
            else:
                print(f"Line {line_num}: {line}")
        
        print()


def analyze_log_levels(log_file: Path) -> Dict[str, int]:
    """Analyze log levels distribution in a log file."""
    level_counts = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parsed = parse_log_line(line)
            if parsed and "level" in parsed:
                level = parsed["level"].upper()
                level_counts[level] = level_counts.get(level, 0) + 1
    
    return level_counts


def visualize_log_levels(log_file: Path):
    """Visualize log level distribution from a log file."""
    level_counts = analyze_log_levels(log_file)
    
    if not level_counts:
        print("No log levels found in the log file.")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Define colors for different log levels
    colors = {
        'DEBUG': 'lightblue',
        'INFO': 'green',
        'WARNING': 'orange',
        'ERROR': 'red',
        'CRITICAL': 'darkred'
    }
    
    # Create the pie chart
    labels = list(level_counts.keys())
    sizes = list(level_counts.values())
    
    # Get colors for each level, defaulting to gray if not in our map
    level_colors = [colors.get(level, 'gray') for level in labels]
    
    plt.pie(sizes, labels=labels, colors=level_colors, autopct='%1.1f%%', 
            startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Log Level Distribution - {log_file.name}')
    
    # Save the plot
    output_file = log_file.parent / f"{log_file.stem}_levels.png"
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Log level visualization saved to: {output_file}")
    
    # Print summary
    total = sum(level_counts.values())
    print("\nLog Level Summary:")
    print("-" * 40)
    for level, count in level_counts.items():
        print(f"{level:<10} {count:>6} ({count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="View and analyze log files")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List logs command
    list_parser = subparsers.add_parser("list", help="List available log files")
    list_parser.add_argument("--dir", type=str, default="logs", help="Log directory")
    
    # View log command
    view_parser = subparsers.add_parser("view", help="View a log file")
    view_parser.add_argument("log_file", type=str, help="Log file to view (number or path)")
    view_parser.add_argument("--dir", type=str, default="logs", help="Log directory")
    view_parser.add_argument("--tail", type=int, help="Show only the last N lines")
    view_parser.add_argument("--grep", type=str, help="Filter lines containing pattern")
    view_parser.add_argument("--level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                            help="Filter by log level")
    view_parser.add_argument("--json", action="store_true", help="Format output as JSON")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search across all log files")
    search_parser.add_argument("pattern", type=str, help="Pattern to search for")
    search_parser.add_argument("--dir", type=str, default="logs", help="Log directory")
    search_parser.add_argument("--level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                              help="Filter by log level")
    search_parser.add_argument("--json", action="store_true", help="Format output as JSON")
    
    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Visualize performance metrics")
    perf_parser.add_argument("log_file", type=str, help="Log file to analyze (number or path)")
    perf_parser.add_argument("--dir", type=str, default="logs", help="Log directory")
    
    # Levels command
    levels_parser = subparsers.add_parser("levels", help="Visualize log level distribution")
    levels_parser.add_argument("log_file", type=str, help="Log file to analyze (number or path)")
    levels_parser.add_argument("--dir", type=str, default="logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Default to list if no command specified
    if not args.command:
        args.command = "list"
        args.dir = "logs"
    
    # Get log directory
    log_dir = Path(args.dir)
    
    # List command
    if args.command == "list":
        log_files = list_log_files(log_dir)
        print_log_list(log_files)
    
    # View command
    elif args.command == "view":
        log_files = list_log_files(log_dir)
        
        # Determine which log file to view
        log_file = None
        try:
            # If input is a number, use it as an index
            index = int(args.log_file)
            if 0 <= index < len(log_files):
                log_file = log_files[index]
            else:
                print(f"Invalid log file index: {index}")
                return
        except ValueError:
            # If input is not a number, treat it as a path
            log_file = Path(args.log_file)
        
        if log_file:
            view_log(
                log_file, 
                tail=args.tail, 
                grep=args.grep, 
                level=args.level,
                json_format=args.json
            )
    
    # Search command
    elif args.command == "search":
        log_files = list_log_files(log_dir)
        search_all_logs(
            log_files, 
            args.pattern, 
            level=args.level,
            json_format=args.json
        )
    
    # Performance command
    elif args.command == "performance":
        log_files = list_log_files(log_dir)
        
        # Determine which log file to analyze
        log_file = None
        try:
            # If input is a number, use it as an index
            index = int(args.log_file)
            if 0 <= index < len(log_files):
                log_file = log_files[index]
            else:
                print(f"Invalid log file index: {index}")
                return
        except ValueError:
            # If input is not a number, treat it as a path
            log_file = Path(args.log_file)
        
        if log_file:
            visualize_performance(log_file)
    
    # Levels command
    elif args.command == "levels":
        log_files = list_log_files(log_dir)
        
        # Determine which log file to analyze
        log_file = None
        try:
            # If input is a number, use it as an index
            index = int(args.log_file)
            if 0 <= index < len(log_files):
                log_file = log_files[index]
            else:
                print(f"Invalid log file index: {index}")
                return
        except ValueError:
            # If input is not a number, treat it as a path
            log_file = Path(args.log_file)
        
        if log_file:
            visualize_log_levels(log_file)


if __name__ == "__main__":
    main() 