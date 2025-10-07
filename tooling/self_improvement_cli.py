import argparse
import json
from collections import defaultdict

LOG_FILE_PATH = 'logs/activity.log.jsonl'
ACTION_TYPE_MAP = {
    "set_plan": "PLAN_UPDATE"
}

def analyze_planning_efficiency(log_file):
    """
    Analyzes the log file to find tasks with multiple plan revisions.

    Args:
        log_file (str): Path to the activity log file.

    Returns:
        dict: A dictionary mapping task IDs to the number of plan updates.
    """
    task_plan_updates = defaultdict(int)
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Support both direct action types and mapped tool names
                    action_type = entry.get('action', {}).get('type')
                    tool_name = entry.get('action', {}).get('details', {}).get('tool_name')

                    is_plan_update = (action_type == 'PLAN_UPDATE' or
                                      ACTION_TYPE_MAP.get(tool_name) == 'PLAN_UPDATE')

                    if is_plan_update:
                        task_id = entry.get('task', {}).get('id')
                        if task_id:
                            task_plan_updates[task_id] += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file}")
        return {}

    return {task: count for task, count in task_plan_updates.items() if count > 1}

def main():
    """
    Main function to run the self-improvement analysis CLI.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes agent activity logs to identify areas for improvement."
    )
    parser.add_argument(
        '--log-file',
        default=LOG_FILE_PATH,
        help=f"Path to the log file. Defaults to {LOG_FILE_PATH}"
    )
    args = parser.parse_args()

    print("Running analysis for planning inefficiencies...")
    inefficient_tasks = analyze_planning_efficiency(args.log_file)

    if not inefficient_tasks:
        print("\nAnalysis Complete: No tasks with significant planning inefficiencies found.")
    else:
        print("\nAnalysis Complete: Found tasks with multiple plan revisions, indicating potential inefficiencies:")
        for task_id, count in inefficient_tasks.items():
            print(f"  - Task ID: {task_id}, Plan Revisions: {count}")

if __name__ == "__main__":
    main()