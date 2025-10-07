import argparse
import datetime
import json
import os
import shutil
import sys
import uuid

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
POSTMORTEM_TEMPLATE_PATH = os.path.join(ROOT_DIR, 'postmortem.md')
POSTMORTEMS_DIR = os.path.join(ROOT_DIR, 'postmortems')
LOG_FILE_PATH = os.path.join(ROOT_DIR, 'logs', 'activity.log.jsonl')
FSM_DEF_PATH = os.path.join(ROOT_DIR, 'tooling', 'fdc_fsm.json')

ACTION_TYPE_MAP = {
    "set_plan": "plan_op",
    "plan_step_complete": "step_op",
    "submit": "submit_op",
    "create_file_with_block": "write_op", "overwrite_file_with_block": "write_op", "replace_with_git_merge_diff": "write_op",
    "read_file": "read_op", "list_files": "read_op", "grep": "read_op",
    "delete_file": "delete_op", "rename_file": "move_op",
    "run_in_bash_session": "tool_exec", "for_each_file": "loop_op"
}

# --- CLI Subcommands & Helpers ---

def _log_event(log_entry):
    """Appends a new log entry to the activity log, ensuring it's on a new line."""
    content_to_write = json.dumps(log_entry) + '\n'
    with open(LOG_FILE_PATH, 'a+') as f:
        # Check if the file is not empty
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            # Check if the last character is a newline
            f.seek(f.tell() - 1)
            if f.read(1) != '\n':
                f.write('\n')
        f.write(content_to_write)

def _create_log_entry(task_id, action_type, details):
    """Creates a structured log entry dictionary."""
    return {
        "log_id": str(uuid.uuid4()), "session_id": os.getenv("JULES_SESSION_ID", "unknown"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase 6", "task": {"id": task_id, "plan_step": -1},
        "action": {"type": action_type, "details": details},
        "outcome": {"status": "SUCCESS", "message": f"FDC CLI: {action_type} for task {task_id}."}
    }

def close_task(task_id):
    """Automates the closing of a Finite Development Cycle."""
    if not task_id: print("Error: --task-id is required.", file=sys.stderr); sys.exit(1)
    safe_task_id = "".join(c for c in task_id if c.isalnum() or c in ('-', '_'))
    new_path = os.path.join(POSTMORTEMS_DIR, f"{datetime.date.today()}-{safe_task_id}.md")
    try:
        shutil.copyfile(POSTMORTEM_TEMPLATE_PATH, new_path)
        print(f"Successfully created new post-mortem file: {new_path}")
    except Exception as e:
        print(f"Error creating post-mortem file: {e}", file=sys.stderr); sys.exit(1)

    _log_event(_create_log_entry(task_id, "POST_MORTEM", {"summary": f"Post-mortem initiated for '{task_id}'."}))
    _log_event(_create_log_entry(task_id, "TASK_END", {"summary": f"Development phase for FDC task '{task_id}' formally closed."}))

    print(f"Logged POST_MORTEM and TASK_END events for task: {task_id}")

def _validate_action(line_num, line_content, state, fsm, fs, placeholders):
    """Validates a single, non-loop action."""
    # Substitute placeholders like {file1}, {file2}
    for key, val in placeholders.items(): line_content = line_content.replace(key, val)

    parts = line_content.split(); command = parts[0]; args = parts[1:]
    action_type = ACTION_TYPE_MAP.get(command)
    if command == "run_in_bash_session" and "close" in args: action_type = "close_op"
    if not action_type: print(f"Error on line {line_num+1}: Unknown command '{command}'.", file=sys.stderr); sys.exit(1)

    # Syntactic check
    transitions = fsm["transitions"].get(state)
    if action_type not in (transitions or {}):
        print(f"Error on line {line_num+1}: Invalid FSM transition. Cannot perform '{action_type}' from state '{state}'.", file=sys.stderr); sys.exit(1)

    # Semantic check
    if command == "create_file_with_block" and args[0] in fs:
        print(f"Error on line {line_num+1}: Semantic error. Cannot create '{args[0]}' because it already exists.", file=sys.stderr); sys.exit(1)
    if command in ["read_file", "delete_file", "replace_with_git_merge_diff"] and args[0] not in fs:
        print(f"Error on line {line_num+1}: Semantic error. Cannot access '{args[0]}' because it does not exist.", file=sys.stderr); sys.exit(1)

    # Apply state changes
    if command == "create_file_with_block": fs.add(args[0])
    if command == "delete_file": fs.remove(args[0])

    next_state = transitions[action_type]
    print(f"  Line {line_num+1}: OK. Action '{command}' ({action_type}) transitions from {state} -> {next_state}")
    return next_state, fs

def _validate_plan_recursive(lines, start_index, indent_level, state, fs, placeholders, fsm):
    """Recursively validates a block of a plan."""
    i = start_index
    while i < len(lines):
        line_num, line_content = lines[i]
        current_indent = len(line_content) - len(line_content.lstrip(' '))

        if current_indent < indent_level: return state, fs, i # End of current block
        if current_indent > indent_level: print(f"Error on line {line_num+1}: Unexpected indentation.", file=sys.stderr); sys.exit(1)

        line_content = line_content.strip()
        command = line_content.split()[0]

        if command == "for_each_file":
            loop_depth = len(placeholders) + 1
            placeholder_key = f"{{file{loop_depth}}}"
            dummy_file = f"dummy_file_for_loop_{loop_depth}"

            # Find loop body
            loop_body_start = i + 1
            j = loop_body_start
            while j < len(lines) and (len(lines[j][1]) - len(lines[j][1].lstrip(' '))) > indent_level:
                j += 1

            # Validate one logical iteration of the loop
            loop_fs = fs.copy(); loop_fs.add(dummy_file)
            new_placeholders = placeholders.copy(); new_placeholders[placeholder_key] = dummy_file

            state, loop_fs, _ = _validate_plan_recursive(lines, loop_body_start, indent_level + 2, state, loop_fs, new_placeholders, fsm)

            fs.update(loop_fs) # Merge FS changes
            i = j
        else:
            state, fs = _validate_action(line_num, line_content, state, fsm, fs, placeholders)
            i += 1

    return state, fs, i

def validate_plan(plan_filepath):
    try:
        with open(FSM_DEF_PATH, 'r') as f: fsm = json.load(f)
        with open(plan_filepath, 'r') as f: lines = [(i, line.rstrip('\n')) for i, line in enumerate(f) if line.strip()]
    except FileNotFoundError as e: print(f"Error: Could not find file {e.filename}", file=sys.stderr); sys.exit(1)

    # Initialize the simulated file system with the actual state of the repository
    simulated_fs = set()
    for root, dirs, files in os.walk('.'):
        # Exclude .git directory from the walk
        if '.git' in dirs:
            dirs.remove('.git')
        for name in files:
            simulated_fs.add(os.path.join(root, name).replace('./', ''))

    print(f"Starting validation with {len(simulated_fs)} files pre-loaded...")
    final_state, _, _ = _validate_plan_recursive(lines, 0, 0, fsm["start_state"], simulated_fs, {}, fsm)

    if final_state in fsm["accept_states"]:
        print(f"\nValidation successful! Plan is syntactically and semantically valid.")
    else:
        print(f"\nValidation failed. Plan ends in non-accepted state: '{final_state}'", file=sys.stderr); sys.exit(1)

def analyze_plan(plan_filepath):
    """Analyzes a plan file to determine its complexity class and modality."""
    try:
        with open(plan_filepath, 'r') as f:
            plan_lines_with_indent = f.readlines()
        plan_lines = [line.strip() for line in plan_lines_with_indent if line.strip()]
    except FileNotFoundError:
        print(f"Error: Plan file not found at {plan_filepath}", file=sys.stderr); sys.exit(1)

    # --- Complexity Analysis ---
    loop_indents = []
    for line in plan_lines_with_indent:
        if line.strip().startswith("for_each_file"):
            indent = len(line) - len(line.lstrip(' '))
            loop_indents.append(indent)

    if not loop_indents:
        complexity = "Constant (O(1))"
    elif max(loop_indents) > min(loop_indents):
        complexity = "Exponential (EXPTIME-Class)"
    else:
        complexity = "Polynomial (P-Class)"

    # --- Modality Analysis ---
    has_write_op = False
    write_ops = {"write_op", "delete_op", "move_op"}
    for line in plan_lines:
        command = line.split()[0]
        action_type = ACTION_TYPE_MAP.get(command)
        if action_type in write_ops:
            has_write_op = True
            break

    modality = "Construction (Read-Write)" if has_write_op else "Analysis (Read-Only)"

    print(f"Plan Analysis Results:")
    print(f"  - Complexity: {complexity}")
    print(f"  - Modality:   {modality}")

def main():
    parser = argparse.ArgumentParser(description="A tool to manage the Finite Development Cycle (FDC).")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands", required=True)

    close_parser = subparsers.add_parser("close", help="Closes a task, initiating the post-mortem process.")
    close_parser.add_argument("--task-id", required=True, help="The unique identifier for the task.")

    validate_parser = subparsers.add_parser("validate", help="Validates a plan file against the FDC FSM.")
    validate_parser.add_argument("plan_file", help="The path to the plan file to validate.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyzes a plan to determine its complexity class.")
    analyze_parser.add_argument("plan_file", help="The path to the plan file to analyze.")

    args = parser.parse_args()
    if args.command == "close": close_task(args.task_id)
    elif args.command == "validate": validate_plan(args.plan_file)
    elif args.command == "analyze": analyze_plan(args.plan_file)

if __name__ == "__main__":
    main()