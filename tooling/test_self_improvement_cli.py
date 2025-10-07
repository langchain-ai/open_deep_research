import unittest
import os
import json
from tooling.self_improvement_cli import analyze_planning_efficiency

class TestSelfImprovementCLI(unittest.TestCase):

    def setUp(self):
        """Set up a temporary log file for testing."""
        self.test_log_path = "temp_test_activity.log.jsonl"
        # Create some sample log data that mimics the real log structure
        log_entries = [
            # An efficient task with only one plan update
            {"task": {"id": "task-efficient-1"}, "action": {"details": {"tool_name": "set_plan"}}},
            {"task": {"id": "task-efficient-1"}, "action": {"details": {"tool_name": "read_file"}}},

            # An inefficient task with three plan revisions via the 'set_plan' tool
            {"task": {"id": "task-inefficient-1"}, "action": {"details": {"tool_name": "set_plan"}}},
            {"task": {"id": "task-inefficient-1"}, "action": {"details": {"tool_name": "list_files"}}},
            {"task": {"id": "task-inefficient-1"}, "action": {"details": {"tool_name": "set_plan"}}},
            {"task": {"id": "task-inefficient-1"}, "action": {"details": {"tool_name": "read_file"}}},
            {"task": {"id": "task-inefficient-1"}, "action": {"details": {"tool_name": "set_plan"}}},

            # An inefficient task with two plan revisions via the 'PLAN_UPDATE' action type
            {"task": {"id": "task-inefficient-2"}, "action": {"type": "PLAN_UPDATE"}},
            {"task": {"id": "task-inefficient-2"}, "action": {"type": "PLAN_UPDATE"}},

            # A task with no planning actions at all
            {"task": {"id": "task-no-plan"}, "action": {"details": {"tool_name": "run_in_bash_session"}}},
        ]
        with open(self.test_log_path, 'w') as f:
            for entry in log_entries:
                # Add other required fields to make the log entry valid
                full_entry = {
                    "log_id": "test-id", "session_id": "test-session", "timestamp": "now",
                    "phase": "Phase 5", "outcome": {"status": "SUCCESS"}
                }
                full_entry.update(entry)
                f.write(json.dumps(full_entry) + '\n')

    def tearDown(self):
        """Clean up the temporary log file."""
        if os.path.exists(self.test_log_path):
            os.remove(self.test_log_path)

    def test_analyze_planning_efficiency(self):
        """Test that the analysis correctly identifies tasks with multiple plan updates."""
        inefficient_tasks = analyze_planning_efficiency(self.test_log_path)

        # Check that only the inefficient tasks are reported
        self.assertIn("task-inefficient-1", inefficient_tasks)
        self.assertIn("task-inefficient-2", inefficient_tasks)
        self.assertEqual(len(inefficient_tasks), 2, "Should only find the two inefficient tasks")

        # Check the counts of plan revisions are correct
        self.assertEqual(inefficient_tasks["task-inefficient-1"], 3)
        self.assertEqual(inefficient_tasks["task-inefficient-2"], 2)

if __name__ == '__main__':
    unittest.main()