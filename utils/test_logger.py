import unittest
import os
import json
import shutil
from jsonschema import ValidationError
from utils.logger import Logger

class TestLogger(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.test_dir = "test_temp_logger"
        self.schema_dir = os.path.join(self.test_dir, 'config')
        self.log_dir = os.path.join(self.test_dir, 'logs')
        os.makedirs(self.schema_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.schema_path = os.path.join(self.schema_dir, "LOGGING_SCHEMA.md")
        self.log_path = os.path.join(self.log_dir, "activity.log.jsonl")

        # The correct v1.1 schema for testing
        self.test_schema_dict = {
            "type": "object",
            "properties": {
                "log_id": {"type": "string"},
                "session_id": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "phase": {"type": "string"},
                "task": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "plan_step": {"type": "integer"}
                    },
                    "required": ["id", "plan_step"]
                },
                "action": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "details": {"type": "object"}
                    },
                    "required": ["type", "details"]
                },
                "outcome": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"}
                    },
                    "required": ["status"]
                },
                "evidence_citation": {"type": "string"}
            },
            "required": ["log_id", "session_id", "timestamp", "phase", "task", "action", "outcome"]
        }

        # Write the schema to the test file
        with open(self.schema_path, 'w') as f:
            f.write("```json\n")
            f.write(json.dumps(self.test_schema_dict))
            f.write("\n```")

        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def tearDown(self):
        """Clean up the temporary environment after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_log_success_with_correct_schema(self):
        """Test that a valid log entry is written successfully with the correct schema."""
        logger = Logger(schema_path=self.schema_path, log_path=self.log_path)
        logger.log(
            phase="Phase 5",
            task_id="test-task-01",
            plan_step=1,
            action_type="TOOL_EXEC",
            action_details={"command": "ls"},
            outcome_status="SUCCESS"
        )

        self.assertTrue(os.path.exists(self.log_path))
        with open(self.log_path, 'r') as f:
            log_data = json.load(f)
            self.assertEqual(log_data['task']['id'], "test-task-01")

    def test_log_failure_with_incorrect_schema_data(self):
        """Test that logging fails when data violates the v1.1 schema."""
        logger = Logger(schema_path=self.schema_path, log_path=self.log_path)

        # This will fail because the schema requires a "task" object, not a "task_id" at the root.
        # However, my current logger implementation doesn't accept arbitrary fields.
        # The validation should catch the missing 'task' object.
        # Let's test by passing an invalid 'phase' which is simpler to check.
        with self.assertRaises(ValidationError):
            logger.log(
                phase=123, # Invalid type
                task_id="invalid-task",
                plan_step=1,
                action_type="INFO",
                action_details={},
                outcome_status="SUCCESS"
            )

        self.assertFalse(os.path.exists(self.log_path))

if __name__ == '__main__':
    unittest.main()