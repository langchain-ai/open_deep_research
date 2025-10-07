import unittest
import os
import json
import shutil
import subprocess
from unittest.mock import patch
from tooling.symbol_map_generator import has_ctags, generate_symbols_with_ctags, generate_symbols_with_ast, main as symbol_main

class TestSymbolMapGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_dir = "temp_symbol_test_repo"
        self.knowledge_core_dir = os.path.join(self.test_dir, 'knowledge_core')
        os.makedirs(self.knowledge_core_dir, exist_ok=True)

        self.py_file_path = os.path.join(self.test_dir, 'module.py')
        with open(self.py_file_path, 'w') as f:
            f.write("class MyClass:\n    def my_method(self):\n        pass\n\ndef my_function():\n    pass")

        # Mock the output path to be within the test directory
        self.symbols_output_path = os.path.join(self.knowledge_core_dir, 'symbols.json')

    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('tooling.symbol_map_generator.subprocess.run')
    def test_generate_with_ctags_success(self, mock_subprocess_run):
        """Test successful symbol generation using a mocked ctags call."""

        # Mock the ctags command to "create" a file with dummy JSON-lines output
        def side_effect(*args, **kwargs):
            ctags_output_path = args[0][args[0].index('-f') + 1]
            with open(ctags_output_path, 'w') as f:
                f.write('{"_type": "tag", "name": "MyClass", "path": "module.py", "kind": "class"}\n')
                f.write('{"_type": "tag", "name": "my_method", "path": "module.py", "kind": "method"}\n')
            return subprocess.CompletedProcess(args=args, returncode=0)

        mock_subprocess_run.side_effect = side_effect

        original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            result = generate_symbols_with_ctags('.')
            self.assertTrue(result)
            self.assertTrue(os.path.exists('knowledge_core/symbols.json'))

            with open('knowledge_core/symbols.json', 'r') as f:
                data = json.load(f)

            self.assertIn('symbols', data)
            self.assertEqual(len(data['symbols']), 2)
            self.assertEqual(data['symbols'][0]['name'], 'MyClass')
        finally:
            os.chdir(original_cwd)

    def test_generate_with_ast_fallback(self):
        """Test the AST-based fallback for Python files."""
        symbols_data = generate_symbols_with_ast(self.test_dir)
        self.assertIn('symbols', symbols_data)
        self.assertEqual(len(symbols_data['symbols']), 3) # class, method, function

        names = {s['name'] for s in symbols_data['symbols']}
        self.assertEqual(names, {'MyClass', 'my_method', 'my_function'})

        my_class_symbol = next(s for s in symbols_data['symbols'] if s['name'] == 'MyClass')
        self.assertEqual(my_class_symbol['kind'], 'class')
        self.assertEqual(my_class_symbol['path'], self.py_file_path)

    @patch('tooling.symbol_map_generator.has_ctags', return_value=False)
    def test_main_with_ast_fallback(self, mock_has_ctags):
        """Test that main function uses the AST fallback when ctags is not present."""
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            from tooling import symbol_map_generator
            symbol_map_generator.main()

            self.assertTrue(os.path.exists('knowledge_core/symbols.json'))
            with open('knowledge_core/symbols.json', 'r') as f:
                data = json.load(f)
            self.assertEqual(len(data['symbols']), 3)
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()