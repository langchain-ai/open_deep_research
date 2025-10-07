import unittest
import os
import json
import shutil
from tooling.dependency_graph_generator import (
    find_package_json_files,
    find_requirements_txt_files,
    parse_package_json,
    parse_requirements_txt,
    generate_dependency_graph
)

class TestDependencyGraphGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_dir = "temp_test_repo"
        os.makedirs(self.test_dir, exist_ok=True)

        # JS project setup
        self.js_project_dir = os.path.join(self.test_dir, 'js_app')
        os.makedirs(self.js_project_dir, exist_ok=True)
        self.pkg_json_path = os.path.join(self.js_project_dir, 'package.json')
        with open(self.pkg_json_path, 'w') as f:
            json.dump({
                "name": "js-app",
                "dependencies": {
                    "react": "18.0.0",
                    "root-python-project": "1.0.0" # Internal dependency
                }
            }, f)

        # Python project setup in a subdirectory
        self.py_project_dir = os.path.join(self.test_dir, 'py_app')
        os.makedirs(self.py_project_dir, exist_ok=True)
        self.req_txt_path = os.path.join(self.py_project_dir, 'requirements.txt')
        with open(self.req_txt_path, 'w') as f:
            f.write("flask>=2.0.0\n# This is a comment\nrequests\n")

        # Root python project setup
        self.root_req_txt_path = os.path.join(self.test_dir, 'requirements.txt')
        with open(self.root_req_txt_path, 'w') as f:
            f.write("jsonschema\n")


    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_find_files(self):
        """Test finding both package.json and requirements.txt files."""
        js_files = find_package_json_files(self.test_dir)
        py_files = find_requirements_txt_files(self.test_dir)
        self.assertEqual(len(js_files), 1)
        self.assertIn(self.pkg_json_path, js_files)
        self.assertEqual(len(py_files), 2)
        self.assertIn(self.req_txt_path, py_files)
        self.assertIn(self.root_req_txt_path, py_files)

    def test_parse_package_json(self):
        """Test parsing a single package.json file."""
        info = parse_package_json(self.pkg_json_path)
        self.assertIsNotNone(info)
        self.assertEqual(info['project_name'], 'js-app')
        self.assertEqual(info['type'], 'javascript')
        self.assertIn('react', info['dependencies'])
        self.assertIn('root-python-project', info['dependencies'])

    def test_parse_requirements_txt(self):
        """Test parsing a single requirements.txt file."""
        # Test a nested requirements file
        info_nested = parse_requirements_txt(self.req_txt_path, self.test_dir)
        self.assertIsNotNone(info_nested)
        self.assertEqual(info_nested['project_name'], 'py_app')
        self.assertEqual(info_nested['type'], 'python')
        self.assertEqual(info_nested['dependencies'], ['flask', 'requests'])

        # Test a root requirements file
        info_root = parse_requirements_txt(self.root_req_txt_path, self.test_dir)
        self.assertIsNotNone(info_root)
        self.assertEqual(info_root['project_name'], 'root-python-project')

    def test_generate_dependency_graph(self):
        """Test the full graph generation logic with mixed project types."""
        graph = generate_dependency_graph(self.test_dir)

        nodes = graph['nodes']
        edges = graph['edges']

        node_ids = {n['id'] for n in nodes}

        # Check all projects and dependencies are nodes
        expected_nodes = {
            'js-app', 'py_app', 'root-python-project', # projects
            'react', # js external
            'flask', 'requests', 'jsonschema' # python external
        }
        self.assertEqual(node_ids, expected_nodes)

        # Check all edges are created
        self.assertEqual(len(edges), 5)

        edge_set = {(e['source'], e['target']) for e in edges}
        expected_edge_set = {
            ('js-app', 'react'),
            ('js-app', 'root-python-project'), # Internal edge
            ('py_app', 'flask'),
            ('py_app', 'requests'),
            ('root-python-project', 'jsonschema')
        }
        self.assertEqual(edge_set, expected_edge_set)

if __name__ == '__main__':
    unittest.main()