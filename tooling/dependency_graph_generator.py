import os
import json
import glob
import re

# --- Finder Functions ---

def find_package_json_files(root_dir):
    """Finds all package.json files in the repository, excluding node_modules."""
    all_files = glob.glob(os.path.join(root_dir, '**', 'package.json'), recursive=True)
    return [f for f in all_files if 'node_modules' not in f]

def find_requirements_txt_files(root_dir):
    """Finds all requirements.txt files in the repository."""
    return glob.glob(os.path.join(root_dir, '**', 'requirements.txt'), recursive=True)

# --- Parser Functions ---

def parse_package_json(package_json_path):
    """Parses a single package.json file to extract its name and dependencies."""
    try:
        with open(package_json_path, 'r') as f:
            data = json.load(f)

        package_name = data.get('name', os.path.basename(os.path.dirname(package_json_path)))
        dependencies = list(data.get('dependencies', {}).keys())
        dev_dependencies = list(data.get('devDependencies', {}).keys())

        return {
            "project_name": package_name,
            "path": package_json_path,
            "dependencies": dependencies + dev_dependencies,
            "type": "javascript"
        }
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not parse {package_json_path}. Error: {e}")
        return None

def parse_requirements_txt(requirements_path, root_dir):
    """Parses a requirements.txt file to extract its dependencies."""
    try:
        with open(requirements_path, 'r') as f:
            lines = f.readlines()

        dependencies = []
        for line in lines:
            # Strip comments and whitespace
            line = line.split('#')[0].strip()
            if line:
                # Use regex to get just the package name, ignoring version, extras, etc.
                match = re.match(r'^[a-zA-Z0-9_.-]+', line)
                if match:
                    dependencies.append(match.group(0))

        # If the file is at the root of the scan, give it a special name.
        # Otherwise, use its parent directory's name.
        dir_name = os.path.dirname(requirements_path)
        if os.path.abspath(dir_name) == os.path.abspath(root_dir):
            project_name = "root-python-project"
        else:
            project_name = os.path.basename(dir_name)

        return {
            "project_name": project_name,
            "path": requirements_path,
            "dependencies": dependencies,
            "type": "python"
        }
    except IOError as e:
        print(f"Warning: Could not parse {requirements_path}. Error: {e}")
        return None

# --- Graph Generation ---

def generate_dependency_graph(root_dir='.'):
    """Generates a dependency graph for all supported dependency files found."""
    graph = {"nodes": [], "edges": []}
    all_projects = []

    # Consolidate all discovered projects
    for pf in find_package_json_files(root_dir):
        info = parse_package_json(pf)
        if info: all_projects.append(info)

    for rf in find_requirements_txt_files(root_dir):
        info = parse_requirements_txt(rf, root_dir)
        if info: all_projects.append(info)

    # Add all projects as nodes
    project_names = {p["project_name"] for p in all_projects}
    for proj in all_projects:
        graph["nodes"].append({
            "id": proj["project_name"],
            "path": proj["path"],
            "type": f"{proj['type']}-project"
        })

    # Add dependencies as nodes and create edges
    for proj in all_projects:
        source_id = proj["project_name"]
        for dep in proj["dependencies"]:
            target_id = dep

            # If the dependency is another project in our repo, it's an internal edge
            if target_id in project_names:
                graph["edges"].append({"source": source_id, "target": target_id})
            # Otherwise, it's an external dependency
            else:
                # Add the external dependency as a node if it doesn't exist yet
                if not any(n['id'] == target_id for n in graph['nodes']):
                    graph["nodes"].append({
                        "id": target_id,
                        "path": None,
                        "type": f"{proj['type']}-external"
                    })
                graph["edges"].append({"source": source_id, "target": target_id})

    return graph

def main():
    """Main function to generate and save the dependency graph."""
    print("Generating dependency graph...")
    graph = generate_dependency_graph()

    output_path = os.path.join('knowledge_core', 'dependency_graph.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2)

    print(f"Dependency graph successfully generated at {output_path}")

if __name__ == '__main__':
    main()