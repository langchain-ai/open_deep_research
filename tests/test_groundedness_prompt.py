import ast
import re
from pathlib import Path


def _top_level_node(path: Path, name: str) -> ast.AST:
    module = ast.parse(path.read_text())
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return node.value
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_groundedness_prompt_matches_claim_model() -> None:
    tests_dir = Path(__file__).parent
    prompt_node = _top_level_node(tests_dir / "prompts.py", "GROUNDEDNESS_PROMPT")
    prompt = ast.literal_eval(prompt_node).format(context="context", report="report")
    claim_example = re.search(
        r'<Output Format>.*?"claims"\s*:\s*\[\s*\{(.*?)\}',
        prompt,
        re.DOTALL,
    )
    assert claim_example is not None
    prompt_fields = set(re.findall(r'"([^"]+)"\s*:', claim_example.group(1)))

    evaluator_node = _top_level_node(
        tests_dir / "evaluators.py",
        "GroundednessClaim",
    )
    assert isinstance(evaluator_node, ast.ClassDef)
    model_fields = {
        node.target.id
        for node in evaluator_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert prompt_fields == model_fields
