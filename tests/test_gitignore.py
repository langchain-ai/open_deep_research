import subprocess
from pathlib import Path


def test_documented_dot_venv_is_ignored(tmp_path) -> None:
    repo_root = Path(__file__).parents[1]
    (tmp_path / ".gitignore").write_text(
        (repo_root / ".gitignore").read_text()
    )
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    virtualenv_file = tmp_path / ".venv" / "bin" / "python"
    virtualenv_file.parent.mkdir(parents=True)
    virtualenv_file.touch()

    result = subprocess.run(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--quiet",
            ".venv/bin/python",
        ],
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 0
