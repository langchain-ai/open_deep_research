# Baseline Test Notes

Command:

```powershell
python -m pytest -q
```

Key error:

```text
ModuleNotFoundError: No module named 'langchain'
```

Reason:

The baseline test run fails during pytest collection because the active Python environment does not have the upstream `langchain` dependency installed.
