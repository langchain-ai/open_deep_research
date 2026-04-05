$ErrorActionPreference = 'Stop'

Write-Host '=== PR1-PR3 Acceptance Runner ==='

$python = 'd:/AI_Projects/open_deep_research/.venv/Scripts/python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python"
    exit 1
}

Write-Host 'Step 0: Print Python version'
& $python --version

Write-Host 'Step 1: Run static checks for changed implementation files'
& $python -m py_compile src/open_deep_research/configuration.py
& $python -m py_compile src/open_deep_research/ingestion.py
& $python -m py_compile src/open_deep_research/rag.py
& $python -m py_compile src/open_deep_research/utils.py
& $python -m py_compile src/open_deep_research/deep_researcher.py
& $python -m py_compile src/open_deep_research/prompts.py

Write-Host 'Step 2: Run acceptance validations (PR1-PR3)'
& $python tests/pr1_pr3_acceptance/validate_pr1_pr3.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "Acceptance failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host '=== Acceptance Passed ==='
