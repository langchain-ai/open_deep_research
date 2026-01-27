#!/usr/bin/env python3
"""
Patch script for OpenWebUI Pipelines to fix the pipeline. prefix routing issue.

This script patches the main.py file in the Pipelines container to handle
the 'pipeline.' prefix that newer versions of OpenWebUI add to pipeline IDs.

Issue: https://github.com/open-webui/pipelines/issues/543

Usage:
  Mount this script into the container and run it on startup, OR
  Apply the patch manually to main.py
"""

import os
import re

MAIN_PY_PATH = "/app/main.py"


def patch_main_py():
    """Patch main.py to handle pipeline. prefix in filter endpoints."""

    if not os.path.exists(MAIN_PY_PATH):
        print(f"[PATCH] {MAIN_PY_PATH} not found, skipping patch")
        return False

    with open(MAIN_PY_PATH, "r") as f:
        content = f.read()

    # Check if already patched
    if (
        'pipeline_id.split(".")[-1]' in content
        or "pipeline_id.split('.')[-1]" in content
    ):
        print("[PATCH] main.py is already patched")
        return True

    # Patch filter_inlet function
    inlet_pattern = r"(async def filter_inlet\(pipeline_id: str, form_data: FilterForm\):)\s*\n(\s*)(if pipeline_id not in app\.state\.PIPELINES:)"
    inlet_replacement = r'\1\n\2# Patch: Handle pipeline. prefix from newer OpenWebUI versions\n\2pipeline_id = pipeline_id.split(".")[-1]\n\2\3'

    new_content, inlet_count = re.subn(inlet_pattern, inlet_replacement, content)

    # Patch filter_outlet function
    outlet_pattern = r"(async def filter_outlet\(pipeline_id: str, form_data: FilterForm\):)\s*\n(\s*)(if pipeline_id not in app\.state\.PIPELINES:)"
    outlet_replacement = r'\1\n\2# Patch: Handle pipeline. prefix from newer OpenWebUI versions\n\2pipeline_id = pipeline_id.split(".")[-1]\n\2\3'

    new_content, outlet_count = re.subn(outlet_pattern, outlet_replacement, new_content)

    if inlet_count > 0 or outlet_count > 0:
        with open(MAIN_PY_PATH, "w") as f:
            f.write(new_content)
        print(
            f"[PATCH] Successfully patched main.py (inlet: {inlet_count}, outlet: {outlet_count})"
        )
        return True
    else:
        print("[PATCH] Could not find patterns to patch in main.py")
        return False


if __name__ == "__main__":
    patch_main_py()
