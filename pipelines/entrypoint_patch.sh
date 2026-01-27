#!/bin/bash
# Entrypoint script that patches Pipelines and then starts the server

# Apply the patch
python /app/patches/patch_pipelines.py

# Start the original pipelines server
exec python /app/main.py
