#!/bin/bash
source venv/bin/activate
uvicorn rewrite_server:app --host 0.0.0.0 --port 8000 --reload > rewrite.log 2>&1 &
echo "🚀 Rewrite Server running on port 8000"
