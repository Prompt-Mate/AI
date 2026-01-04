#!/bin/bash
source venv/bin/activate
uvicorn judge_server:app --host 0.0.0.0 --port 8001 --reload > judge.log 2>&1 &
echo "🧠 Judge Server running on port 8001"
