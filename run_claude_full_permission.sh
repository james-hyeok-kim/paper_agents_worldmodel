#!/bin/bash
# World Model 연구 파이프라인 실행 스크립트
# 사용법: ./run_claude_full_permission.sh

WORKSPACE="/home/jovyan/workspace/paper_agents_worldmodel"

echo "Starting paper_agents_worldmodel Claude session..."
echo "Working directory: $WORKSPACE"

cd "$WORKSPACE"

claude \
  --dangerously-skip-permissions \
  --model claude-sonnet-4-6 \
  "$@"
