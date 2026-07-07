#!/usr/bin/env bash
# Render build script — CPU PyTorch + app dependencies (Python 3.11)
set -euo pipefail

python --version
pip install --upgrade pip
pip install -r requirements-torch-cpu.txt
pip install -r requirements.txt
