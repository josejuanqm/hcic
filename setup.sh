#!/bin/bash

set -e

GREEN='\033[92m'
RED='\033[91m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo "${BOLD}CURATOR — Setup${RESET}"
echo "${DIM}Human Connection as Intelligent Context${RESET}"
echo ""

# Find Python with sqlite3 extension support
# macOS system Python lacks this — Homebrew Python has it
PYTHON=""
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c "import sqlite3; c = sqlite3.connect(':memory:'); c.enable_load_extension(True)" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "${RED}Error: No Python with sqlite3 extension support found.${RESET}"
    echo ""
    echo "Fix: brew install python"
    echo "Then run ./setup.sh again."
    exit 1
fi

echo "${DIM}Using: $PYTHON${RESET}"

# Recreate venv if it was built with wrong Python
if [ -d ".venv" ]; then
    rm -rf .venv
fi

echo "${DIM}Creating virtual environment...${RESET}"
"$PYTHON" -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
echo "${DIM}Installing dependencies...${RESET}"
pip install -r requirements.txt -q

echo ""
echo "${GREEN}✓ Setup complete${RESET}"
echo ""
echo "To run tests:"
echo "  ${DIM}source .venv/bin/activate${RESET}"
echo "  ${DIM}export ANTHROPIC_API_KEY=your_key_here${RESET}"
echo "  ${DIM}python3 test_curator.py${RESET}"
echo ""
