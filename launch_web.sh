#!/bin/bash

# Quick start script for Transformer LM Web Interface
# Usage: ./launch_web.sh [config_name] [additional_args]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Transformer LM Web Interface Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Warning: python command not found, trying python3...${NC}"
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Default config
CONFIG_NAME="${1:-tinystories}"
shift 2>/dev/null || true

# Map friendly names to config paths
case $CONFIG_NAME in
    tinystories|ts)
        CONFIG_PATH="config/generate_tinystories.json"
        ;;
    openwebtext|owt)
        CONFIG_PATH="config/generate_openwebtext.json"
        ;;
    *)
        # Assume it's a full path
        CONFIG_PATH="$CONFIG_NAME"
        ;;
esac

echo -e "${GREEN}✓${NC} Using config: ${CONFIG_PATH}"
echo -e "${GREEN}✓${NC} Python command: ${PYTHON_CMD}"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${YELLOW}Warning: Config file not found: ${CONFIG_PATH}${NC}"
    echo -e "Available configs:"
    ls -1 config/generate_*.json 2>/dev/null || echo "  No configs found in config/"
    echo ""
    echo "Usage: $0 [tinystories|openwebtext|path/to/config.json] [additional args]"
    exit 1
fi

# Check if Gradio is installed
$PYTHON_CMD -c "import gradio" 2>/dev/null || {
    echo -e "${YELLOW}Gradio not found! Installing...${NC}"
    pip install gradio>=4.0.0
    echo -e "${GREEN}✓${NC} Gradio installed successfully"
    echo ""
}

echo -e "${BLUE}Starting web interface...${NC}"
echo -e "${BLUE}Additional arguments: $@${NC}"
echo ""

# Launch the web interface
$PYTHON_CMD generate.py --config "$CONFIG_PATH" "$@"
