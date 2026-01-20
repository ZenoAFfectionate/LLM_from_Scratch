#!/bin/bash
# Script to run TinyStories training experiments
# This script will run all experiments sequentially on GPU 0

set -e  # Exit on error

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate llm

# Set working directory
cd /home/kemove/Courses/STF_LLM/Assignment_1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================================================"
echo -e "${BLUE}Starting TinyStories Training Experiments${NC}"
echo "========================================================================"
echo ""

# Function to run training with error handling
run_training() {
    local config=$1
    local experiment_name=$(basename "$config" .json)

    echo "------------------------------------------------------------------------"
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Starting: ${GREEN}$experiment_name${NC}"
    echo "------------------------------------------------------------------------"

    if CUDA_VISIBLE_DEVICES=0 python train.py --config "$config"; then
        echo -e "${GREEN}✓ Successfully completed: $experiment_name${NC}"
        echo ""
    else
        echo -e "${RED}✗ Failed: $experiment_name${NC}"
        echo -e "${RED}Error occurred. Check logs above for details.${NC}"
        echo ""
        # Uncomment the line below if you want to stop on first error
        # exit 1
    fi
}

# ============================================================================
# Experiment 1: MHA + FFN
# ============================================================================
run_training "config/[MHA+FFN]train_tinystories.json"

# ============================================================================
# Experiment 2: MHA + MoE
# ============================================================================
run_training "config/[MHA+MoE]train_tinystories.json"

# ============================================================================
# Experiment 3: GQA + FFN
# ============================================================================
run_training "config/[GQA+FFN]train_tinystories.json"

# ============================================================================
# Experiment 4: GQA + MoE
# ============================================================================
run_training "config/[GQA+MoE]train_tinystories.json"

# ============================================================================
# Experiment 5: MLA + MoE
# ============================================================================
run_training "config/[MLA+MoE]train_tinystories.json"

# ============================================================================
# All experiments completed
# ============================================================================
echo "========================================================================"
echo -e "${GREEN}All TinyStories experiments completed!${NC}"
echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC}"
echo "========================================================================"
echo ""
echo "Check the checkpoints directory for saved models:"
echo "  /home/kemove/Courses/STF_LLM/Assignment_1/checkpoints/"
echo ""
