#!/bin/bash
# Script to run OpenWebText training experiments
#
# Configuration:
# - All models run on single GPU (GPU 1)
# - FFN models: batch_size=8, gradient_accumulation=8 (effective batch=64)
# - MoE models: batch_size=2, gradient_accumulation=32 (effective batch=64)
#
# Both configurations achieve the same effective batch size of 64 for fair comparison

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
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo "========================================================================"
echo -e "${BLUE}Starting OpenWebText Training Experiments${NC}"
echo "========================================================================"
echo -e "${MAGENTA}All models run on GPU 1${NC}"
echo "========================================================================"
echo ""

# Unified function to run training
run_training() {
    local config=$1
    local experiment_name=$(basename "$config" .json)

    # Determine gradient accumulation steps based on model type
    # MoE models need more accumulation due to smaller batch size
    if [[ $experiment_name == *"MoE"* ]]; then
        local grad_accum_steps=32
        local model_type="MoE"
    else
        local grad_accum_steps=8
        local model_type="FFN"
    fi

    echo "------------------------------------------------------------------------"
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Starting: ${GREEN}$experiment_name${NC}"
    echo -e "${BLUE}GPU: GPU 1${NC}"
    echo -e "${BLUE}Model Type: ${model_type}${NC}"
    echo -e "${BLUE}Gradient Accumulation Steps: ${grad_accum_steps}${NC}"
    echo "------------------------------------------------------------------------"

    if CUDA_VISIBLE_DEVICES=1 python train.py --config "$config" --gradient_accumulation_steps ${grad_accum_steps}; then
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
run_training "config/[MHA+FFN]train_openwebtext.json"

# ============================================================================
# Experiment 2: GQA + FFN
# ============================================================================
run_training "config/[GQA+FFN]train_openwebtext.json"

# ============================================================================
# Experiment 3: MHA + MoE
# ============================================================================
run_training "config/[MHA+MoE]train_openwebtext.json"

# ============================================================================
# Experiment 4: GQA + MoE
# ============================================================================
run_training "config/[GQA+MoE]train_openwebtext.json"

# ============================================================================
# Experiment 5: MLA + MoE
# ============================================================================
run_training "config/[MLA+MoE]train_openwebtext.json"

# ============================================================================
# All experiments completed
# ============================================================================
echo "========================================================================"
echo -e "${GREEN}All OpenWebText experiments completed!${NC}"
echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC}"
echo "========================================================================"
echo ""
echo "Check the checkpoints directory for saved models:"
echo "  /home/kemove/Courses/STF_LLM/Assignment_1/checkpoints/"
echo ""
echo "Configuration Summary:"
echo "  All experiments run on: GPU 1"
echo ""
echo "  FFN models (MHA+FFN, GQA+FFN):"
echo "    - Batch size: 8"
echo "    - Gradient accumulation: 8 steps"
echo "    - Effective batch size: 64 (8 × 8)"
echo ""
echo "  MoE models (MHA+MoE, GQA+MoE, MLA+MoE):"
echo "    - Batch size: 2"
echo "    - Gradient accumulation: 32 steps"
echo "    - Effective batch size: 64 (2 × 32)"
echo ""
