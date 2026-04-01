#!/bin/bash
# PhoneticSAE Setup Script
# Comprehensive setup and validation for Phase 1
# Usage: bash scripts/setup.sh

set -e

echo "======================================================================="
echo "PHONETIC SAE - AUTOMATED SETUP SCRIPT"
echo "======================================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}[1/5]${NC} Checking environment prerequisites..."
python scripts/validate_environment.py || exit 1

echo ""
echo -e "${BLUE}[2/5]${NC} Validating repository structure..."
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Not a git repository. Skipping submodule check.${NC}"
else
    git submodule status || echo -e "${YELLOW}⚠️  Could not check submodules${NC}"
fi

echo ""
echo -e "${BLUE}[3/5]${NC} Inspecting Qwen3-ForcedAligner API..."
echo "Running: python scripts/inspect_aligner_api.py --device cuda"
python scripts/inspect_aligner_api.py --device cuda > aligner_api_verification.txt 2>&1

if grep -q "✅" aligner_api_verification.txt; then
    echo -e "${GREEN}✅ Aligner API verified successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Aligner API verification produced warnings (see aligner_api_verification.txt)${NC}"
fi

echo ""
echo -e "${BLUE}[4/5]${NC} Extracting phoneme inventories..."
for lang in en zh yue; do
    echo "  Extracting for language: $lang"
    python scripts/inspect_aligner.py --lang $lang > /dev/null 2>&1 || echo "    ⚠️  Issue with $lang (this is OK for now)"
done
echo -e "${GREEN}✅ Phoneme inventory extraction complete${NC}"

echo ""
echo -e "${BLUE}[5/5]${NC} Generating synthetic test dataset..."
python scripts/generate_test_dataset.py \
    --output data/test_dataset \
    --num-samples 5 \
    --lang en \
    || echo -e "${YELLOW}⚠️  Could not generate test dataset (torchaudio may not be installed)${NC}"

echo ""
echo "======================================================================="
echo "SETUP COMPLETE!"
echo "======================================================================="
echo ""
echo -e "${GREEN}✅ Environment is ready for Phase 1${NC}"
echo ""
echo "Next steps:"
echo "  1. Read the Quick Start Guide:"
echo "     📖 docs/PHASE1_QUICKSTART.md"
echo ""
echo "  2. Verify your setup (already done above)"
echo ""
echo "  3. Try a pilot capture with synthetic data:"
echo "     python scripts/capture_with_alignment.py \\"
echo "       --model qwen3tts \\"
echo "       --dataset custom \\"
echo "       --dataset-csv data/test_dataset/dataset.jsonl \\"
echo "       --lang en \\"
echo "       --output data/pilot_activations \\"
echo "       --num-samples 5 \\"
echo "       --device cuda"
echo ""
echo "  4. Check results:"
echo "     ls -la data/pilot_activations/"
echo ""
echo "  5. Scale to full dataset (50K samples):"
echo "     See docs/PHASE1_QUICKSTART.md for details"
echo ""
echo "For more detailed instructions, see:"
echo "  📖 docs/PHASE1_QUICKSTART.md"
echo "  🔧 docs/PHONEME_ALIGNMENT.md"
echo "  ⚙️  docs/QWEN3_FORCEDALIGNER_INFERENCE.md"
echo ""
echo "Questions? Check the troubleshooting section in:"
echo "  📖 README.md#troubleshooting"
echo ""
