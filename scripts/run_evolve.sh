#!/bin/bash
###############################################################################
# ReCreate Evolution Script
# Runs scaffold evolution and testing for a single domain.
###############################################################################

set -e

# Cleanup containers
echo "Cleaning up containers..."
docker container prune -f > /dev/null 2>&1 || true
docker ps -q --filter "name=dacode_" --filter "name=appworld_" --filter "name=minisweagent-" 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
echo "Done"

###############################################################################
# Default Configuration
###############################################################################

# Domain (swe, dacode, ds1000, math, appworld)
DOMAIN="${DOMAIN:-swe}"

# SWE-bench
SWE_SUBSET="${SWE_SUBSET:-django}"
SWE_EVOLVE_SIZE="${SWE_EVOLVE_SIZE:-20}"
SWE_TEST_SIZE="${SWE_TEST_SIZE:-}"

# DA-Code (subsets: all, sa, ml, visual, di, dm)
DACODE_SUBSET="${DACODE_SUBSET:-ml}"
DACODE_EVOLVE_SIZE="${DACODE_EVOLVE_SIZE:-20}"
DACODE_TEST_SIZE="${DACODE_TEST_SIZE:-}"

# DS-1000 (subsets: Pandas, Numpy, Matplotlib, Sklearn, Scipy, Pytorch, Tensorflow)
DS1000_SUBSET="${DS1000_SUBSET:-Numpy}"
DS1000_EVOLVE_SIZE="${DS1000_EVOLVE_SIZE:-20}"
DS1000_TEST_SIZE="${DS1000_TEST_SIZE:-}"

# Math (datasets: math500, aime24, aime25, amc23)
MATH_EVOLVE_DATASET="${MATH_EVOLVE_DATASET:-aime24}"
MATH_TEST_DATASETS="${MATH_TEST_DATASETS:-math500}"
MATH_TEST_SIZE="${MATH_TEST_SIZE:-}"

# AppWorld
APP_EVOLVE_DATASET="${APP_EVOLVE_DATASET:-dev}"
APP_EVOLVE_SIZE="${APP_EVOLVE_SIZE:-18}"
APP_TEST_NORMAL_SIZE="${APP_TEST_NORMAL_SIZE:-}"
APP_TEST_CHALLENGE_SIZE="${APP_TEST_CHALLENGE_SIZE:-}"

# Parallel settings
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"
N_REPEAT="${N_REPEAT:-2}"

# Model settings
AGENT_MODEL="${AGENT_MODEL:-gpt-5-mini}"
RECREATE_MODEL="${RECREATE_MODEL:-${META_MODEL:-claude-opus-4-5-20251101}}"
AGENT_TEMP="${AGENT_TEMP:-1.0}"
RECREATE_TEMP="${RECREATE_TEMP:-${META_TEMP:-0.5}}"

# Test settings
IF_TEST="${IF_TEST:-true}"
USE_EXISTING="${USE_EXISTING:-false}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"

# Output (convert to absolute path for Docker mount compatibility)
OUTPUT_DIR="${OUTPUT_DIR:-${RECREATE_OUTPUT_DIR:-./output}}"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$OUTPUT_DIR")" || OUTPUT_DIR="$(pwd)/output"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

# Other
SEED="${SEED:-0}"
MAX_RETRIES="${MAX_RETRIES:-1}"

# Ablation flags (all enabled by default)
ABLATION_TRAJECTORY="${ABLATION_TRAJECTORY:-true}"
ABLATION_ENVIRONMENT="${ABLATION_ENVIRONMENT:-true}"
ABLATION_EVAL_RESULTS="${ABLATION_EVAL_RESULTS:-true}"
ABLATION_MODIFICATION_GUIDANCE="${ABLATION_MODIFICATION_GUIDANCE:-true}"

###############################################################################
# Help
###############################################################################
show_help() {
    cat << 'EOF'
ReCreate Evolution Script

Usage: ./scripts/run_evolve.sh <domain> [options]

Domains:
    swe        SWE-bench (software engineering)
    dacode     DA-Code (data science)
    ds1000     DS-1000 (code completion)
    math       Math (mathematical reasoning)
    appworld   AppWorld (API interaction)

Common Options:
    --subset <name>          Subset filter
    --evolve-size <n>        Evolution set size
    --test-size <n>          Test set size (empty=all remaining)
    --batch-size <n>         Evolution batch size (default: 4)
    --workers <n>            Test parallelism (default: 4)
    --agent-model <name>     Agent model
    --recreate-model <name>  ReCreate-Agent model (or --meta-model for backward compatibility)
    --no-test                Skip testing phase
    --use-existing <dir>     Use existing experiment directory
    --output-dir <path>      Output directory
    --help, -h               Show this help

Ablation Options:
    --no-trajectory            Disable trajectory analysis
    --no-environment           Disable environment inspection
    --no-eval-results          Disable evaluation results access
    --no-modification-guidance Disable modification guidance

Examples:
    ./scripts/run_evolve.sh swe --subset django --evolve-size 5
    ./scripts/run_evolve.sh math --evolve-dataset aime24
    ./scripts/run_evolve.sh appworld --app-evolve-size 18

EOF
    exit 0
}

###############################################################################
# Argument Parsing
###############################################################################
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
    DOMAIN="$1"
    shift
fi

if [ -z "$DOMAIN" ]; then
    echo "Error: No domain specified"
    echo "Usage: $0 <domain> [options]"
    exit 1
fi

CMD_SUBSET=""
CMD_EVOLVE_SIZE=""
CMD_TEST_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset) CMD_SUBSET="$2"; shift 2 ;;
        --evolve-size) CMD_EVOLVE_SIZE="$2"; shift 2 ;;
        --test-size) CMD_TEST_SIZE="$2"; shift 2 ;;
        --evolve-dataset) MATH_EVOLVE_DATASET="$2"; shift 2 ;;
        --test-datasets) MATH_TEST_DATASETS="$2"; shift 2 ;;
        --app-evolve-dataset) APP_EVOLVE_DATASET="$2"; shift 2 ;;
        --app-evolve-size) APP_EVOLVE_SIZE="$2"; shift 2 ;;
        --app-test-normal-size) APP_TEST_NORMAL_SIZE="$2"; shift 2 ;;
        --app-test-challenge-size) APP_TEST_CHALLENGE_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --n-repeat) N_REPEAT="$2"; shift 2 ;;
        --agent-model) AGENT_MODEL="$2"; shift 2 ;;
        --meta-model|--recreate-model) RECREATE_MODEL="$2"; shift 2 ;;
        --agent-temp) AGENT_TEMP="$2"; shift 2 ;;
        --meta-temp|--recreate-temp) RECREATE_TEMP="$2"; shift 2 ;;
        --no-test) IF_TEST=false; shift ;;
        --use-existing) USE_EXISTING=true; EXPERIMENT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --no-trajectory) ABLATION_TRAJECTORY=false; shift ;;
        --no-environment) ABLATION_ENVIRONMENT=false; shift ;;
        --no-eval-results) ABLATION_EVAL_RESULTS=false; shift ;;
        --no-modification-guidance) ABLATION_MODIFICATION_GUIDANCE=false; shift ;;
        --help|-h) show_help ;;
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

###############################################################################
# Initialization
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SHUFFLED_DATA_DIR="$PROJECT_ROOT/shuffled_data"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

export GITHUB_TOKEN="${GITHUB_TOKEN:-}"

###############################################################################
# Domain Configuration
###############################################################################
case $DOMAIN in
    swe)
        DOMAIN_ID="swe"
        DOMAIN_NAME="SWE-bench"
        SUBSET="${CMD_SUBSET:-$SWE_SUBSET}"
        EVOLVE_SIZE="${CMD_EVOLVE_SIZE:-$SWE_EVOLVE_SIZE}"
        TEST_SIZE="${CMD_TEST_SIZE:-$SWE_TEST_SIZE}"
        SHUFFLE_FILE="$SHUFFLED_DATA_DIR/swebench_verified_${SUBSET}_shuffled.json"
        ;;
    dacode)
        DOMAIN_ID="data_science"
        DOMAIN_NAME="DA-Code"
        SUBSET="${CMD_SUBSET:-$DACODE_SUBSET}"
        EVOLVE_SIZE="${CMD_EVOLVE_SIZE:-$DACODE_EVOLVE_SIZE}"
        TEST_SIZE="${CMD_TEST_SIZE:-$DACODE_TEST_SIZE}"
        SHUFFLE_FILE="$SHUFFLED_DATA_DIR/dacode_${SUBSET}_shuffled.json"
        ;;
    ds1000)
        DOMAIN_ID="ds1000"
        DOMAIN_NAME="DS-1000"
        SUBSET="${CMD_SUBSET:-$DS1000_SUBSET}"
        EVOLVE_SIZE="${CMD_EVOLVE_SIZE:-$DS1000_EVOLVE_SIZE}"
        TEST_SIZE="${CMD_TEST_SIZE:-$DS1000_TEST_SIZE}"
        if [ -n "$SUBSET" ]; then
            SHUFFLE_FILE="$SHUFFLED_DATA_DIR/ds1000_${SUBSET}_shuffled.json"
        else
            SHUFFLE_FILE="$SHUFFLED_DATA_DIR/ds1000_shuffled.json"
        fi
        ;;
    math)
        DOMAIN_ID="math"
        DOMAIN_NAME="Math"
        EVOLVE_DATASET="$MATH_EVOLVE_DATASET"
        TEST_DATASETS="$MATH_TEST_DATASETS"
        TEST_SIZE="${CMD_TEST_SIZE:-$MATH_TEST_SIZE}"
        ;;
    appworld)
        DOMAIN_ID="appworld"
        DOMAIN_NAME="AppWorld"
        ;;
    *)
        echo -e "${RED}Error: Unknown domain '$DOMAIN'${NC}"
        echo "Available: swe, dacode, ds1000, math, appworld"
        exit 1
        ;;
esac

if [ -z "$TEST_SIZE" ]; then
    TEST_SIZE_DISPLAY="all"
    TEST_SIZE_VALUE=9999
else
    TEST_SIZE_DISPLAY="$TEST_SIZE"
    TEST_SIZE_VALUE=$TEST_SIZE
fi

###############################################################################
# Display Configuration
###############################################################################
echo "========================================"
echo -e "${CYAN} ReCreate Evolution ${NC}"
echo "========================================"
echo -e "Domain: ${GREEN}$DOMAIN_NAME${NC} ($DOMAIN_ID)"

case $DOMAIN in
    swe|dacode|ds1000)
        [ -n "$SUBSET" ] && echo -e "Subset: ${GREEN}$SUBSET${NC}"
        echo -e "Evolution: ${GREEN}$EVOLVE_SIZE instances${NC}"
        echo -e "Test: ${GREEN}$TEST_SIZE_DISPLAY${NC}"
        ;;
    math)
        echo -e "Evolution: ${GREEN}$EVOLVE_DATASET${NC}"
        echo -e "Test: ${GREEN}$TEST_DATASETS${NC}"
        ;;
    appworld)
        echo -e "Evolution: ${GREEN}$APP_EVOLVE_DATASET ($APP_EVOLVE_SIZE instances)${NC}"
        echo -e "Test: ${GREEN}test_normal + test_challenge${NC}"
        ;;
esac

echo ""
echo "Models: Agent=$AGENT_MODEL, ReCreate=$RECREATE_MODEL"
echo "Parallel: batch=$BATCH_SIZE, workers=$WORKERS"
[ "$N_REPEAT" -gt 1 ] && echo "Rounds: $N_REPEAT"
echo "Test: $IF_TEST"
[ "$USE_EXISTING" = true ] && echo -e "Mode: ${YELLOW}Using existing${NC} $EXPERIMENT_DIR"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

###############################################################################
# Evolution Phase
###############################################################################
if [ "$USE_EXISTING" = true ]; then
    echo -e "${YELLOW}Skipping evolution, using: $EXPERIMENT_DIR${NC}"
    
    if [ ! -d "$EXPERIMENT_DIR" ]; then
        echo -e "${RED}Error: Directory not found: $EXPERIMENT_DIR${NC}"
        exit 1
    fi
else
    echo -e "${CYAN}[1/2] Evolution...${NC}"
    
    CMD="python3 $SCRIPT_DIR/parallel_evolve.py"
    CMD="$CMD --domain $DOMAIN_ID"
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --n-repeat $N_REPEAT"
    CMD="$CMD --agent-model $AGENT_MODEL"
    CMD="$CMD --recreate-model $RECREATE_MODEL"
    CMD="$CMD --agent-temp $AGENT_TEMP"
    CMD="$CMD --recreate-temp $RECREATE_TEMP"
    CMD="$CMD --output-dir $OUTPUT_DIR"
    
    case $DOMAIN in
        swe|dacode)
            CMD="$CMD --subset $SUBSET"
            CMD="$CMD --max-instances $EVOLVE_SIZE"
            [ -n "$SHUFFLE_FILE" ] && [ -f "$SHUFFLE_FILE" ] && CMD="$CMD --shuffle-file $SHUFFLE_FILE"
            ;;
        ds1000)
            [ -n "$SUBSET" ] && CMD="$CMD --subset $SUBSET"
            CMD="$CMD --max-instances $EVOLVE_SIZE"
            [ -n "$SHUFFLE_FILE" ] && [ -f "$SHUFFLE_FILE" ] && CMD="$CMD --shuffle-file $SHUFFLE_FILE"
            ;;
        math)
            CMD="$CMD --data-source $EVOLVE_DATASET"
            CMD="$CMD --max-instances 999"
            ;;
        appworld)
            CMD="$CMD --dataset-name $APP_EVOLVE_DATASET"
            CMD="$CMD --max-instances $APP_EVOLVE_SIZE"
            ;;
    esac
    
    [ -n "$EXPERIMENT_NAME" ] && CMD="$CMD --experiment-name $EXPERIMENT_NAME"
    
    [ "$ABLATION_TRAJECTORY" = "false" ] && CMD="$CMD --no-ablation-trajectory"
    [ "$ABLATION_ENVIRONMENT" = "false" ] && CMD="$CMD --no-ablation-environment"
    [ "$ABLATION_EVAL_RESULTS" = "false" ] && CMD="$CMD --no-ablation-eval-results"
    [ "$ABLATION_MODIFICATION_GUIDANCE" = "false" ] && CMD="$CMD --no-ablation-modification-guidance"
    
    echo "Command: $CMD"
    echo ""
    
    eval $CMD
    
    # Find experiment directory
    if [ -n "$EXPERIMENT_NAME" ]; then
        EXPERIMENT_DIR=$(ls -td $OUTPUT_DIR/${EXPERIMENT_NAME}_${DOMAIN_ID}_* 2>/dev/null | head -1)
    fi
    [ -z "$EXPERIMENT_DIR" ] && EXPERIMENT_DIR=$(ls -td $OUTPUT_DIR/${DOMAIN_ID}_* 2>/dev/null | head -1)
    [ -z "$EXPERIMENT_DIR" ] && EXPERIMENT_DIR=$(ls -td $OUTPUT_DIR/*_${DOMAIN_ID}_* 2>/dev/null | head -1)
    
    if [ -z "$EXPERIMENT_DIR" ]; then
        echo -e "${RED}Error: Experiment directory not found${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}Evolution complete!${NC}"
    echo "Directory: $EXPERIMENT_DIR"
fi

###############################################################################
# Testing Phase
###############################################################################
if [ "$IF_TEST" = false ]; then
    echo ""
    echo -e "${YELLOW}Skipping test (--no-test)${NC}"
else
    echo ""
    echo -e "${CYAN}[2/2] Testing...${NC}"
    
    SCAFFOLD_DIR="$EXPERIMENT_DIR/workspace/scaffold_final"
    [ ! -d "$SCAFFOLD_DIR" ] && SCAFFOLD_DIR="$EXPERIMENT_DIR/workspace/current"
    EVOLVED_SCAFFOLD="$SCAFFOLD_DIR/scaffold.yaml"
    
    if [ ! -f "$EVOLVED_SCAFFOLD" ]; then
        echo -e "${RED}Error: Scaffold not found: $EVOLVED_SCAFFOLD${NC}"
        exit 1
    fi
    
    TEST_OUTPUT_DIR="$EXPERIMENT_DIR/test_results"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    echo "Scaffold: $EVOLVED_SCAFFOLD"
    echo "Output: $TEST_OUTPUT_DIR"
    echo ""
    
    case $DOMAIN in
        swe)
            python3 $SCRIPT_DIR/tests/run_swebench_test.py \
                --scaffold "$EVOLVED_SCAFFOLD" \
                --subset "$SUBSET" \
                --skip-instances $EVOLVE_SIZE \
                --max-instances $TEST_SIZE_VALUE \
                --output-dir "$TEST_OUTPUT_DIR/evolved" \
                --model "$AGENT_MODEL" \
                --temperature $AGENT_TEMP \
                --parallel $WORKERS \
                --tools-dir "$SCAFFOLD_DIR"
            ;;
        dacode)
            python3 $SCRIPT_DIR/tests/run_dacode_test.py \
                --scaffold "$EVOLVED_SCAFFOLD" \
                --subset "$SUBSET" \
                --skip-instances $EVOLVE_SIZE \
                --max-instances $TEST_SIZE_VALUE \
                --output-dir "$TEST_OUTPUT_DIR/evolved" \
                --model "$AGENT_MODEL" \
                --temperature $AGENT_TEMP \
                --parallel $WORKERS \
                --tools-dir "$SCAFFOLD_DIR"
            ;;
        ds1000)
            DS1000_TEST_ARGS=""
            [ -n "$SUBSET" ] && DS1000_TEST_ARGS="--subset $SUBSET"
            python3 $SCRIPT_DIR/tests/run_ds1000_test.py \
                --scaffold "$EVOLVED_SCAFFOLD" \
                $DS1000_TEST_ARGS \
                --skip $EVOLVE_SIZE \
                --max-instances $TEST_SIZE_VALUE \
                --output "$TEST_OUTPUT_DIR/evolved" \
                --model "$AGENT_MODEL" \
                --temperature $AGENT_TEMP \
                --workers $WORKERS \
                --tools-dir "$SCAFFOLD_DIR"
            ;;
        math)
            IFS=',' read -ra DATASETS <<< "$TEST_DATASETS"
            for dataset in "${DATASETS[@]}"; do
                echo ""
                echo -e "${CYAN}Testing: $dataset${NC}"
                python3 $SCRIPT_DIR/tests/run_math_test.py \
                    --scaffold "$EVOLVED_SCAFFOLD" \
                    --data-source "$dataset" \
                    --max-instances $TEST_SIZE_VALUE \
                    --output "$TEST_OUTPUT_DIR/evolved_$dataset" \
                    --model "$AGENT_MODEL" \
                    --temperature $AGENT_TEMP \
                    --workers $WORKERS \
                    --tools-dir "$SCAFFOLD_DIR"
            done
            ;;
        appworld)
            echo -e "${CYAN}Testing: test_normal${NC}"
            APP_NORMAL_ARGS=""
            [ -n "$APP_TEST_NORMAL_SIZE" ] && APP_NORMAL_ARGS="--max-instances $APP_TEST_NORMAL_SIZE"
            python3 $SCRIPT_DIR/tests/run_appworld_test.py \
                --scaffold-path "$EVOLVED_SCAFFOLD" \
                --dataset test_normal \
                $APP_NORMAL_ARGS \
                --output-dir "$TEST_OUTPUT_DIR/evolved_test_normal" \
                --model "$AGENT_MODEL" \
                --temperature $AGENT_TEMP \
                --workers $WORKERS \
                --tools-dir "$SCAFFOLD_DIR"
            
            echo ""
            echo -e "${CYAN}Testing: test_challenge${NC}"
            APP_CHALLENGE_ARGS=""
            [ -n "$APP_TEST_CHALLENGE_SIZE" ] && APP_CHALLENGE_ARGS="--max-instances $APP_TEST_CHALLENGE_SIZE"
            python3 $SCRIPT_DIR/tests/run_appworld_test.py \
                --scaffold-path "$EVOLVED_SCAFFOLD" \
                --dataset test_challenge \
                $APP_CHALLENGE_ARGS \
                --output-dir "$TEST_OUTPUT_DIR/evolved_test_challenge" \
                --model "$AGENT_MODEL" \
                --temperature $AGENT_TEMP \
                --workers $WORKERS \
                --tools-dir "$SCAFFOLD_DIR"
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}Testing complete!${NC}"
    
    # Analyze tool/memory usage
    echo ""
    echo -e "${CYAN}Analyzing tool/memory usage...${NC}"
    case $DOMAIN in
        swe|dacode|ds1000)
            python3 $SCRIPT_DIR/analyze_tool_memory_usage.py --test-dir "$TEST_OUTPUT_DIR/evolved" 2>/dev/null || true
            ;;
        math)
            IFS=',' read -ra DATASETS <<< "$TEST_DATASETS"
            for dataset in "${DATASETS[@]}"; do
                [ -d "$TEST_OUTPUT_DIR/evolved_$dataset" ] && {
                    echo ""
                    echo "--- $dataset ---"
                    python3 $SCRIPT_DIR/analyze_tool_memory_usage.py --test-dir "$TEST_OUTPUT_DIR/evolved_$dataset" 2>/dev/null || true
                }
            done
            ;;
        appworld)
            for split in test_normal test_challenge; do
                [ -d "$TEST_OUTPUT_DIR/evolved_$split" ] && {
                    echo ""
                    echo "--- $split ---"
                    python3 $SCRIPT_DIR/analyze_tool_memory_usage.py --test-dir "$TEST_OUTPUT_DIR/evolved_$split" 2>/dev/null || true
                }
            done
            ;;
    esac
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo "========================================"
echo -e "${GREEN} Complete! ${NC}"
echo "========================================"
echo "Experiment: $EXPERIMENT_DIR"

if [ "$IF_TEST" = true ]; then
    echo ""
    echo "Results:"
    case $DOMAIN in
        swe|dacode|ds1000)
            echo "  $TEST_OUTPUT_DIR/evolved/summary.json"
            ;;
        math)
            IFS=',' read -ra DATASETS <<< "$TEST_DATASETS"
            for dataset in "${DATASETS[@]}"; do
                echo "  $dataset: $TEST_OUTPUT_DIR/evolved_$dataset/summary.json"
            done
            ;;
        appworld)
            echo "  test_normal: $TEST_OUTPUT_DIR/evolved_test_normal/summary.json"
            echo "  test_challenge: $TEST_OUTPUT_DIR/evolved_test_challenge/summary.json"
            ;;
    esac
fi

echo ""
echo "Logs: $EXPERIMENT_DIR/evolution.log"
