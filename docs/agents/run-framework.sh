#!/bin/bash
# =============================================================================
# Claude Code Agents Framework Runner (Linux/macOS)
# =============================================================================
# Usage:
#   ./run-framework.sh           # Run all steps interactively
#   ./run-framework.sh --all     # Run all steps without pausing
#   ./run-framework.sh 1         # Run only step 1
#   ./run-framework.sh 4.6       # Run only step 4.6
#   ./run-framework.sh 1 3       # Run steps 1 through 3
#   ./run-framework.sh --list    # List all available steps
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root is two levels up from script location (docs/agents -> root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Framework prompts directory relative to project root
FRAMEWORK_DIR="docs/agents"

# Define steps in order
declare -a STEPS=(
    "01-step1-find-code-patterns.md"
    "02-step2-teach-patterns-to-ai.md"
    "03-step3-create-ai-assistants.md"
    "04-step4-connect-everything.md"
    "04.5-step4.5-wire-agents-to-skills.md"
    "04.6-step4.6-register-in-claude-md.md"
    "05-step5-test-on-real-work.md"
    "06-step6-automatic-improvements.md"
    "07-weekly-check-system-health.md"
    "08-monthly-clean-up-duplicates.md"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "  Claude Code Agents Framework Runner"
    echo "=============================================="
    echo -e "${NC}"
    echo -e "Project root: ${GREEN}$PROJECT_ROOT${NC}"
    echo -e "Prompts dir:  ${GREEN}$FRAMEWORK_DIR${NC}"
    echo ""
}

# Print step info
print_step() {
    local step_num=$1
    local step_file=$2
    echo -e "${GREEN}"
    echo "=============================================="
    echo "  Step $step_num: $step_file"
    echo "=============================================="
    echo -e "${NC}"
}

# Run a single step
run_step() {
    local step_file=$1
    local step_path="$PROJECT_ROOT/$FRAMEWORK_DIR/$step_file"

    if [[ ! -f "$step_path" ]]; then
        echo -e "${RED}Error: File not found: $step_path${NC}"
        return 1
    fi

    echo -e "${YELLOW}Changing to project root: $PROJECT_ROOT${NC}"
    cd "$PROJECT_ROOT"

    echo -e "${YELLOW}Sending prompt to Claude...${NC}"
    echo ""

    # Pipe the file content to claude (running from project root)
    cat "$step_path" | claude -p

    echo ""
    echo -e "${GREEN}Step completed.${NC}"
}

# Get step index by step number (e.g., "1", "4.5", "4.6")
get_step_index() {
    local step_num=$1
    local index=0

    for step in "${STEPS[@]}"; do
        # Extract step number from filename (e.g., "01" from "01-step1-...")
        if [[ "$step" =~ ^0?${step_num}[.-] ]] || [[ "$step" =~ ^${step_num}-step ]]; then
            echo $index
            return 0
        fi
        ((index++))
    done

    echo -1
    return 1
}

# Show usage
show_usage() {
    echo "Usage:"
    echo "  $0                  Run all steps interactively (pause between steps)"
    echo "  $0 --all            Run all steps without pausing"
    echo "  $0 --list           List all available steps"
    echo "  $0 <step>           Run a specific step (e.g., 1, 4.5, 4.6)"
    echo "  $0 <start> <end>    Run steps from start to end"
    echo ""
    echo "Examples:"
    echo "  $0 1                Run step 1 only"
    echo "  $0 4.6              Run step 4.6 only"
    echo "  $0 1 3              Run steps 1, 2, and 3"
    echo "  $0 4 4.6            Run steps 4, 4.5, and 4.6"
    echo ""
    echo "Note: Claude CLI runs from project root: $PROJECT_ROOT"
}

# List all steps
list_steps() {
    echo "Available steps:"
    echo ""
    local index=1
    for step in "${STEPS[@]}"; do
        echo "  $index. $step"
        ((index++))
    done
    echo ""
    echo "Use step identifiers: 1, 2, 3, 4, 4.5, 4.6, 5, 6, 7, 8"
}

# Verify project root has expected structure
verify_project_root() {
    if [[ ! -f "$PROJECT_ROOT/CLAUDE.md" ]]; then
        echo -e "${RED}Error: CLAUDE.md not found at project root: $PROJECT_ROOT${NC}"
        echo "Make sure this script is located in docs/agents/"
        exit 1
    fi

    if [[ ! -d "$PROJECT_ROOT/.claude" ]]; then
        echo -e "${YELLOW}Note: .claude folder not found yet at project root${NC}"
        echo "The framework will create it during step execution."
        echo ""
    fi
}

# Main execution
main() {
    print_banner
    verify_project_root

    # Parse arguments
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --list|-l)
            list_steps
            exit 0
            ;;
        --all|-a)
            # Run all steps without pausing
            local step_num=1
            for step in "${STEPS[@]}"; do
                print_step $step_num "$step"
                run_step "$step"
                echo ""
                ((step_num++))
            done
            echo -e "${GREEN}All steps completed!${NC}"
            exit 0
            ;;
        "")
            # Interactive mode - run all steps with pauses
            local step_num=1
            for step in "${STEPS[@]}"; do
                print_step $step_num "$step"
                run_step "$step"
                echo ""

                # Don't pause after the last step
                if [[ $step_num -lt ${#STEPS[@]} ]]; then
                    echo -e "${YELLOW}Press Enter to continue to next step (or Ctrl+C to stop)...${NC}"
                    read -r
                fi
                ((step_num++))
            done
            echo -e "${GREEN}All steps completed!${NC}"
            exit 0
            ;;
        *)
            # Run specific step(s)
            if [[ -n "${2:-}" ]]; then
                # Range mode: run from step $1 to step $2
                local start_idx=$(get_step_index "$1")
                local end_idx=$(get_step_index "$2")

                if [[ $start_idx -eq -1 ]]; then
                    echo -e "${RED}Error: Step '$1' not found${NC}"
                    exit 1
                fi

                if [[ $end_idx -eq -1 ]]; then
                    echo -e "${RED}Error: Step '$2' not found${NC}"
                    exit 1
                fi

                for ((i=start_idx; i<=end_idx; i++)); do
                    print_step $((i+1)) "${STEPS[$i]}"
                    run_step "${STEPS[$i]}"
                    echo ""
                done
                echo -e "${GREEN}Selected steps completed!${NC}"
            else
                # Single step mode
                local step_idx=$(get_step_index "$1")

                if [[ $step_idx -eq -1 ]]; then
                    echo -e "${RED}Error: Step '$1' not found${NC}"
                    echo ""
                    list_steps
                    exit 1
                fi

                print_step $((step_idx+1)) "${STEPS[$step_idx]}"
                run_step "${STEPS[$step_idx]}"
                echo -e "${GREEN}Step completed!${NC}"
            fi
            ;;
    esac
}

main "$@"
