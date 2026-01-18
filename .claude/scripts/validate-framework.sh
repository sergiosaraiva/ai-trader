#!/bin/bash
# Framework Validation Script
# Validates YAML frontmatter in all skills and agents
# Run: .claude/scripts/validate-framework.sh
# Exit codes: 0 = pass, 1 = errors found

# Don't use set -e since we're checking return codes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CLAUDE_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

errors=0
warnings=0
skills_checked=0
agents_checked=0

echo "======================================"
echo "  Framework Validation"
echo "======================================"
echo ""

# Function to validate YAML frontmatter
validate_yaml_frontmatter() {
    local file="$1"
    local type="$2"  # "skill" or "agent"
    local expected_name="$3"  # For skills: skill name, for agents: filename without .md

    local has_errors=0

    # Check file starts with ---
    if [ "$(head -1 "$file")" != "---" ]; then
        echo -e "  ${RED}ERROR${NC}: File doesn't start with '---'"
        has_errors=1
    fi

    # Check for closing --- in first 50 lines (agents can have longer frontmatter due to multiline descriptions)
    local close_line=$(head -50 "$file" | grep -n "^---$" | tail -1 | cut -d: -f1)
    if [ -z "$close_line" ] || [ "$close_line" -lt 2 ]; then
        echo -e "  ${RED}ERROR${NC}: Missing or misplaced closing '---' for frontmatter"
        has_errors=1
    fi

    # Check name field exists
    if ! grep -q "^name:" "$file"; then
        echo -e "  ${RED}ERROR${NC}: Missing 'name:' field"
        has_errors=1
    else
        # Check name matches expected value
        local actual_name=$(grep "^name:" "$file" | head -1 | sed 's/name: *//')
        if [ -n "$expected_name" ] && [ "$actual_name" != "$expected_name" ]; then
            echo -e "  ${RED}ERROR${NC}: name '$actual_name' doesn't match expected '$expected_name'"
            has_errors=1
        fi

        # Check name length (max 64 chars)
        local name_len=${#actual_name}
        if [ "$name_len" -gt 64 ]; then
            echo -e "  ${RED}ERROR${NC}: name exceeds 64 characters ($name_len chars)"
            has_errors=1
        fi

        # Check name format (lowercase, numbers, hyphens only)
        if [ -n "$actual_name" ] && ! echo "$actual_name" | grep -qE "^[a-z0-9-]+$"; then
            echo -e "  ${RED}ERROR${NC}: name must contain only lowercase letters, numbers, and hyphens"
            has_errors=1
        fi
    fi

    # Check description field exists
    if ! grep -q "^description:" "$file"; then
        echo -e "  ${RED}ERROR${NC}: Missing 'description:' field"
        has_errors=1
    fi

    # Agent-specific checks
    if [ "$type" = "agent" ]; then
        # Check model field
        if ! grep -q "^model:" "$file"; then
            echo -e "  ${RED}ERROR${NC}: Missing 'model:' field (required for agents)"
            has_errors=1
        else
            local model=$(grep "^model:" "$file" | head -1 | sed 's/model: *//')
            case "$model" in
                inherit|opus|sonnet|haiku) ;;
                *)
                    echo -e "  ${RED}ERROR${NC}: Invalid model '$model' (must be: inherit, opus, sonnet, or haiku)"
                    has_errors=1
                    ;;
            esac
        fi

        # Check color field (warning only)
        if ! grep -q "^color:" "$file"; then
            echo -e "  ${YELLOW}WARNING${NC}: Missing 'color:' field"
            ((warnings++))
        fi
    fi

    # Skill-specific checks
    if [ "$type" = "skill" ]; then
        # Check version field (warning only)
        if ! grep -q "^version:" "$file"; then
            echo -e "  ${YELLOW}WARNING${NC}: Missing 'version:' field (recommended)"
            ((warnings++))
        fi
    fi

    return $has_errors
}

# Validate Skills
echo "Validating Skills..."
echo "--------------------------------------"

# Find all SKILL.md files
for skill_dir in "$CLAUDE_DIR"/skills/*/; do
    if [ -d "$skill_dir" ]; then
        folder_name=$(basename "$skill_dir")
        skill_file="$skill_dir/SKILL.md"

        # Skip archived skills
        if [ "$folder_name" = "_archived" ]; then
            continue
        fi

        if [ -f "$skill_file" ]; then
            ((skills_checked++))

            # Extract actual name from file
            actual_name=$(grep "^name:" "$skill_file" | head -1 | sed 's/name: *//')

            # Determine expected name based on pattern:
            # If folder is a category (backend, frontend, etc.) with generic SKILL.md,
            # the name in file is the skill name (doesn't need to match folder)
            # If folder matches common skill naming (gerund form like routing-to-skills),
            # the name should match folder
            if echo "$folder_name" | grep -qE "^(backend|frontend|database|build-deployment|testing|feature-engineering|data-layer|trading-domain|quality-testing)$"; then
                # Category folder - use actual name from file
                expected_name="$actual_name"
            else
                # Skill-specific folder - name should match folder
                expected_name="$folder_name"
            fi

            echo -e "\n${NC}Checking: $folder_name/SKILL.md (name: $actual_name)"

            if ! validate_yaml_frontmatter "$skill_file" "skill" "$expected_name"; then
                ((errors++))
            else
                echo -e "  ${GREEN}PASS${NC}"
            fi
        fi

        # Also check other .md files in the directory (sub-skills)
        for alt_file in "$skill_dir"/*.md; do
            if [ -f "$alt_file" ] && [ "$(basename "$alt_file")" != "SKILL.md" ]; then
                alt_name=$(basename "$alt_file" .md)
                ((skills_checked++))
                echo -e "\n${NC}Checking: $folder_name/$alt_name.md"

                if ! validate_yaml_frontmatter "$alt_file" "skill" "$alt_name"; then
                    ((errors++))
                else
                    echo -e "  ${GREEN}PASS${NC}"
                fi
            fi
        done
    fi
done

# Validate Agents
echo ""
echo "--------------------------------------"
echo "Validating Agents..."
echo "--------------------------------------"

for agent_file in "$CLAUDE_DIR"/agents/*.md; do
    if [ -f "$agent_file" ]; then
        agent_name=$(basename "$agent_file" .md)
        ((agents_checked++))
        echo -e "\n${NC}Checking: $agent_name.md"

        if ! validate_yaml_frontmatter "$agent_file" "agent" "$agent_name"; then
            ((errors++))
        else
            echo -e "  ${GREEN}PASS${NC}"
        fi
    fi
done

# Summary
echo ""
echo "======================================"
echo "  Validation Summary"
echo "======================================"
echo ""
echo "Skills checked: $skills_checked"
echo "Agents checked: $agents_checked"
echo ""

if [ $errors -gt 0 ]; then
    echo -e "${RED}ERRORS: $errors${NC}"
fi

if [ $warnings -gt 0 ]; then
    echo -e "${YELLOW}WARNINGS: $warnings${NC}"
fi

if [ $errors -eq 0 ]; then
    echo -e "${GREEN}All validations passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Framework validation failed with $errors error(s).${NC}"
    echo "Fix the errors above before committing."
    exit 1
fi
