#!/bin/bash
# Pre-commit hook to validate framework files before commit
#
# Installation:
#   cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or to append to existing pre-commit:
#   cat .claude/hooks/pre-commit-framework-check.sh >> .git/hooks/pre-commit

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Running framework validation on staged files..."
echo ""

# Check if any framework files are being committed
framework_files=$(git diff --cached --name-only | grep -E "^\.claude/(skills|agents)/" || true)

if [ -z "$framework_files" ]; then
    echo -e "${GREEN}No framework files in commit, skipping validation.${NC}"
    exit 0
fi

echo "Framework files staged for commit:"
echo "$framework_files" | sed 's/^/  /'
echo ""

errors=0

# Validate each modified skill
while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    # Skip if file was deleted
    if [ ! -f "$file" ]; then
        continue
    fi

    # Skills validation
    if echo "$file" | grep -q "\.claude/skills/.*/.*\.md$"; then
        folder=$(dirname "$file" | xargs basename)
        filename=$(basename "$file" .md)

        # For SKILL.md, expected name is folder name
        # For other files, expected name is filename
        if [ "$(basename "$file")" = "SKILL.md" ]; then
            expected_name="$folder"
        else
            expected_name="$filename"
        fi

        echo "Validating skill: $file"

        # Check YAML frontmatter starts with ---
        if [ "$(head -1 "$file")" != "---" ]; then
            echo -e "  ${RED}ERROR${NC}: File doesn't start with '---'"
            ((errors++))
        fi

        # Check name field
        if ! grep -q "^name:" "$file"; then
            echo -e "  ${RED}ERROR${NC}: Missing 'name:' field"
            ((errors++))
        else
            actual_name=$(grep "^name:" "$file" | head -1 | sed 's/name: *//')
            if [ "$actual_name" != "$expected_name" ]; then
                echo -e "  ${RED}ERROR${NC}: name '$actual_name' doesn't match expected '$expected_name'"
                ((errors++))
            fi
        fi

        # Check description field
        if ! grep -q "^description:" "$file"; then
            echo -e "  ${RED}ERROR${NC}: Missing 'description:' field"
            ((errors++))
        fi

        # Check name format
        if grep -q "^name:" "$file"; then
            name=$(grep "^name:" "$file" | head -1 | sed 's/name: *//')
            if ! echo "$name" | grep -qE "^[a-z0-9-]+$"; then
                echo -e "  ${RED}ERROR${NC}: name must contain only lowercase letters, numbers, and hyphens"
                ((errors++))
            fi
            if [ ${#name} -gt 64 ]; then
                echo -e "  ${RED}ERROR${NC}: name exceeds 64 characters"
                ((errors++))
            fi
        fi
    fi

    # Agents validation
    if echo "$file" | grep -q "\.claude/agents/.*\.md$"; then
        filename=$(basename "$file" .md)
        echo "Validating agent: $file"

        # Check YAML frontmatter
        if [ "$(head -1 "$file")" != "---" ]; then
            echo -e "  ${RED}ERROR${NC}: File doesn't start with '---'"
            ((errors++))
        fi

        # Check name field matches filename
        if ! grep -q "^name:" "$file"; then
            echo -e "  ${RED}ERROR${NC}: Missing 'name:' field"
            ((errors++))
        else
            actual_name=$(grep "^name:" "$file" | head -1 | sed 's/name: *//')
            if [ "$actual_name" != "$filename" ]; then
                echo -e "  ${RED}ERROR${NC}: name '$actual_name' doesn't match filename '$filename'"
                ((errors++))
            fi
        fi

        # Check description field
        if ! grep -q "^description:" "$file"; then
            echo -e "  ${RED}ERROR${NC}: Missing 'description:' field"
            ((errors++))
        fi

        # Check model field
        if ! grep -q "^model:" "$file"; then
            echo -e "  ${RED}ERROR${NC}: Missing 'model:' field"
            ((errors++))
        else
            model=$(grep "^model:" "$file" | head -1 | sed 's/model: *//')
            case "$model" in
                inherit|opus|sonnet|haiku) ;;
                *)
                    echo -e "  ${RED}ERROR${NC}: Invalid model '$model' (must be: inherit, opus, sonnet, or haiku)"
                    ((errors++))
                    ;;
            esac
        fi

        # Check color field (warning)
        if ! grep -q "^color:" "$file"; then
            echo -e "  ${YELLOW}WARNING${NC}: Missing 'color:' field (recommended)"
        else
            color=$(grep "^color:" "$file" | head -1 | sed 's/color: *//')
            case "$color" in
                blue|cyan|green|yellow|magenta|red) ;;
                *)
                    echo -e "  ${YELLOW}WARNING${NC}: Unusual color '$color'"
                    ;;
            esac
        fi
    fi

done <<< "$framework_files"

echo ""

if [ $errors -gt 0 ]; then
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}  COMMIT BLOCKED: $errors error(s) found${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo "Fix the errors above and try again."
    echo ""
    echo "To run full validation:"
    echo "  .claude/scripts/validate-framework.sh"
    echo ""
    echo "To bypass this hook (not recommended):"
    echo "  git commit --no-verify"
    exit 1
fi

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  All framework validations passed!${NC}"
echo -e "${GREEN}======================================${NC}"
exit 0
