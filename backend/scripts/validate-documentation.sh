#!/bin/bash
# validate-documentation.sh
# Validates Phase 9 documentation for completeness, correctness, and consistency

# Don't use set -e as we expect some checks to fail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Report file
REPORT_FILE="/home/sergio/ai-trader/docs/DOCUMENTATION-VALIDATION-REPORT.md"

# Initialize report
init_report() {
    # Create docs directory if it doesn't exist
    mkdir -p "$(dirname "$REPORT_FILE")"

    cat > "$REPORT_FILE" << EOF
# Documentation Validation Report

**Generated**: $(date +"%Y-%m-%d %H:%M:%S")
**Validator**: validate-documentation.sh
**Phase**: Phase 9 - Documentation

## Executive Summary

EOF
}

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_CHECKS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Start report section
start_section() {
    echo ""
    echo "## $1"
    echo ""
    {
        echo ""
        echo "## $1"
        echo ""
    } >> "$REPORT_FILE"
}

# Record check result
record_check() {
    local status=$1
    local description=$2
    local details=$3

    ((TOTAL_CHECKS++))

    if [ "$status" = "PASS" ]; then
        log_success "$description"
        echo "- ✅ **PASS**: $description" >> "$REPORT_FILE"
    else
        log_error "$description"
        echo "- ❌ **FAIL**: $description" >> "$REPORT_FILE"
        if [ -n "$details" ]; then
            echo "  - $details" >> "$REPORT_FILE"
        fi
    fi
}

# Check file existence
check_file_existence() {
    start_section "1. File Existence Validation"

    local files=(
        "/home/sergio/ai-trader/docs/AI-TRADING-AGENT.md"
        "/home/sergio/ai-trader/docs/AGENT-OPERATIONS-GUIDE.md"
        "/home/sergio/ai-trader/docs/AGENT-API-REFERENCE.md"
        "/home/sergio/ai-trader/docs/CHANGELOG.md"
        "/home/sergio/ai-trader/docs/AGENT-QUICK-REFERENCE.md"
        "/home/sergio/ai-trader/README.md"
    )

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            record_check "PASS" "File exists: $(basename $file)" ""
        else
            record_check "FAIL" "File missing: $(basename $file)" "Expected at: $file"
        fi
    done
}

# Check markdown syntax
check_markdown_syntax() {
    start_section "2. Markdown Syntax Validation"

    local docs_dir="/home/sergio/ai-trader/docs"
    local doc_files=(
        "$docs_dir/AI-TRADING-AGENT.md"
        "$docs_dir/AGENT-OPERATIONS-GUIDE.md"
        "$docs_dir/AGENT-API-REFERENCE.md"
        "$docs_dir/CHANGELOG.md"
        "$docs_dir/AGENT-QUICK-REFERENCE.md"
    )

    for file in "${doc_files[@]}"; do
        if [ ! -f "$file" ]; then
            continue
        fi

        local filename=$(basename "$file")

        # Check for headers
        if grep -q "^# " "$file"; then
            record_check "PASS" "$filename has top-level headers" ""
        else
            record_check "FAIL" "$filename missing top-level headers" "No '# Header' found"
        fi

        # Check for unclosed code blocks
        local backtick_count=$(grep -c '```' "$file" || true)
        if [ $((backtick_count % 2)) -eq 0 ]; then
            record_check "PASS" "$filename has balanced code blocks" ""
        else
            record_check "FAIL" "$filename has unbalanced code blocks" "Found $backtick_count backtick markers - should be even"
        fi

        # Check for broken tables (basic check)
        if grep -q "^|" "$file"; then
            local table_lines=$(grep "^|" "$file" | wc -l)
            if [ "$table_lines" -gt 0 ]; then
                record_check "PASS" "$filename contains tables" ""
            fi
        fi
    done
}

# Check internal links
check_internal_links() {
    start_section "3. Internal Link Validation"

    local docs_dir="/home/sergio/ai-trader/docs"
    local doc_files=(
        "$docs_dir/AI-TRADING-AGENT.md"
        "$docs_dir/AGENT-OPERATIONS-GUIDE.md"
        "$docs_dir/AGENT-API-REFERENCE.md"
        "$docs_dir/CHANGELOG.md"
        "$docs_dir/AGENT-QUICK-REFERENCE.md"
    )

    local broken_links=0

    for file in "${doc_files[@]}"; do
        if [ ! -f "$file" ]; then
            continue
        fi

        local filename=$(basename "$file")

        # Extract markdown links: [text](path)
        grep -oP '\[([^\]]+)\]\(([^)]+)\)' "$file" | while IFS= read -r link; do
            local target=$(echo "$link" | grep -oP '\(([^)]+)\)' | tr -d '()')

            # Skip external links (http/https)
            if [[ "$target" =~ ^https?:// ]]; then
                continue
            fi

            # Skip anchors only
            if [[ "$target" =~ ^# ]]; then
                continue
            fi

            # Resolve relative path
            local link_dir=$(dirname "$file")
            local full_path="$link_dir/$target"

            # Remove anchor if present
            full_path="${full_path%#*}"

            if [ ! -f "$full_path" ]; then
                record_check "FAIL" "$filename broken link: $target" "Target does not exist: $full_path"
                ((broken_links++))
            fi
        done
    done

    if [ "$broken_links" -eq 0 ]; then
        record_check "PASS" "All internal file links are valid" ""
    fi
}

# Check code block language tags
check_code_blocks() {
    start_section "4. Code Block Language Tag Validation"

    local docs_dir="/home/sergio/ai-trader/docs"
    local doc_files=(
        "$docs_dir/AI-TRADING-AGENT.md"
        "$docs_dir/AGENT-OPERATIONS-GUIDE.md"
        "$docs_dir/AGENT-API-REFERENCE.md"
        "$docs_dir/CHANGELOG.md"
        "$docs_dir/AGENT-QUICK-REFERENCE.md"
    )

    for file in "${doc_files[@]}"; do
        if [ ! -f "$file" ]; then
            continue
        fi

        local filename=$(basename "$file")

        # Count code blocks with and without language tags
        local total_blocks=$(grep -c '^```' "$file" || true)
        total_blocks=$((total_blocks / 2))  # Divide by 2 since each block has opening and closing

        local untagged_blocks=$(grep -c '^```$' "$file" || true)

        if [ "$total_blocks" -gt 0 ]; then
            if [ "$untagged_blocks" -eq 0 ]; then
                record_check "PASS" "$filename all code blocks have language tags - $total_blocks blocks" ""
            else
                record_check "FAIL" "$filename has untagged code blocks" "$untagged_blocks out of $total_blocks blocks missing language tags"
            fi
        fi
    done
}

# Check required sections
check_required_sections() {
    start_section "5. Required Sections Validation"

    local docs_dir="/home/sergio/ai-trader/docs"

    # AI-TRADING-AGENT.md sections
    local ai_agent_file="$docs_dir/AI-TRADING-AGENT.md"
    if [ -f "$ai_agent_file" ]; then
        local required_sections=(
            "Overview"
            "Architecture"
            "Installation"
            "Configuration"
            "Usage"
        )

        for section in "${required_sections[@]}"; do
            if grep -qi "^## $section\|^# $section" "$ai_agent_file"; then
                record_check "PASS" "AI-TRADING-AGENT.md has '$section' section" ""
            else
                record_check "FAIL" "AI-TRADING-AGENT.md missing '$section' section" "Required section not found"
            fi
        done
    fi

    # AGENT-OPERATIONS-GUIDE.md sections
    local ops_guide_file="$docs_dir/AGENT-OPERATIONS-GUIDE.md"
    if [ -f "$ops_guide_file" ]; then
        local required_sections=(
            "Deployment"
            "Monitoring"
            "Troubleshooting"
        )

        for section in "${required_sections[@]}"; do
            if grep -qi "^## $section\|^# $section" "$ops_guide_file"; then
                record_check "PASS" "AGENT-OPERATIONS-GUIDE.md has '$section' section" ""
            else
                record_check "FAIL" "AGENT-OPERATIONS-GUIDE.md missing '$section' section" "Required section not found"
            fi
        done
    fi

    # AGENT-API-REFERENCE.md sections
    local api_ref_file="$docs_dir/AGENT-API-REFERENCE.md"
    if [ -f "$api_ref_file" ]; then
        local required_sections=(
            "Endpoints"
            "Schemas"
            "Authentication"
        )

        for section in "${required_sections[@]}"; do
            if grep -qi "^## $section\|^# $section" "$api_ref_file"; then
                record_check "PASS" "AGENT-API-REFERENCE.md has '$section' section" ""
            else
                record_check "FAIL" "AGENT-API-REFERENCE.md missing '$section' section" "Required section not found"
            fi
        done
    fi

    # CHANGELOG.md format
    local changelog_file="$docs_dir/CHANGELOG.md"
    if [ -f "$changelog_file" ]; then
        if grep -q "^## \[" "$changelog_file"; then
            record_check "PASS" "CHANGELOG.md follows version format" ""
        else
            record_check "FAIL" "CHANGELOG.md doesn't follow version format" "Expected '## [X.Y.Z]' headers"
        fi
    fi
}

# Check environment variables documentation
check_env_vars() {
    start_section "6. Environment Variables Documentation"

    local env_example="/home/sergio/ai-trader/.env.example"
    local docs_dir="/home/sergio/ai-trader/docs"

    if [ ! -f "$env_example" ]; then
        record_check "FAIL" ".env.example file missing" "Cannot validate env var documentation"
        return
    fi

    # Extract env vars from .env.example
    local env_vars=$(grep -E '^[A-Z_]+=' "$env_example" | cut -d'=' -f1 | sort)
    local env_count=$(echo "$env_vars" | wc -l)

    # Check if docs mention these env vars
    local documented_count=0
    while IFS= read -r var; do
        if grep -rq "$var" "$docs_dir/"*.md; then
            ((documented_count++))
        fi
    done <<< "$env_vars"

    if [ "$documented_count" -eq "$env_count" ]; then
        record_check "PASS" "All environment variables documented - $env_count of $env_count" ""
    else
        record_check "FAIL" "Some environment variables not documented" "$documented_count of $env_count variables found in documentation"
    fi
}

# Check API endpoint documentation
check_api_endpoints() {
    start_section "7. API Endpoints Documentation"

    local routes_file="/home/sergio/ai-trader/backend/src/api/routes/agent.py"
    local api_ref_file="/home/sergio/ai-trader/docs/AGENT-API-REFERENCE.md"

    if [ ! -f "$routes_file" ]; then
        record_check "FAIL" "routes/agent.py not found" "Cannot validate API endpoint documentation"
        return
    fi

    if [ ! -f "$api_ref_file" ]; then
        record_check "FAIL" "AGENT-API-REFERENCE.md not found" "API documentation missing"
        return
    fi

    # Extract endpoints from routes/agent.py
    local endpoints=$(grep -oP '@router\.(get|post|put|delete|patch)\("(/[^"]+)' "$routes_file" | grep -oP '"/[^"]+' | tr -d '"' | sort -u)

    if [ -z "$endpoints" ]; then
        record_check "WARN" "No endpoints found in routes/agent.py" "File may be empty or use different routing pattern"
        return
    fi

    # Check each endpoint is documented
    local endpoint_count=0
    local documented_count=0

    while IFS= read -r endpoint; do
        ((endpoint_count++))
        if grep -q "$endpoint" "$api_ref_file"; then
            ((documented_count++))
        fi
    done <<< "$endpoints"

    if [ "$documented_count" -eq "$endpoint_count" ]; then
        record_check "PASS" "All API endpoints documented - $documented_count of $endpoint_count" ""
    else
        record_check "FAIL" "Some API endpoints not documented" "$documented_count of $endpoint_count endpoints found in documentation"
    fi
}

# Check version consistency
check_version_consistency() {
    start_section "8. Version Consistency Validation"

    local docs_dir="/home/sergio/ai-trader/docs"
    local doc_files=(
        "$docs_dir/AI-TRADING-AGENT.md"
        "$docs_dir/AGENT-OPERATIONS-GUIDE.md"
        "$docs_dir/AGENT-API-REFERENCE.md"
        "$docs_dir/CHANGELOG.md"
    )

    # Extract version numbers from all docs
    local versions=()

    for file in "${doc_files[@]}"; do
        if [ ! -f "$file" ]; then
            continue
        fi

        # Look for version patterns: v1.0.0, Version 1.0.0, etc.
        local found_versions=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "$file" | sort -u | head -1)

        if [ -n "$found_versions" ]; then
            versions+=("$found_versions")
        fi
    done

    # Check if all versions are the same
    if [ "${#versions[@]}" -gt 0 ]; then
        local first_version="${versions[0]}"
        local all_same=true

        for version in "${versions[@]}"; do
            if [ "$version" != "$first_version" ]; then
                all_same=false
                break
            fi
        done

        if [ "$all_same" = true ]; then
            record_check "PASS" "Version numbers consistent across documentation - v$first_version" ""
        else
            record_check "FAIL" "Version numbers inconsistent across documentation" "Found multiple versions: ${versions[*]}"
        fi
    else
        record_check "WARN" "No version numbers found in documentation" "Consider adding version information"
    fi
}

# Generate final summary
generate_summary() {
    local pass_rate=0
    if [ "$TOTAL_CHECKS" -gt 0 ]; then
        pass_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    fi

    # Update executive summary
    sed -i '/^## Executive Summary$/a\\n**Total Checks**: '"$TOTAL_CHECKS"'\n**Passed**: '"$PASSED_CHECKS"'\n**Failed**: '"$FAILED_CHECKS"'\n**Pass Rate**: '"$pass_rate"'%\n' "$REPORT_FILE"

    # Add conclusion
    cat >> "$REPORT_FILE" << EOF

## Conclusion

EOF

    if [ "$FAILED_CHECKS" -eq 0 ]; then
        echo "✅ **All documentation validation checks passed!**" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "The Phase 9 documentation is complete, consistent, and ready for use." >> "$REPORT_FILE"
    else
        echo "❌ **Some documentation validation checks failed.**" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "Please review the failed checks above and update the documentation accordingly." >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "### Recommendations" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "1. Fix all broken links and missing files" >> "$REPORT_FILE"
        echo "2. Add language tags to all code blocks" >> "$REPORT_FILE"
        echo "3. Ensure all required sections are present" >> "$REPORT_FILE"
        echo "4. Document all environment variables and API endpoints" >> "$REPORT_FILE"
        echo "5. Maintain version consistency across all documentation" >> "$REPORT_FILE"
    fi

    # Add validation metadata
    cat >> "$REPORT_FILE" << EOF

## Validation Metadata

- **Validator Version**: 1.0.0
- **Generated**: $(date +"%Y-%m-%d %H:%M:%S")
- **Total Runtime**: ${SECONDS}s
- **Exit Code**: $([ "$FAILED_CHECKS" -eq 0 ] && echo "0 (Success)" || echo "1 (Failures detected)")

---

*Generated by validate-documentation.sh*
EOF

    echo ""
    echo "================================"
    echo "   VALIDATION SUMMARY"
    echo "================================"
    echo "Total Checks:  $TOTAL_CHECKS"
    echo "Passed:        $PASSED_CHECKS"
    echo "Failed:        $FAILED_CHECKS"
    echo "Pass Rate:     $pass_rate%"
    echo "================================"
    echo ""
    echo "Report saved to: $REPORT_FILE"
    echo ""

    if [ "$FAILED_CHECKS" -eq 0 ]; then
        echo -e "${GREEN}All checks passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some checks failed. Please review the report.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting documentation validation..."
    log_info "Target: Phase 9 - Documentation"
    echo ""

    init_report

    check_file_existence
    check_markdown_syntax
    check_internal_links
    check_code_blocks
    check_required_sections
    check_env_vars
    check_api_endpoints
    check_version_consistency

    generate_summary
}

# Run main function
main
