#!/bin/bash
# Tool Manager for SWE-bench Workspace
# Manages reusable tools across evaluation tasks
set -e

# Auto-detect tools directory:
# - In ReCreate-Agent environment (host): use ./current/agent_tools
# - In Agent environment (Docker): use /workspace
if [[ -d "./current/agent_tools" ]]; then
    TOOLS_DIR="${TOOLS_DIR:-./current/agent_tools}"
elif [[ -d "/workspace" ]]; then
    TOOLS_DIR="${TOOLS_DIR:-/workspace}"
else
    TOOLS_DIR="${TOOLS_DIR:-./agent_tools}"
fi

# Extract field from YAML frontmatter
extract_frontmatter_field() {
    local file="$1"
    local field="$2"
    if [[ -f "$file" ]]; then
        # Extract content between --- markers, then get the field
        sed -n '/^---$/,/^---$/p' "$file" 2>/dev/null | grep "^${field}:" | sed "s/^${field}: *//" | head -1
    fi
}

cmd_list() {
    category="$1"
    if [[ -z "$category" ]]; then
        echo "Available tools:"
        echo ""
        find "$TOOLS_DIR" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | while read -r tool_path; do
            cat_name=$(basename $(dirname "$tool_path"))
            tool_name=$(basename "$tool_path")
            readme_file="$tool_path/README.md"
            
            # Extract description from YAML frontmatter
            desc=$(extract_frontmatter_field "$readme_file" "description")
            if [[ -z "$desc" ]]; then
                desc="No description"
            fi
            
            printf "  %-35s %s\n" "$cat_name/$tool_name" "- $desc"
        done
    else
        if [[ ! -d "$TOOLS_DIR/$category" ]]; then
            echo "Error: Category '$category' not found" >&2
            return 1
        fi
        echo "Tools in category '$category':"
        echo ""
        find "$TOOLS_DIR/$category" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | while read -r tool_path; do
            tool_name=$(basename "$tool_path")
            readme_file="$tool_path/README.md"
            
            desc=$(extract_frontmatter_field "$readme_file" "description")
            if [[ -z "$desc" ]]; then
                desc="No description"
            fi
            
            printf "  %-25s %s\n" "$tool_name" "- $desc"
        done
    fi
}

cmd_show() {
    tool_path="$1"
    if [[ -z "$tool_path" ]]; then
        echo "Usage: show <category/tool_name>" >&2
        return 1
    fi
    full_path="$TOOLS_DIR/$tool_path"
    if [[ ! -d "$full_path" ]]; then
        echo "Error: Tool '$tool_path' not found" >&2
        return 1
    fi
    
    # Show full README.md content
    if [[ -f "$full_path/README.md" ]]; then
        cat "$full_path/README.md"
    else
        echo "No README.md found for $tool_path"
    fi
    
    echo ""
    echo "---"
    echo "Files in tool directory:"
    ls -lh "$full_path" | tail -n +2
}

cmd_create() {
    category="$1"
    tool_name="$2"
    description="$3"
    lang="${4:-python}"  # Default python, optional bash
    
    if [[ -z "$category" || -z "$tool_name" ]]; then
        echo "Usage: create <category> <tool_name> \"<description>\" [python|bash]" >&2
        return 1
    fi
    
    # Create category if needed
    mkdir -p "$TOOLS_DIR/$category"
    
    # Create tool directory (fail if exists)
    tool_dir="$TOOLS_DIR/$category/$tool_name"
    if ! mkdir "$tool_dir" 2>/dev/null; then
        echo "Error: Tool '$category/$tool_name' already exists" >&2
        return 1
    fi
    
    if [[ "$lang" == "bash" ]]; then
        # Create main.sh template
    cat > "$tool_dir/main.sh" << 'SHEOF'
#!/bin/bash
# Tool main script
set -e

usage() {
    echo "Usage: $0 [arguments]"
    echo "  -h, --help    Show this help"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        *)
            break
            ;;
    esac
done

# TODO: Implement your tool logic here
echo "Tool executed successfully"
SHEOF
    chmod +x "$tool_dir/main.sh"
        main_file="main.sh"
        usage_cmd="bash /workspace/$category/$tool_name/main.sh [args]"
    else
        # Create main.py template (default)
        cat > "$tool_dir/main.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Tool: TOOL_NAME
Description: TOOL_DESC

Usage:
    python3 /workspace/CATEGORY/TOOL_NAME/main.py [arguments]
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="TOOL_DESC")
    parser.add_argument("input", nargs="?", help="Input argument")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    # TODO: Implement your tool logic here
    result = f"Processed: {args.input}"
    print(result)


if __name__ == "__main__":
    main()
PYEOF
        # Replace placeholders
        sed -i "s/TOOL_NAME/$tool_name/g" "$tool_dir/main.py"
        sed -i "s/TOOL_DESC/${description:-No description}/g" "$tool_dir/main.py"
        sed -i "s/CATEGORY/$category/g" "$tool_dir/main.py"
        chmod +x "$tool_dir/main.py"
        main_file="main.py"
        usage_cmd="python3 /workspace/$category/$tool_name/main.py [args]"
    fi
    
    # Create README.md with YAML frontmatter
    cat > "$tool_dir/README.md" << MDEOF
---
name: $tool_name
description: ${description:-No description provided}
category: $category
usage: $usage_cmd
---

# $tool_name

${description:-No description provided}

## Usage

\`\`\`bash
$usage_cmd
\`\`\`

## Arguments

- \`-h, --help\`: Show help message

## Examples

\`\`\`bash
$usage_cmd example_input
\`\`\`

## Implementation

Edit \`$main_file\` with your actual implementation.
MDEOF
    
    echo "Created tool: $category/$tool_name ($main_file)"
    echo ""
    echo "Tool path in Agent's Docker: /workspace/$category/$tool_name/$main_file"
    echo ""
    echo "Next: Edit the implementation:"
    echo "  cat > $tool_dir/$main_file << 'EOF'"
    echo "  # Your implementation here"
    echo "  EOF"
}

cmd_test() {
    tool_path="$1"
    if [[ -z "$tool_path" ]]; then
        echo "Usage: test <category/tool_name>" >&2
        return 1
    fi
    
    full_path="$TOOLS_DIR/$tool_path"
    if [[ ! -d "$full_path" ]]; then
        echo "Error: Tool '$tool_path' not found" >&2
        return 1
    fi
    
    echo "Testing tool: $tool_path"
    
    # Check README.md has proper frontmatter
    if [[ -f "$full_path/README.md" ]]; then
        if grep -q "^---$" "$full_path/README.md"; then
            desc=$(extract_frontmatter_field "$full_path/README.md" "description")
            if [[ -n "$desc" ]]; then
                echo "✓ README.md has valid YAML frontmatter"
                echo "  description: $desc"
            else
                echo "! README.md missing description in frontmatter"
            fi
        else
            echo "! README.md missing YAML frontmatter (---)"
        fi
    else
        echo "! README.md not found"
    fi
    
    # Check which implementation exists
    has_sh=false
    has_py=false
    
    if [[ -f "$full_path/main.sh" ]]; then
        has_sh=true
        echo "✓ main.sh found"
        
        # Check if it's still a template
        if grep -q "TODO: Implement" "$full_path/main.sh"; then
            echo "! WARNING: main.sh still contains TODO placeholder"
        fi
    fi
    
    if [[ -f "$full_path/main.py" ]]; then
        has_py=true
        echo "✓ main.py found"
    fi
    
    if [[ "$has_sh" == false && "$has_py" == false ]]; then
        echo "Error: No main.sh or main.py found"
        return 1
    fi
    
    # Test Shell script if exists
    if [[ "$has_sh" == true ]]; then
        if [[ ! -x "$full_path/main.sh" ]]; then
            echo "! main.sh not executable, fixing..."
            chmod +x "$full_path/main.sh"
        fi
        echo "✓ main.sh is executable"
        
        # Try to run with --help
        if bash "$full_path/main.sh" --help >/dev/null 2>&1; then
            echo "✓ main.sh --help works"
        else
            echo "! main.sh --help not working"
        fi
    fi
    
    # Test Python script if exists
    if [[ "$has_py" == true ]]; then
        if [[ ! -x "$full_path/main.py" ]]; then
            echo "! main.py not executable, fixing..."
            chmod +x "$full_path/main.py"
        fi
        echo "✓ main.py is executable"
        
        # Check Python syntax
        if python3 -m py_compile "$full_path/main.py" 2>/dev/null; then
            echo "✓ Python syntax valid"
        else
            echo "Error: Python syntax error"
            return 1
        fi
        
        # Try help
        if python3 "$full_path/main.py" --help >/dev/null 2>&1; then
            echo "✓ main.py --help works"
        else
            echo "! main.py --help not working"
        fi
    fi
    
    echo "✓ Basic checks passed"
}

# Main command dispatcher
case "$1" in
    list)
        cmd_list "$2"
        ;;
    show)
        cmd_show "$2"
        ;;
    create)
        cmd_create "$2" "$3" "$4" "$5"
        ;;
    test)
        cmd_test "$2"
        ;;
    *)
        echo "Tool Manager - Usage:"
        echo "  list [category]                      - List tools with descriptions"
        echo "  show <category/name>                 - Show full README.md"
        echo "  create <cat> <name> <desc> [py|bash] - Create new tool (default: python)"
        echo "  test <category/name>                 - Test a tool"
        echo ""
        echo "Categories: analysis, testing, debugging, utils"
        echo ""
        echo "Examples:"
        echo "  bash tools/tool_manager.sh create utils find_pattern \"Find regex pattern in files\""
        echo "  bash tools/tool_manager.sh create debugging trace_calls \"Trace function calls\" bash"
        echo "  bash tools/tool_manager.sh list"
        echo "  bash tools/tool_manager.sh test utils/find_pattern"
        exit 1
        ;;
esac
