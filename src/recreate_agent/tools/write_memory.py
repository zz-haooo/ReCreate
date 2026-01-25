#!/usr/bin/env python3
"""
Agent Memory Writer - Allows Agent to write experiences during execution.

Usage:
    python3 /workspace/extras/agent_memory/write_memory.py \
        --title "Lesson learned" \
        --content "Description of what was learned" \
        --tags "tag1,tag2"
        
    python3 /workspace/agent_memory/write_memory.py \
        --title "Error solution" \
        --content "How I fixed this error" \
        --tags "error,solution"
"""
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def simple_yaml_dump(data: dict) -> str:
    """Simple YAML serialization when PyYAML is not available."""
    lines = ["memories:"]
    for mem in data.get("memories", []):
        lines.append(f"  - id: {mem.get('id', '')}")
        lines.append(f"    title: \"{mem.get('title', '')}\"")
        content = mem.get('content', '').replace('"', '\\"')
        if '\n' in content:
            lines.append("    content: |")
            for line in content.split('\n'):
                lines.append(f"      {line}")
        else:
            lines.append(f"    content: \"{content}\"")
        tags = mem.get('tags', [])
        if tags:
            lines.append(f"    tags: [{', '.join(tags)}]")
        else:
            lines.append("    tags: []")
        lines.append(f"    source: {mem.get('source', 'agent_runtime')}")
        lines.append(f"    created: \"{mem.get('created', '')}\"")
    return '\n'.join(lines) + '\n'


def simple_yaml_load(content: str) -> dict:
    """Simple YAML parser for memories.yaml when PyYAML is not available."""
    memories = []
    current = {}
    in_content = False
    content_lines = []
    
    for line in content.split('\n'):
        stripped = line.strip()
        
        if not stripped or stripped.startswith('#'):
            continue
        
        if line.startswith('  - id:') or line.startswith('- id:'):
            if current:
                if content_lines:
                    current['content'] = '\n'.join(content_lines)
                memories.append(current)
            current = {'id': stripped.replace('- id:', '').strip()}
            in_content = False
            content_lines = []
        elif 'title:' in line and not in_content:
            val = stripped.split('title:', 1)[1].strip().strip('"\'')
            current['title'] = val
            in_content = False
        elif 'content:' in line and not in_content:
            val = stripped.split('content:', 1)[1].strip()
            if val == '|':
                in_content = True
                content_lines = []
            else:
                current['content'] = val.strip('"\'')
                in_content = False
        elif 'tags:' in line and not in_content:
            tags_str = stripped.split('tags:', 1)[1].strip()
            if tags_str.startswith('[') and tags_str.endswith(']'):
                tags = [t.strip().strip("'\"") for t in tags_str[1:-1].split(',') if t.strip()]
                current['tags'] = tags
            else:
                current['tags'] = []
            in_content = False
        elif 'created:' in line and not in_content:
            current['created'] = stripped.split('created:', 1)[1].strip().strip('"\'')
            in_content = False
        elif 'source:' in line and not in_content:
            current['source'] = stripped.split('source:', 1)[1].strip().strip('"\'')
            in_content = False
        elif in_content and (line.startswith('      ') or line.startswith('    ')):
            content_lines.append(line.strip())
        elif in_content and stripped and not any(k in stripped for k in ['id:', 'title:', 'tags:', 'created:', 'source:']):
            content_lines.append(stripped)
    
    if current:
        if content_lines:
            current['content'] = '\n'.join(content_lines)
        memories.append(current)
    
    return {"memories": memories}


def find_memory_file() -> Path | None:
    """Find the memories.yaml file in various possible locations."""
    for path in [
        Path("/workspace/extras/agent_memory/memories.yaml"),
        Path("/workspace/agent_memory/memories.yaml"),
        Path("./agent_memory/memories.yaml"),
        Path("./memories.yaml"),
    ]:
        if path.parent.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Write a memory for future tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 write_memory.py --title "API pagination" --content "Always check for next_page" --tags "api,pagination"
  python3 write_memory.py --title "Error fix" --content "Use try-except for file ops" --tags "error,solution"
""")
    parser.add_argument("--title", required=True, help="Short descriptive title (max 100 chars)")
    parser.add_argument("--content", required=True, help="What you learned (max 500 chars)")
    parser.add_argument("--tags", default="", help="Comma-separated tags for categorization")
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.title) > 100:
        print(f"Warning: Title truncated to 100 chars")
        args.title = args.title[:100]
    
    if len(args.content) > 500:
        print(f"Warning: Content truncated to 500 chars")
        args.content = args.content[:500]
    
    # Find memory file
    memory_file = find_memory_file()
    if memory_file is None:
        print("Error: Cannot find memory directory. Make sure agent_memory is mounted.")
        return 1
    
    # Load existing memories
    if memory_file.exists():
        content = memory_file.read_text()
        if yaml:
            try:
                data = yaml.safe_load(content) or {"memories": []}
            except Exception:
                data = simple_yaml_load(content)
        else:
            data = simple_yaml_load(content)
    else:
        data = {"memories": []}
    
    # Generate unique ID
    mem_id = f"agent_{hashlib.md5(f'{args.title}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
    
    # Check for duplicate titles
    existing_titles = {m.get('title', '').lower() for m in data.get('memories', [])}
    if args.title.lower() in existing_titles:
        print(f"Memory with similar title already exists. Skipping.")
        return 0
    
    # Create new memory
    new_memory = {
        "id": mem_id,
        "title": args.title,
        "content": args.content,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "source": "agent_runtime",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    data["memories"].append(new_memory)
    
    # Save
    if yaml:
        memory_file.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))
    else:
        memory_file.write_text(simple_yaml_dump(data))
    
    print(f"✓ Memory saved: [{mem_id}] {args.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

