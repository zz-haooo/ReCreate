#!/usr/bin/env python3
"""
Search Memory - Agent tool to search historical experiences

This script is mounted in the Agent's Docker container at /workspace/memory/
Memories are maintained by ReCreate-Agent based on past experiences.

Usage:
    python3 /workspace/memory/search_memory.py "keyword"
    python3 /workspace/memory/search_memory.py --list
    python3 /workspace/memory/search_memory.py --tag error
    python3 /workspace/memory/search_memory.py --id mem_001
"""
import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # Fallback: simple YAML parser for basic cases
    yaml = None


def simple_yaml_parse(content: str) -> list:
    """Simple YAML parser for memories.yaml when PyYAML is not available."""
    memories = []
    current = {}
    in_content = False
    content_lines = []
    
    for line in content.split('\n'):
        stripped = line.strip()
        
        # Skip comments and empty lines at top level
        if not stripped or stripped.startswith('#'):
            continue
        
        # New memory entry
        if line.startswith('  - id:'):
            if current:
                if content_lines:
                    current['content'] = '\n'.join(content_lines)
                memories.append(current)
            current = {'id': stripped.replace('- id:', '').strip()}
            in_content = False
            content_lines = []
        elif line.startswith('    title:'):
            current['title'] = stripped.replace('title:', '').strip().strip('"')
            in_content = False
        elif line.startswith('    content:'):
            val = stripped.replace('content:', '').strip()
            if val == '|':
                in_content = True
                content_lines = []
            else:
                current['content'] = val.strip('"')
                in_content = False
        elif line.startswith('    tags:'):
            tags_str = stripped.replace('tags:', '').strip()
            # Parse [tag1, tag2] format
            if tags_str.startswith('[') and tags_str.endswith(']'):
                tags = [t.strip().strip("'\"") for t in tags_str[1:-1].split(',') if t.strip()]
                current['tags'] = tags
            else:
                current['tags'] = []
            in_content = False
        elif line.startswith('    created:'):
            current['created'] = stripped.replace('created:', '').strip().strip('"')
            in_content = False
        elif line.startswith('    source:'):
            current['source'] = stripped.replace('source:', '').strip().strip('"')
            in_content = False
        elif in_content and line.startswith('      '):
            content_lines.append(line[6:])  # Remove 6 spaces
    
    # Don't forget last entry
    if current:
        if content_lines:
            current['content'] = '\n'.join(content_lines)
        memories.append(current)
    
    return memories


def load_memories() -> list:
    """Load memories from YAML file."""
    # Try multiple locations based on different domain mount points
    for path in [
        Path("/workspace/agent_memory/memories.yaml"),  # SWE-bench
        Path("/workspace/extras/agent_memory/memories.yaml"),  # DA-Code, DS-1000
        Path("./memories.yaml"),  # Same directory as script
        Path("./agent_memory/memories.yaml"),
    ]:
        if path.exists():
            content = path.read_text()
            if yaml:
                try:
                    data = yaml.safe_load(content) or {}
                    return data.get("memories", [])
                except Exception:
                    pass
            # Fallback to simple parser
            return simple_yaml_parse(content)
    return []


def search_memories(query: str, memories: list) -> list:
    """Search memories by keyword."""
    query_lower = query.lower()
    results = []
    
    for mem in memories:
        score = 0
        # Title match (highest)
        if query_lower in mem.get("title", "").lower():
            score += 3
        # Content match
        if query_lower in mem.get("content", "").lower():
            score += 2
        # Tag match
        for tag in mem.get("tags", []):
            if query_lower in tag.lower():
                score += 1
        
        if score > 0:
            results.append((score, mem))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in results]


def format_memory(mem: dict, full: bool = False) -> str:
    """Format a memory for display."""
    lines = [f"[{mem.get('id', '?')}] {mem.get('title', 'Untitled')}"]
    tags = mem.get("tags", [])
    if tags:
        lines.append(f"  Tags: {', '.join(tags)}")
    
    content = mem.get("content", "")
    if full or len(content) <= 200:
        lines.append(f"  {content}")
    else:
        lines.append(f"  {content[:200]}...")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Search agent memories (historical experiences)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 /workspace/memory/search_memory.py "import error"
  python3 /workspace/memory/search_memory.py --tag django
  python3 /workspace/memory/search_memory.py --list
  python3 /workspace/memory/search_memory.py --id mem_001
""")
    
    parser.add_argument("query", nargs="?", help="Search keyword")
    parser.add_argument("--list", action="store_true", help="List all memories")
    parser.add_argument("--tag", help="Filter by tag")
    parser.add_argument("--id", dest="mem_id", help="Show specific memory by ID")
    
    args = parser.parse_args()
    
    memories = load_memories()
    
    if not memories:
        print("No memories available.")
        print("(Memories are added by ReCreate-Agent based on past experiences)")
        return
    
    # Show specific memory
    if args.mem_id:
        for mem in memories:
            if mem.get("id") == args.mem_id:
                print(format_memory(mem, full=True))
                return
        print(f"Memory '{args.mem_id}' not found.")
        return
    
    # Filter by tag
    if args.tag:
        filtered = [m for m in memories if args.tag.lower() in [t.lower() for t in m.get("tags", [])]]
        if not filtered:
            print(f"No memories with tag '{args.tag}'.")
            return
        print(f"Memories with tag '{args.tag}' ({len(filtered)}):\n")
        for mem in filtered:
            print(format_memory(mem))
            print()
        return
    
    # List all
    if args.list:
        print(f"All memories ({len(memories)}):\n")
        for mem in memories:
            print(format_memory(mem))
            print()
        return
    
    # Search
    if args.query:
        results = search_memories(args.query, memories)
        if not results:
            print(f"No memories found for '{args.query}'.")
            return
        print(f"Found {len(results)} memories for '{args.query}':\n")
        for mem in results[:5]:  # Top 5
            print(format_memory(mem))
            print()
        if len(results) > 5:
            print(f"... and {len(results) - 5} more. Use --list to see all.")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()

