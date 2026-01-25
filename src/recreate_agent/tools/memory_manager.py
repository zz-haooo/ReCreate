#!/usr/bin/env python3
"""
Memory Manager - Manage agent memories (historical experiences and lessons)

Usage:
    python3 tools/memory_manager.py add --title "Title" --content "Content" [--tags "tag1,tag2"] [--source "instance_id"]
    python3 tools/memory_manager.py list [--tag TAG]
    python3 tools/memory_manager.py search "keyword"
    python3 tools/memory_manager.py remove --id mem_XXX

Memories are stored in current/agent_memory/memories.yaml
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml


def get_memory_file() -> Path:
    """Find the memories file."""
    # Try ReCreate-Agent workspace first
    for path in [
        Path("./current/agent_memory/memories.yaml"),
        Path("./working/agent_memory/memories.yaml"),
        Path("./agent_memory/memories.yaml"),
    ]:
        if path.parent.exists():
            return path
    # Default to current/agent_memory
    return Path("./current/agent_memory/memories.yaml")


def load_memories(path: Path) -> list:
    """Load memories from YAML file."""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data.get("memories", [])
    except Exception:
        return []


def save_memories(path: Path, memories: list) -> None:
    """Save memories to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "# Agent Memory": "Historical experiences and lessons learned",
        "# Maintained by": "ReCreate-Agent",
        "# Used by": "Agent via search_memory.py",
        "memories": memories,
    }
    # Custom YAML dump with comments
    lines = [
        "# Agent Memory - Historical experiences and lessons learned",
        "# Maintained by ReCreate-Agent. Agent can search and read.",
        "",
        "memories:",
    ]
    for mem in memories:
        lines.append(f"  - id: {mem['id']}")
        lines.append(f"    title: \"{mem['title']}\"")
        # Handle multi-line content
        content_str = mem['content'].replace('"', '\\"')
        if '\n' in content_str:
            lines.append("    content: |")
            for line in mem['content'].split('\n'):
                lines.append(f"      {line}")
        else:
            lines.append(f"    content: \"{content_str}\"")
        lines.append(f"    tags: {mem.get('tags', [])}")
        lines.append(f"    created: \"{mem.get('created', '')}\"")
        if mem.get('source'):
            lines.append(f"    source: \"{mem['source']}\"")
        lines.append("")
    
    path.write_text("\n".join(lines))


def generate_id(memories: list) -> str:
    """Generate next memory ID."""
    if not memories:
        return "mem_001"
    max_num = 0
    for mem in memories:
        try:
            num = int(mem["id"].replace("mem_", ""))
            max_num = max(max_num, num)
        except (ValueError, KeyError):
            pass
    return f"mem_{max_num + 1:03d}"


def cmd_add(args):
    """Add a new memory."""
    path = get_memory_file()
    memories = load_memories(path)
    
    # Parse tags
    tags = []
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    
    # Validate content length
    if len(args.content) > 500:
        print(f"Warning: Content is {len(args.content)} chars. Consider keeping it under 500 for efficiency.")
    
    new_memory = {
        "id": generate_id(memories),
        "title": args.title.strip(),
        "content": args.content.strip(),
        "tags": tags,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "source": args.source or "",
    }
    
    memories.append(new_memory)
    save_memories(path, memories)
    
    print(f"✓ Added memory: {new_memory['id']}")
    print(f"  Title: {new_memory['title']}")
    print(f"  Tags: {', '.join(tags) if tags else '(none)'}")
    print(f"  File: {path}")


def cmd_list(args):
    """List all memories."""
    path = get_memory_file()
    memories = load_memories(path)
    
    if not memories:
        print("No memories found.")
        return
    
    # Filter by tag if specified
    if args.tag:
        memories = [m for m in memories if args.tag.lower() in [t.lower() for t in m.get("tags", [])]]
    
    print(f"Memories ({len(memories)}):\n")
    for mem in memories:
        tags_str = ", ".join(mem.get("tags", []))
        print(f"  [{mem['id']}] {mem['title']}")
        print(f"       Tags: {tags_str or '(none)'} | Created: {mem.get('created', 'unknown')}")
        # Show first 100 chars of content
        content_preview = mem['content'][:100].replace('\n', ' ')
        if len(mem['content']) > 100:
            content_preview += "..."
        print(f"       {content_preview}")
        print()


def cmd_search(args):
    """Search memories by keyword."""
    path = get_memory_file()
    memories = load_memories(path)
    
    if not memories:
        print("No memories to search.")
        return
    
    query = args.query.lower()
    results = []
    
    for mem in memories:
        score = 0
        # Check title (highest weight)
        if query in mem["title"].lower():
            score += 3
        # Check content
        if query in mem["content"].lower():
            score += 2
        # Check tags
        for tag in mem.get("tags", []):
            if query in tag.lower():
                score += 1
        
        if score > 0:
            results.append((score, mem))
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    if not results:
        print(f"No memories found for: '{args.query}'")
        return
    
    print(f"Found {len(results)} memories for '{args.query}':\n")
    for score, mem in results[:10]:  # Top 10
        print(f"  [{mem['id']}] {mem['title']}")
        print(f"       {mem['content'][:150].replace(chr(10), ' ')}{'...' if len(mem['content']) > 150 else ''}")
        print()


def cmd_remove(args):
    """Remove a memory by ID."""
    path = get_memory_file()
    memories = load_memories(path)
    
    original_len = len(memories)
    memories = [m for m in memories if m["id"] != args.id]
    
    if len(memories) == original_len:
        print(f"Memory '{args.id}' not found.")
        return
    
    save_memories(path, memories)
    print(f"✓ Removed memory: {args.id}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage agent memories (historical experiences)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a memory about error handling
  python3 tools/memory_manager.py add \\
    --title "Circular import fix" \\
    --content "Move import inside function or use lazy import" \\
    --tags "python,import,error" \\
    --source "django__django-12345"

  # List all memories with 'error' tag
  python3 tools/memory_manager.py list --tag error

  # Search for django-related memories
  python3 tools/memory_manager.py search "django migration"

  # Remove a memory
  python3 tools/memory_manager.py remove --id mem_001
""")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new memory")
    add_parser.add_argument("--title", required=True, help="Short title (max 100 chars)")
    add_parser.add_argument("--content", required=True, help="Memory content (max 500 chars recommended)")
    add_parser.add_argument("--tags", help="Comma-separated tags (e.g., 'python,error,import')")
    add_parser.add_argument("--source", help="Source instance ID (e.g., 'django__django-12345')")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all memories")
    list_parser.add_argument("--tag", help="Filter by tag")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search keyword")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a memory")
    remove_parser.add_argument("--id", required=True, help="Memory ID to remove")
    
    args = parser.parse_args()
    
    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "remove":
        cmd_remove(args)


if __name__ == "__main__":
    main()

