"""Scaffold operations - loading, creation, and merging."""

import shutil
import logging
import yaml
from pathlib import Path

from .utils import safe_load_yaml, clean_yaml_content, console


def load_scaffold(scaffold_dir: Path) -> dict:
    """Load scaffold from directory."""
    scaffold_file = scaffold_dir / "scaffold.yaml"
    if not scaffold_file.exists():
        return {}
    
    scaffold = safe_load_yaml(scaffold_file)
    if not scaffold:
        try:
            content = clean_yaml_content(scaffold_file.read_text())
            scaffold = yaml.safe_load(content) or {}
        except Exception as e:
            logging.error(f"Failed to load scaffold from {scaffold_file}: {e}")
            return {}
    
    if "action_observation_template" in scaffold:
        template = scaffold["action_observation_template"]
        if "{{ observation }}" in template or "{{observation}}" in template:
            template = template.replace("{{ observation }}", "{{ output.output }}")
            template = template.replace("{{observation}}", "{{output.output}}")
            scaffold["action_observation_template"] = template
        if "{{ output }}" in template and "{{ output.output }}" not in template:
            template = template.replace("{{ output }}", "{{ output.output }}")
            scaffold["action_observation_template"] = template
    
    if "memory_template" in scaffold and scaffold["memory_template"]:
        memory_section = scaffold["memory_template"].strip()
        system_tpl = scaffold.get("system_template", "")
        if memory_section and memory_section not in system_tpl:
            scaffold["system_template"] = f"{system_tpl}\n\n{memory_section}"
    
    return scaffold


def init_agent_memory(scaffold_dir: Path):
    """Initialize agent_memory directory with search and write tools."""
    memory_dir = scaffold_dir / "agent_memory"
    memory_dir.mkdir(exist_ok=True)
    
    tools_src = Path(__file__).parent.parent.parent / "src" / "recreate_agent" / "tools"
    
    search_memory_src = tools_src / "search_memory.py"
    if search_memory_src.exists():
        shutil.copy(search_memory_src, memory_dir / "search_memory.py")
    
    write_memory_src = tools_src / "write_memory.py"
    if write_memory_src.exists():
        shutil.copy(write_memory_src, memory_dir / "write_memory.py")
    
    memories_file = memory_dir / "memories.yaml"
    if not memories_file.exists():
        memories_file.write_text(
            "# Agent Memory - Experiences and lessons learned\n"
            "# Agent can read with search_memory.py and write with write_memory.py\n\n"
            "memories: []\n"
        )


def create_submission_only_tool(target_path: Path):
    """Create a minimal read_trajectory.py that only supports submission command (for ablation)."""
    submission_only_script = '''#!/usr/bin/env python3
"""Read Trajectory - Submission Only (Ablation Mode)"""
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Read agent submission from trajectory")
    parser.add_argument("command", choices=["submission"], help="Only 'submission' is available")
    parser.add_argument("path", help="Path to trajectory file")
    args = parser.parse_args()
    
    try:
        data = json.loads(Path(args.path).read_text())
        submission = data.get("info", {}).get("submission", "")
        if submission:
            print("=" * 60)
            print("Submission Analysis")
            print("=" * 60)
            print(f"\\nPatch length: {len(submission)} chars")
            print(f"Files modified: {submission.count('diff --git')}")
            print("\\n--- Full Patch Content ---")
            print(submission[:5000] if len(submission) > 5000 else submission)
        else:
            print("No submission found in trajectory")
    except FileNotFoundError:
        print(f"Error: File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    target_path.write_text(submission_only_script)


def indent_yaml(text: str, indent: int = 2) -> str:
    """Indent multi-line text for YAML."""
    if not text:
        return "  (empty)"
    lines = text.strip().split('\n')
    return '\n'.join(' ' * indent + line for line in lines)


def create_symlink(workspace: Path):
    """Create current symlink to global_v000."""
    current = workspace / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to("global_v000")


def create_default_scaffold(workspace: Path, domain: str, domain_config: dict | None = None):
    """Create default scaffold from domain config (extracted from adapter)."""
    scaffold_dir = workspace / "global_v000"
    scaffold_dir.mkdir(parents=True, exist_ok=True)
    (scaffold_dir / "agent_tools").mkdir(exist_ok=True)
    init_agent_memory(scaffold_dir)
    
    # Get domain-specific scaffold template
    if domain_config and domain_config.get("scaffold_system"):
        system_tpl = domain_config.get("scaffold_system", "")
        instance_tpl = domain_config.get("scaffold_instance", "## Task\n{{task}}")
        format_err_tpl = domain_config.get("scaffold_format_error", "Invalid format. Use ONE code block in triple backticks.")
        step_limit = domain_config.get("step_limit", 50)
        cost_limit = domain_config.get("cost_limit", 2.0)
        
        # Get memory path
        memory_path = domain_config.get("memory_path", "/workspace/extras/agent_memory")
        if domain == "swe":
            memory_path = "/workspace/agent_memory"
        
        # Generate default memory_template
        memory_tpl = f'''## Memory System
You can read and write memories to learn from your experiences.

### Read (search past lessons):
python3 {memory_path}/search_memory.py "keyword"

### Write (save a lesson you learned):
python3 {memory_path}/write_memory.py --title "Short title" --content "What you learned" --tags "tag1,tag2"

**When to write a memory:**
- You discovered a useful pattern or workaround
- You solved a tricky error and want to remember the solution
- You found domain-specific knowledge that could help future tasks'''
        
        scaffold_content = f'''system_template: |
{indent_yaml(system_tpl)}

instance_template: |
{indent_yaml(instance_tpl)}

memory_template: |
{indent_yaml(memory_tpl)}

action_observation_template: |
  <output>
  {{{{output.output}}}}
  </output>

format_error_template: |
{indent_yaml(format_err_tpl)}

step_limit: {step_limit}
cost_limit: {cost_limit}
'''
        console.print(f"[dim]Created scaffold from {domain} adapter config[/dim]")
        (scaffold_dir / "scaffold.yaml").write_text(scaffold_content)
        create_symlink(workspace)
        return
    
    # Fallback: generate minimal default scaffold
    work_dir = "/workspace" if domain in ("data_science", "ds1000", "math", "appworld") else "/testbed"
    code_lang = "python" if domain == "appworld" else "bash"
    memory_path = "/workspace/agent_memory" if domain == "swe" else "/workspace/extras/agent_memory"
    
    default_scaffold = f'''system_template: |
  You are an AI assistant solving {domain} tasks.
  Working directory: {work_dir}
  Respond with ONE {code_lang} command in triple backticks.

instance_template: |
  ## Task
  {{{{task}}}}

memory_template: |
  ## Memory System
  You can read and write memories to learn from experiences.
  
  Read: python3 {memory_path}/search_memory.py "keyword"
  Write: python3 {memory_path}/write_memory.py --title "Title" --content "Content" --tags "tag"

action_observation_template: |
  <output>
  {{{{output.output}}}}
  </output>

format_error_template: |
  Invalid format. Use ONE command in triple backticks.

step_limit: 50
cost_limit: 2.0
'''
    
    (scaffold_dir / "scaffold.yaml").write_text(default_scaffold)
    create_symlink(workspace)


def merge_tools_and_memories(
    current_global: Path,
    modifications: list,
    new_global: Path,
    synthesis_working: Path,
):
    """Merge tools and memories from batch modifications."""
    # Start with synthesis working dir (Meta-Agent may have added tools)
    if (synthesis_working / "agent_tools").exists():
        shutil.copytree(synthesis_working / "agent_tools", new_global / "agent_tools")
    elif (current_global / "agent_tools").exists():
        shutil.copytree(current_global / "agent_tools", new_global / "agent_tools")
    else:
        (new_global / "agent_tools").mkdir()
    
    if (synthesis_working / "agent_memory").exists():
        shutil.copytree(synthesis_working / "agent_memory", new_global / "agent_memory")
    elif (current_global / "agent_memory").exists():
        shutil.copytree(current_global / "agent_memory", new_global / "agent_memory")
    else:
        init_agent_memory(new_global)
    
    # Merge new tools from modifications
    # Structure: agent_tools/<category>/<tool_name>/main.py
    for mod in modifications:
        tools_path = mod.get("tools_path")
        if tools_path and Path(tools_path).exists():
            for category in Path(tools_path).iterdir():
                if category.is_dir():
                    # Ensure category dir exists
                    category_dest = new_global / "agent_tools" / category.name
                    category_dest.mkdir(exist_ok=True)
                    # Copy each tool in the category
                    for tool in category.iterdir():
                        if tool.is_dir():
                            tool_dest = category_dest / tool.name
                            if not tool_dest.exists():
                                shutil.copytree(tool, tool_dest)
    
    # Merge memories from modifications
    memories_to_merge = []
    for mod in modifications:
        mem_path = mod.get("memory_path")
        if mem_path:
            mem_file = Path(mem_path) / "memories.yaml"
            data = safe_load_yaml(mem_file)
            memories_to_merge.extend(data.get("memories", []))
    
    # Merge into new global memories
    new_mem_file = new_global / "agent_memory" / "memories.yaml"
    existing = safe_load_yaml(new_mem_file)
    existing_mems = existing.get("memories", [])
    existing_ids = {m.get("id") for m in existing_mems if m.get("id")}
    
    for mem in memories_to_merge:
        if mem.get("id") and mem.get("id") not in existing_ids:
            existing_mems.append(mem)
            existing_ids.add(mem.get("id"))
    
    new_mem_file.write_text(yaml.dump({"memories": existing_mems}, allow_unicode=True))
