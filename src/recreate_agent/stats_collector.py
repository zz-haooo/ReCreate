#!/usr/bin/env python3
"""
Statistics Collector - Collects and records statistics during evolution.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AgentRunStats:
    """Agent single run statistics."""
    instance_id: str
    scaffold_version: str
    
    # Basic results
    success: bool = False
    exit_status: str = ""
    retry_num: int = 0  # Retry number: 0=first run, 1=first retry
    
    # Execution stats
    n_steps: int = 0
    total_cost: float = 0.0
    duration_seconds: float = 0.0
    
    # Tool usage stats
    tool_calls: dict[str, int] = field(default_factory=dict)  # tool_name -> call count
    custom_tool_calls: int = 0  # /workspace/ tool call count
    
    # Command stats
    command_types: dict[str, int] = field(default_factory=dict)  # command_type -> count
    
    # Error stats
    format_errors: int = 0
    command_errors: int = 0
    timeouts: int = 0
    
    # Evaluation results
    eval_resolved: bool = False  # Whether passed test evaluation
    failed_tests: str = ""  # Failed test cases
    
    # Key events
    key_events: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        retry_info = f" (retry_{self.retry_num})" if self.retry_num > 0 else ""
        lines = [
            f"# Agent Run Stats: {self.instance_id}{retry_info}",
            f"",
            f"## Basic Info",
            f"- Scaffold version: {self.scaffold_version}",
            f"- Retry number: {self.retry_num}",
            f"- Result: {'✓ Success' if self.success else '✗ Failed'}",
            f"- Exit status: {self.exit_status}",
            f"",
            f"## Execution Stats",
            f"- Steps: {self.n_steps}",
            f"- Cost: ${self.total_cost:.4f}",
            f"- Duration: {self.duration_seconds:.1f}s",
            f"",
        ]
        
        if self.tool_calls:
            lines.append("## Tool Calls")
            for tool, count in sorted(self.tool_calls.items(), key=lambda x: -x[1]):
                lines.append(f"- {tool}: {count}")
            lines.append(f"- Custom tool calls: {self.custom_tool_calls}")
            lines.append("")
        
        if self.command_types:
            lines.append("## Command Type Distribution")
            for cmd_type, count in sorted(self.command_types.items(), key=lambda x: -x[1]):
                lines.append(f"- {cmd_type}: {count}")
            lines.append("")
        
        if self.format_errors or self.command_errors or self.timeouts:
            lines.append("## Error Stats")
            if self.format_errors:
                lines.append(f"- Format errors: {self.format_errors}")
            if self.command_errors:
                lines.append(f"- Command errors: {self.command_errors}")
            if self.timeouts:
                lines.append(f"- Timeouts: {self.timeouts}")
            lines.append("")
        
        # Evaluation results
        lines.append("## Evaluation Results")
        lines.append(f"- Tests passed: {'✓ Yes' if self.eval_resolved else '✗ No'}")
        if self.failed_tests:
            lines.append(f"- Failed tests: {self.failed_tests[:500]}")
        lines.append("")
        
        if self.key_events:
            lines.append("## Key Events")
            for event in self.key_events[:10]:
                lines.append(f"- {event}")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class EvolutionStats:
    """ReCreate-Agent evolution statistics."""
    version: int
    source_instance: str
    previous_version: int
    
    # Execution stats
    n_steps: int = 0
    total_cost: float = 0.0
    exit_status: str = ""
    
    # Change stats
    scaffold_changed: bool = False
    scaffold_diff_lines: int = 0
    tools_added: list[str] = field(default_factory=list)
    tools_modified: list[str] = field(default_factory=list)
    memories_added: int = 0
    
    # Detailed action stats
    actions: dict[str, int] = field(default_factory=dict)  # action_type -> count
    component_edits: dict[str, int] = field(default_factory=dict)  # component -> edit count
    
    # Problem analysis
    identified_issues: list[str] = field(default_factory=list)
    intervention_type: str = ""  # small_fix, restructure, significant_change
    
    # Reasoning summary
    reasoning_summary: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"# Evolution Stats: v{self.previous_version:03d} → v{self.version:03d}",
            f"",
            f"## Source",
            f"- Analyzed instance: {self.source_instance}",
            f"- Exit status: {self.exit_status}",
            f"",
            f"## Execution",
            f"- Steps: {self.n_steps}",
            f"- Cost: ${self.total_cost:.4f}",
            f"",
            f"## ReCreate-Agent Action Stats",
        ]
        
        # Detailed action stats
        if self.actions:
            for action, count in sorted(self.actions.items()):
                lines.append(f"- {action}: {count}")
        else:
            lines.append("- (no actions recorded)")
        
        lines.append("")
        lines.append("## Component Edit Stats")
        
        if self.component_edits:
            for comp, count in sorted(self.component_edits.items()):
                lines.append(f"- {comp}: {count}")
        else:
            lines.append("- (no edits)")
        
        lines.append("")
        lines.append("## Change Results")
        lines.append(f"- Scaffold: {'modified' if self.scaffold_changed else 'unchanged'} ({self.scaffold_diff_lines} line diff)")
        
        if self.tools_added:
            lines.append(f"- Tools added: {', '.join(self.tools_added)}")
        if self.tools_modified:
            lines.append(f"- Tools modified: {', '.join(self.tools_modified)}")
        if self.memories_added > 0:
            lines.append(f"- Memories added: {self.memories_added}")
        
        lines.append("")
        
        if self.identified_issues:
            lines.append("## Identified Issues")
            for issue in self.identified_issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        if self.intervention_type:
            lines.append(f"## Intervention Type: {self.intervention_type}")
            lines.append("")
        
        if self.reasoning_summary:
            lines.append("## Reasoning Summary")
            lines.append(self.reasoning_summary)
            lines.append("")
        
        return "\n".join(lines)


def analyze_trajectory(traj_path: Path) -> AgentRunStats:
    """Analyze agent trajectory file and extract statistics."""
    traj = json.loads(traj_path.read_text())
    
    instance_id = traj.get("instance_id", traj_path.stem)
    messages = traj.get("messages", [])
    info = traj.get("info", {})
    model_stats = info.get("model_stats", {})
    
    stats = AgentRunStats(
        instance_id=instance_id,
        scaffold_version=info.get("scaffold_version", "unknown"),
        success=info.get("exit_status") in ("Resolved", "Submitted"),
        exit_status=info.get("exit_status", "Unknown"),
        n_steps=model_stats.get("api_calls", len([m for m in messages if m.get("role") == "assistant"])),
        total_cost=model_stats.get("instance_cost", 0.0),
    )
    
    # Analyze messages
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "assistant":
            # Analyze command types
            if "```bash" in content:
                cmd_match = re.search(r'```bash\n(.+?)\n```', content, re.DOTALL)
                if cmd_match:
                    cmd = cmd_match.group(1).strip().split()[0] if cmd_match.group(1).strip() else ""
                    if cmd:
                        # Classify command
                        if cmd in ["cat", "head", "tail", "less", "nl"]:
                            cmd_type = "view_file"
                        elif cmd in ["ls", "find", "tree"]:
                            cmd_type = "explore"
                        elif cmd in ["sed", "awk"]:
                            cmd_type = "edit"
                        elif cmd in ["grep", "rg"]:
                            cmd_type = "search"
                        elif cmd in ["python", "python3", "pytest"]:
                            cmd_type = "run_python"
                        elif cmd in ["git"]:
                            cmd_type = "git"
                        elif cmd in ["cd"]:
                            cmd_type = "navigate"
                        elif cmd in ["echo"]:
                            cmd_type = "echo"
                        elif cmd in ["bash"]:
                            cmd_type = "run_script"
                        else:
                            cmd_type = cmd
                        
                        stats.command_types[cmd_type] = stats.command_types.get(cmd_type, 0) + 1
                    
                    # Check tool calls
                    if "/workspace/" in content:
                        stats.custom_tool_calls += 1
                        tool_match = re.search(r'/workspace/([^/\s]+)', content)
                        if tool_match:
                            tool_name = tool_match.group(1)
                            stats.tool_calls[tool_name] = stats.tool_calls.get(tool_name, 0) + 1
        
        elif role == "user":
            # Analyze errors
            if "EXACTLY ONE" in content or "format" in content.lower():
                stats.format_errors += 1
            elif "timed out" in content.lower():
                stats.timeouts += 1
            elif "error" in content.lower() or "Error" in content:
                stats.command_errors += 1
            
            # Extract key events
            if "returncode" in content:
                if ">0<" in content:
                    stats.key_events.append("Step: command success")
                else:
                    error_preview = content[:100].replace("\n", " ")
                    stats.key_events.append(f"Step: command failed - {error_preview}...")
    
    return stats


def analyze_evolution(
    workspace: Path,
    new_version: int,
    prev_version: int,
    source_instance: str,
    meta_traj_path: Path | None = None,
) -> EvolutionStats:
    """Analyze evolution changes and generate statistics."""
    
    stats = EvolutionStats(
        version=new_version,
        source_instance=source_instance,
        previous_version=prev_version,
    )
    
    # Analyze ReCreate-Agent trajectory
    if meta_traj_path and meta_traj_path.exists():
        meta_traj = json.loads(meta_traj_path.read_text())
        stats.n_steps = meta_traj.get("n_steps", 0)
        stats.total_cost = meta_traj.get("cost", 0.0)
        stats.exit_status = meta_traj.get("exit_status", "Unknown")
        
        # Initialize action stats
        stats.actions = {
            "scaffold_edit": 0,
            "tool_create": 0,
            "memory_add": 0,
            "read_trajectory": 0,
            "inspect_docker": 0,
            "web_search": 0,
            "view_file": 0,
        }
        stats.component_edits = {
            "system_template": 0,
            "instance_template": 0,
            "action_observation_template": 0,
            "format_error_template": 0,
        }
        
        # Count actions
        for msg in meta_traj.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # Count scaffold edits
                if "scaffold_editor.py str_replace" in content or "scaffold_editor.py insert" in content:
                    stats.actions["scaffold_edit"] += 1
                    if "system_template:" in content or ("You are a" in content and "THOUGHT" in content):
                        stats.component_edits["system_template"] += 1
                    elif any(kw in content for kw in ["## Workflow", "## Task", "## Instructions", "## Error Recovery", "## File"]):
                        stats.component_edits["instance_template"] += 1
                    elif "action_observation_template" in content or "Command output:" in content:
                        stats.component_edits["action_observation_template"] += 1
                    elif "format_error_template" in content:
                        stats.component_edits["format_error_template"] += 1
                    else:
                        stats.component_edits["instance_template"] += 1
                
                if "tool_manager.sh create" in content:
                    stats.actions["tool_create"] += 1
                
                if "memory_manager.py add" in content:
                    stats.actions["memory_add"] += 1
                    stats.memories_added += 1
                
                if "read_trajectory.py" in content:
                    stats.actions["read_trajectory"] += 1
                
                if "inspect_in_docker.py" in content:
                    stats.actions["inspect_docker"] += 1
                
                if "web_search.py" in content:
                    stats.actions["web_search"] += 1
                
                if "cat current/scaffold.yaml" in content or "scaffold_editor.py view" in content:
                    stats.actions["view_file"] += 1
                
                # Extract reasoning summary
                if "THOUGHT:" in content and not stats.reasoning_summary:
                    thought = content.split("THOUGHT:")[-1].split("```")[0].strip()
                    stats.reasoning_summary = thought[:500]
    
    # Analyze scaffold changes
    old_scaffold = workspace / f"scaffold_v{prev_version:03d}" / "scaffold.yaml"
    new_scaffold = workspace / f"scaffold_v{new_version:03d}" / "scaffold.yaml"
    
    if old_scaffold.exists() and new_scaffold.exists():
        old_content = old_scaffold.read_text()
        new_content = new_scaffold.read_text()
        
        if old_content != new_content:
            stats.scaffold_changed = True
            import difflib
            diff = list(difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
            ))
            stats.scaffold_diff_lines = len([l for l in diff if l.startswith('+') or l.startswith('-')])
    
    # Analyze tool changes
    old_tools = workspace / f"scaffold_v{prev_version:03d}" / "agent_tools"
    new_tools = workspace / f"scaffold_v{new_version:03d}" / "agent_tools"
    
    old_tool_set = set()
    new_tool_set = set()
    
    if old_tools.exists():
        for cat in old_tools.iterdir():
            if cat.is_dir():
                for tool in cat.iterdir():
                    if tool.is_dir():
                        old_tool_set.add(f"{cat.name}/{tool.name}")
    
    if new_tools.exists():
        for cat in new_tools.iterdir():
            if cat.is_dir():
                for tool in cat.iterdir():
                    if tool.is_dir():
                        new_tool_set.add(f"{cat.name}/{tool.name}")
    
    stats.tools_added = list(new_tool_set - old_tool_set)
    
    return stats


def save_agent_stats(stats: AgentRunStats, output_dir: Path):
    """Save agent run statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON format
    (output_dir / "stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False)
    )
    
    # Markdown summary
    (output_dir / "stats.md").write_text(stats.to_summary())


def save_evolution_stats(stats: EvolutionStats, version_dir: Path):
    """Save evolution statistics."""
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON format
    (version_dir / "evolution_stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False)
    )
    
    # Markdown summary
    (version_dir / "evolution_stats.md").write_text(stats.to_summary())


def generate_scaffold_diff(workspace: Path, old_version: int, new_version: int) -> str:
    """Generate scaffold diff report."""
    old_file = workspace / f"scaffold_v{old_version:03d}" / "scaffold.yaml"
    new_file = workspace / f"scaffold_v{new_version:03d}" / "scaffold.yaml"
    
    if not old_file.exists() or not new_file.exists():
        return "Cannot generate diff report: file not found"
    
    import difflib
    
    old_lines = old_file.read_text().splitlines()
    new_lines = new_file.read_text().splitlines()
    
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"scaffold_v{old_version:03d}",
        tofile=f"scaffold_v{new_version:03d}",
        lineterm=""
    )
    
    return "\n".join(diff)
