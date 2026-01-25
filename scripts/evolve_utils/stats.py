"""Statistics collection and saving utilities."""

import json
from pathlib import Path

from .trajectory import analyze_recreate_agent_traj


def collect_batch_stats(
    batch_dir: Path,
    batch_results: dict,
    runs_recreate_dir: Path,
    batch_idx: int,
    new_version: int,
    prev_version: int,
    EvolutionStats,
):
    """
    Collect and aggregate statistics from all ReCreate-Agents in a batch.
    
    Args:
        batch_dir: Batch directory (workspace/batch_XXX)
        batch_results: BatchResult dict
        runs_recreate_dir: ReCreate-Agent trajectory save directory
        batch_idx: Batch index
        new_version: New version number
        prev_version: Previous version number
        EvolutionStats: EvolutionStats class (passed to avoid circular import)
    
    Returns:
        EvolutionStats object with aggregated statistics
    """
    # Aggregate statistics
    total_stats = {
        "read_scaffold": 0,
        "read_submission": 0,
        "read_trajectory": 0,
        "read_eval_result": 0,
        "inspect_docker": 0,
        "system_prompt_edit": 0,
        "instance_prompt_edit": 0,
        "memory_template_edit": 0,
        "tool_create": 0,
        "memory_create": 0,
        "web_search": 0,
    }
    total_steps = 0
    total_cost = 0.0
    instance_ids = []
    tools_added = []
    memories_added = 0
    
    for instance_id, result in batch_results.items():
        instance_ids.append(instance_id)
        
        # Find ReCreate-Agent trajectory files
        # Format: parallel_{instance_id}_{timestamp}/.../*.traj.json
        traj_found = False
        for traj_dir in runs_recreate_dir.iterdir():
            if traj_dir.is_dir() and f"parallel_{instance_id}" in traj_dir.name:
                # Use rglob for recursive search
                for traj_file in traj_dir.rglob("*.traj.json"):
                    stats = analyze_recreate_agent_traj(traj_file)
                    for key in total_stats:
                        total_stats[key] += stats.get(key, 0)
                    total_steps += stats.get("n_steps", 0)
                    total_cost += stats.get("cost", 0.0)
                    traj_found = True
                    break
                if traj_found:
                    break
        
        # Check tool and memory changes
        instance_dir = batch_dir / instance_id
        if result.has_new_tools:
            tools_path = instance_dir / "agent_tools"
            if tools_path.exists():
                for cat in tools_path.iterdir():
                    if cat.is_dir():
                        for tool in cat.iterdir():
                            if tool.is_dir():
                                tools_added.append(f"{cat.name}/{tool.name}")
        
        if result.has_new_memories:
            memories_added += 1
    
    # Create EvolutionStats object
    evolution_stats = EvolutionStats(
        version=new_version,
        source_instance=", ".join(instance_ids[:4]) + ("..." if len(instance_ids) > 4 else ""),
        previous_version=prev_version,
        n_steps=total_steps,
        total_cost=total_cost,
        exit_status=f"Batch {batch_idx}",
        scaffold_changed=any(r.scaffold_changed for r in batch_results.values()),
        tools_added=list(set(tools_added)),
        memories_added=memories_added,
        actions=total_stats,
    )
    
    return evolution_stats


def load_cumulative_stats(prev_version_dir: Path) -> dict:
    """Load cumulative statistics from previous version."""
    cumulative = {
        "actions": {
            "read_scaffold": 0,
            "read_submission": 0,
            "read_trajectory": 0,
            "read_eval_result": 0,
            "inspect_docker": 0,
            "system_prompt_edit": 0,
            "instance_prompt_edit": 0,
            "memory_template_edit": 0,
            "tool_create": 0,
            "memory_create": 0,
            "web_search": 0,
        },
        "total_steps": 0,
        "total_cost": 0.0,
        "total_memories": 0,
        "all_tools": [],
    }
    
    stats_file = prev_version_dir / "evolution_stats.json"
    if stats_file.exists():
        try:
            data = json.loads(stats_file.read_text())
            # Read cumulative values if available
            if "cumulative" in data:
                cumulative = data["cumulative"]
            else:
                # Backward compatible: use current batch values as starting point
                cumulative["actions"] = data.get("actions", cumulative["actions"])
                cumulative["total_steps"] = data.get("n_steps", 0)
                cumulative["total_cost"] = data.get("total_cost", 0.0)
                cumulative["total_memories"] = data.get("memories_added", 0)
                cumulative["all_tools"] = data.get("tools_added", [])
        except Exception:
            pass
    
    return cumulative


def save_batch_evolution_stats(stats, version_dir: Path, prev_version_dir: Path = None):
    """Save batch evolution statistics to global_v00x directory with cumulative stats."""
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Load previous cumulative statistics
    if prev_version_dir and prev_version_dir.exists():
        cumulative = load_cumulative_stats(prev_version_dir)
    else:
        cumulative = {
            "actions": {k: 0 for k in stats.actions.keys()},
            "total_steps": 0,
            "total_cost": 0.0,
            "total_memories": 0,
            "all_tools": [],
        }
    
    # Add current batch statistics
    for action, count in stats.actions.items():
        cumulative["actions"][action] = cumulative["actions"].get(action, 0) + count
    cumulative["total_steps"] += stats.n_steps
    cumulative["total_cost"] += stats.total_cost
    cumulative["total_memories"] += stats.memories_added
    cumulative["all_tools"] = list(set(cumulative["all_tools"] + stats.tools_added))
    
    # Generate Markdown summary
    lines = [
        f"# Evolution Stats: v{stats.previous_version:03d} → v{stats.version:03d}",
        f"",
        f"## Current Batch",
        f"- Instance: {stats.source_instance}",
        f"- Status: {stats.exit_status}",
        f"- Steps: {stats.n_steps}",
        f"- Cost: ${stats.total_cost:.4f}",
        f"",
        f"## Cumulative (All Batches)",
        f"- Total steps: {cumulative['total_steps']}",
        f"- Total cost: ${cumulative['total_cost']:.4f}",
        f"- Total memories: {cumulative['total_memories']}",
        f"- Total tools: {len(cumulative['all_tools'])}",
        f"",
        f"## ReCreate-Agent Operations (Cumulative)",
        f"",
        f"### Read Operations",
    ]
    
    # Read operations
    read_actions = [
        ("read_scaffold", "Read scaffold"),
        ("read_submission", "Read submission"),
        ("read_trajectory", "Read trajectory"),
        ("read_eval_result", "Read eval result"),
        ("inspect_docker", "Inspect Docker"),
    ]
    for action, desc in read_actions:
        count = cumulative["actions"].get(action, 0)
        lines.append(f"- {action}: {count}")
    
    lines.append("")
    lines.append("### Modify Operations")
    
    # Modify operations
    modify_actions = [
        ("system_prompt_edit", "Edit system prompt"),
        ("instance_prompt_edit", "Edit instance prompt"),
        ("memory_template_edit", "Edit memory template"),
        ("tool_create", "Create tool"),
        ("memory_create", "Create memory"),
        ("web_search", "Web search"),
    ]
    for action, desc in modify_actions:
        count = cumulative["actions"].get(action, 0)
        lines.append(f"- {action}: {count}")
    
    lines.append("")
    lines.append("## Batch Changes")
    lines.append(f"- Scaffold: {'modified' if stats.scaffold_changed else 'unchanged'}")
    
    if stats.tools_added:
        lines.append(f"- Tools added: {', '.join(stats.tools_added)}")
    if stats.memories_added > 0:
        lines.append(f"- Memories added: {stats.memories_added}")
    
    lines.append("")
    
    # Save files
    (version_dir / "evolution_stats.md").write_text("\n".join(lines))
    
    # JSON format with cumulative stats
    stats_dict = stats.to_dict()
    stats_dict["cumulative"] = cumulative
    (version_dir / "evolution_stats.json").write_text(
        json.dumps(stats_dict, indent=2, ensure_ascii=False)
    )
