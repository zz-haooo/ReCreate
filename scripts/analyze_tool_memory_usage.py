#!/usr/bin/env python3
"""
Analyze tool and memory usage in test trajectories.

Usage:
    python analyze_tool_memory_usage.py --test-dir /path/to/test_results/evolved
    python analyze_tool_memory_usage.py --test-dir /path/to/test_results/evolved --output stats.json
"""

import json
import re
from pathlib import Path

import typer

app = typer.Typer()


def count_tool_memory_calls(trajectory: list[dict]) -> dict:
    """Count tool and memory calls in a single trajectory."""
    tool_calls = {}
    memory_read_calls = 0   # search_memory.py
    memory_write_calls = 0  # write_memory.py
    
    # Tool call patterns
    tool_patterns = [
        r"python3?\s+/workspace/(?:extras/)?agent_tools/(\w+)/(\w+)/main\.py",
        r"bash\s+/workspace/(?:extras/)?agent_tools/(\w+)/(\w+)/main\.sh",
    ]
    
    # Memory read patterns (search_memory.py)
    memory_read_patterns = [
        r"python3?\s+/workspace/(?:extras/)?agent_memory/search_memory\.py",
        r"search_memory\.py\s+",
    ]
    
    # Memory write patterns (write_memory.py)
    memory_write_patterns = [
        r"python3?\s+/workspace/(?:extras/)?agent_memory/write_memory\.py",
        r"write_memory\.py\s+",
    ]
    
    for msg in trajectory:
        # Only count assistant messages (actual calls, not system instructions)
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        
        # Count tool calls
        for pattern in tool_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    tool_name = f"{match[0]}/{match[1]}"
                else:
                    tool_name = str(match)
                tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
        
        # Count memory reads (search_memory.py)
        for pattern in memory_read_patterns:
            matches = re.findall(pattern, content)
            memory_read_calls += len(matches)
        
        # Count memory writes (write_memory.py)
        for pattern in memory_write_patterns:
            matches = re.findall(pattern, content)
            memory_write_calls += len(matches)
    
    return {
        "tool_calls": tool_calls,
        "total_tool_calls": sum(tool_calls.values()),
        "memory_read_calls": memory_read_calls,
        "memory_write_calls": memory_write_calls,
        "total_memory_calls": memory_read_calls + memory_write_calls,
    }


def analyze_test_results(test_dir: Path) -> dict:
    """Analyze all trajectories in test results directory."""
    stats = {
        "total_instances": 0,
        "instances_with_tool_calls": 0,
        "instances_with_memory_calls": 0,
        "total_tool_calls": 0,
        "total_memory_read_calls": 0,
        "total_memory_write_calls": 0,
        "total_memory_calls": 0,
        "tool_breakdown": {},
        "per_instance": [],
    }
    
    # Find all trajectory files (support multiple formats)
    traj_files = list(test_dir.glob("**/trajectory.json"))
    traj_files.extend(test_dir.glob("**/*.traj.json"))
    
    if not traj_files:
        print(f"Warning: No trajectory files found (trajectory.json)")
        print(f"Search directory: {test_dir}")
        return stats
    
    for traj_file in traj_files:
        instance_id = traj_file.parent.name
        
        try:
            trajectory = json.loads(traj_file.read_text())
            if isinstance(trajectory, dict):
                trajectory = trajectory.get("messages", trajectory.get("trajectory", []))
        except Exception as e:
            print(f"  Warning: Cannot parse {traj_file}: {e}")
            continue
        
        result = count_tool_memory_calls(trajectory)
        
        stats["total_instances"] += 1
        
        if result["total_tool_calls"] > 0:
            stats["instances_with_tool_calls"] += 1
            stats["total_tool_calls"] += result["total_tool_calls"]
            
            # Merge tool stats
            for tool, count in result["tool_calls"].items():
                stats["tool_breakdown"][tool] = stats["tool_breakdown"].get(tool, 0) + count
        
        if result["total_memory_calls"] > 0:
            stats["instances_with_memory_calls"] += 1
            stats["total_memory_read_calls"] += result["memory_read_calls"]
            stats["total_memory_write_calls"] += result["memory_write_calls"]
            stats["total_memory_calls"] += result["total_memory_calls"]
        
        stats["per_instance"].append({
            "instance_id": instance_id,
            **result,
        })
    
    return stats


@app.command()
def main(
    test_dir: Path = typer.Option(..., "--test-dir", "-d", help="Test results directory"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Analyze tool and memory usage in test trajectories."""
    
    if not test_dir.exists():
        print(f"Error: Directory not found: {test_dir}")
        raise typer.Exit(1)
    
    print(f"Analyzing: {test_dir}")
    stats = analyze_test_results(test_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Agent Behavior Statistics")
    print("=" * 50)
    print(f"Total instances: {stats['total_instances']}")
    print(f"Instances with tool calls: {stats['instances_with_tool_calls']}")
    print(f"Instances with memory calls: {stats['instances_with_memory_calls']}")
    print(f"\n--- Tool Calls ---")
    print(f"Total tool calls: {stats['total_tool_calls']}")
    print(f"\n--- Memory Operations ---")
    print(f"Memory reads (search_memory.py): {stats['total_memory_read_calls']}")
    print(f"Memory writes (write_memory.py): {stats['total_memory_write_calls']}")
    print(f"Total memory operations: {stats['total_memory_calls']}")
    
    if stats["tool_breakdown"]:
        print("\n--- Tool Call Breakdown ---")
        for tool, count in sorted(stats["tool_breakdown"].items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
    
    # Save results
    if output is None:
        output = test_dir / "tool_memory_stats.json"
    
    output.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nStats saved: {output}")


if __name__ == "__main__":
    app()
