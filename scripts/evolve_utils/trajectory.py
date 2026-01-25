"""Trajectory analysis and domain config utilities."""

import json
import re
from pathlib import Path


def generate_trajectory_index(traj_path: Path, results_dir: Path):
    """Generate index for trajectory file (for ReCreate-Agent quick lookup)."""
    try:
        traj_data = json.loads(traj_path.read_text())
        messages = traj_data.get("messages", [])
        info = traj_data.get("info", {})
        
        index = {
            "files": {"viewed": {}, "edited": {}, "searched": {}},
            "errors": [],
            "tests": {"ran": [], "passed": [], "failed": []},
            "expected": {"fail_to_pass": [], "pass_to_pass": []},
            "meta": {
                "total_steps": 0,
                "exit_status": info.get("exit_status", "unknown"),
            },
        }
        
        step = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "assistant":
                step += 1
                for match in re.finditer(r"sed\s+-n\s+['\"][^'\"]+['\"]\s+([^\s`]+)", content):
                    path = match.group(1).strip("'\"` ")
                    if "/" in path:
                        index["files"]["viewed"].setdefault(path, []).append(step)
                for match in re.finditer(r"cat\s+(/[^\s|>]+)", content):
                    path = match.group(1).strip()
                    if path:
                        index["files"]["viewed"].setdefault(path, []).append(step)
                for match in re.finditer(r"grep\s+(?:-[nrRilw]+\s+)*['\"]([^'\"]+)['\"]", content):
                    term = match.group(1).strip()
                    if len(term) > 2:
                        index["files"]["searched"].setdefault(term, []).append(step)
                if "pytest" in content.lower():
                    index["tests"]["ran"].append(step)
            
            elif role == "user" and step > 0:
                for pattern, error_type in [
                    (r"(SyntaxError):\s*(.{0,50})", "SyntaxError"),
                    (r"(NameError):\s*(.{0,50})", "NameError"),
                    (r"(TypeError):\s*(.{0,50})", "TypeError"),
                    (r"(AttributeError):\s*(.{0,50})", "AttributeError"),
                ]:
                    match = re.search(pattern, content)
                    if match:
                        index["errors"].append({"step": step, "type": error_type, "preview": match.group(2)[:30] if match.lastindex >= 2 else ""})
                        break
                if "PASSED" in content:
                    index["tests"]["passed"].append(step)
                if "FAILED" in content:
                    index["tests"]["failed"].append(step)
        
        index["meta"]["total_steps"] = step
        
        expected_file = results_dir / "expected_tests.txt"
        if expected_file.exists():
            section = None
            for line in expected_file.read_text().splitlines():
                line = line.strip()
                if "FAIL_TO_PASS" in line:
                    section = "fail_to_pass"
                elif "PASS_TO_PASS" in line:
                    section = "pass_to_pass"
                elif line and not line.startswith("#") and section:
                    index["expected"][section].append(line)
        
        (results_dir / "trajectory_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    except Exception:
        pass  # Index generation failure doesn't affect main flow


def flatten_domain_config(config: dict, ablation_settings: dict | None = None) -> dict:
    """
    Flatten nested domain config for Jinja2 template rendering.
    
    Supports two sources:
    1. Legacy format: nested dict from get_recreate_agent_config()
    2. New format: flat DomainPromptConfig fields
    
    Args:
        config: Domain configuration dictionary
        ablation_settings: Ablation study settings (optional)
    """
    result = {
        "error_file": config.get("evaluation", {}).get("error_file", config.get("eval_file", "")),
        "codebase_path": config.get("environment", {}).get("codebase_path", config.get("codebase_path", "")),
        "tools_path": config.get("meta_tools", {}).get("tools_path", config.get("tools_path", "")),
        "memory_path": config.get("meta_tools", {}).get("memory_path", config.get("memory_path", "/workspace/agent_memory")),
        "inspect_example": config.get("meta_tools", {}).get("inspect_example", config.get("inspect_example", "")),
        
        "domain": config.get("domain", ""),
        "code_block_lang": config.get("code_block_lang", "bash"),
        "example_instance_id": config.get("example_instance_id", "instance_001"),
        
        "submission_checks": config.get("submission_checks", []),
        "error_file_list": config.get("error_file_list", []),
        "memory_examples": config.get("memory_examples", []),
        "search_examples": config.get("search_examples", []),
        "workflow_notes": config.get("workflow_notes", []),
        
        # Scaffold reference template (supports nested and flat formats)
        "scaffold_system": config.get("scaffold_template", {}).get("system_template", config.get("scaffold_system", "")),
        "scaffold_instance": config.get("scaffold_template", {}).get("instance_template", config.get("scaffold_instance", "")),
        "scaffold_format_error": config.get("scaffold_template", {}).get("format_error_template", config.get("scaffold_format_error", "")),
        
        # Ablation flags (all enabled by default)
        "ablation_trajectory": True,
        "ablation_environment": True,
        "ablation_eval_results": True,
        "ablation_modification_guidance": True,
    }
    
    # Override with provided ablation settings
    if ablation_settings:
        result.update(ablation_settings)
    
    return result


def analyze_recreate_agent_traj(traj_path: Path) -> dict:
    """
    Analyze ReCreate-Agent trajectory file and extract operation statistics.
    
    Returns:
        dict with keys for various ReCreate-Agent operations
    """
    stats = {
        # Read operations
        "read_scaffold": 0,
        "read_submission": 0,
        "read_trajectory": 0,
        "read_eval_result": 0,
        "inspect_docker": 0,
        # Modify operations
        "system_prompt_edit": 0,
        "instance_prompt_edit": 0,
        "memory_template_edit": 0,
        "tool_create": 0,
        "memory_create": 0,
        "web_search": 0,
        # Meta info
        "n_steps": 0,
        "cost": 0.0,
        "exit_status": "Unknown",
    }
    
    if not traj_path.exists():
        return stats
    
    try:
        traj = json.loads(traj_path.read_text())
        stats["n_steps"] = traj.get("n_steps", 0)
        stats["cost"] = traj.get("cost", 0.0)
        stats["exit_status"] = traj.get("exit_status", "Unknown")
        
        for msg in traj.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # Count scaffold reads
                if re.search(r"cat\s+current/scaffold\.yaml", content):
                    stats["read_scaffold"] += 1
                
                # Count submission reads
                if re.search(r"read_trajectory\.py\s+submission", content):
                    stats["read_submission"] += 1
                
                # Count trajectory reads (non-submission)
                if re.search(r"read_trajectory\.py\s+(full|summary|failures|steps)", content):
                    stats["read_trajectory"] += 1
                
                # Count eval result reads
                if re.search(r"cat\s+results/[^/]+/(evaluation\.txt|eval_result\.json|test_output\.txt)", content):
                    stats["read_eval_result"] += 1
                
                # Count environment inspections
                if "inspect_in_docker.py" in content:
                    stats["inspect_docker"] += 1
                
                # Count scaffold edits by type
                if "scaffold_editor.py str_replace" in content or "scaffold_editor.py insert" in content:
                    # Check which template was edited
                    if "system_template" in content:
                        stats["system_prompt_edit"] += 1
                    elif "instance_template" in content:
                        stats["instance_prompt_edit"] += 1
                    elif "memory_template" in content or "When to read memories" in content or "When to write memories" in content:
                        stats["memory_template_edit"] += 1
                    else:
                        # Unknown edit, guess from context
                        # Usually "You are" is system, "Task" is instance
                        if "You are" in content or "Response Format" in content or "## Rules" in content:
                            stats["system_prompt_edit"] += 1
                        elif "{{task}}" in content or "Workflow" in content or "## Task" in content:
                            stats["instance_prompt_edit"] += 1
                        else:
                            # Default to system_prompt_edit
                            stats["system_prompt_edit"] += 1
                
                # Count tool creation
                if "tool_manager.sh create" in content:
                    stats["tool_create"] += 1
                
                # Count memory addition
                if "memory_manager.py add" in content:
                    stats["memory_create"] += 1
                
                # Count web search
                if "web_search.py" in content:
                    stats["web_search"] += 1
    except Exception:
        pass
    
    return stats
