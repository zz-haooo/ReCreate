"""Core evolution logic - initial creation, single evolution, and batch synthesis."""

import json
import os
import shutil
import time
import yaml
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from .utils import safe_print, safe_load_yaml, BatchResult, console
from .trajectory import generate_trajectory_index, flatten_domain_config
from .scaffold_ops import (
    load_scaffold, init_agent_memory, create_submission_only_tool,
    create_default_scaffold, merge_tools_and_memories,
)
from .stats import collect_batch_stats, save_batch_evolution_stats


def run_recreate_initial_creation(
    workspace: Path,
    runs_recreate_dir: Path,
    recreate_model: str,
    recreate_temp: float,
    domain: str,
    domain_description: str,
    initial_prompt_template: str,
    domain_config: dict | None = None,
    ablation_settings: dict | None = None,
) -> bool:
    """Create initial scaffold using ReCreate-Agent. Returns True if successful."""
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.agents.default import NonTerminatingException, TerminatingException
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model
    
    console.print("[bold cyan]Phase 0: ReCreate-Agent Creates Initial Agent Scaffold[/bold cyan]")
    
    if domain_config is None:
        domain_config = {
            "domain": domain,
            "domain_description": domain_description,
            "environment": {"codebase_path": "/testbed/" if domain == "swe" else "/workspace/"},
            "scaffold_template": {"system_template": "You are an AI assistant."},
            "rules": {"anti_cheat": []},
            "submission": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
            "suggested_limits": {"step_limit": 100, "cost_limit": 5.0},
            "meta_tools": {"tools_path": "/workspace", "memory_path": "/workspace/agent_memory"},
        }
    
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "recreate_agent" / "prompts"
    system_template_raw = (prompts_dir / "meta_system.jinja2").read_text()
    instance_template_raw = (prompts_dir / initial_prompt_template).read_text()
    
    # Flatten config for DomainPromptConfig fields and ablation settings
    flat_config = flatten_domain_config(domain_config, ablation_settings)
    
    # Merge nested and flat configs for instance template
    merged_config = {**domain_config, **flat_config}
    
    # Render with Jinja2 Template
    system_template = Template(system_template_raw).render(**flat_config)
    instance_template = Template(instance_template_raw).render(**merged_config)
    
    config = AgentConfig(
        system_template=system_template,
        instance_template=instance_template,
        action_observation_template="<output>\n{{output.output}}\n</output>",
        format_error_template="Please provide EXACTLY ONE bash command in triple backticks.",
        step_limit=30,
        cost_limit=1.0,
    )
    
    model_kwargs = {
        "custom_llm_provider": "openai",
        "temperature": recreate_temp,
    }
    api_base = os.getenv("LLM_API_BASE")
    if api_base:
        model_kwargs["api_base"] = api_base
    model = get_model(recreate_model, {"model_kwargs": model_kwargs})
    
    env = LocalEnvironment(cwd=str(workspace), timeout=120, env={"PATH": "/usr/local/bin:/usr/bin:/bin"})
    agent = DefaultAgent(model, env, config_class=lambda **kw: config)
    
    try:
        agent.extra_template_vars |= {"task": f"Create initial scaffold for {domain} tasks"}
        agent.messages = []
        agent.add_message("system", system_template)
        agent.add_message("user", instance_template)
        
        exit_status = "Running"
        while True:
            print(f"\rReCreate-Agent creating initial agent scaffold: step {model.n_calls + 1} (${model.cost:.2f})...", end="", flush=True)
            try:
                agent.step()
            except NonTerminatingException as e:
                agent.add_message("user", str(e))
            except TerminatingException as e:
                exit_status = type(e).__name__
                break
            except Exception as e:
                exit_status = f"Error: {type(e).__name__}"
                console.print(f"\n[yellow]Warning: {e}[/yellow]")
                break
        
        print(f"\rReCreate-Agent: {model.n_calls} steps | ${model.cost:.4f} | {exit_status}          ")
        
    except Exception as e:
        console.print(f"[red]Error during initial creation: {e}[/red]")
        return False
    
    scaffold_file = workspace / "current" / "scaffold.yaml"
    if scaffold_file.exists():
        current_dir = workspace / "current"
        global_v000 = workspace / "global_v000"
        
        if global_v000.exists():
            shutil.rmtree(global_v000)
        shutil.move(str(current_dir), str(global_v000))
        current_dir.symlink_to("global_v000")
        
        (global_v000 / "agent_tools").mkdir(exist_ok=True)
        init_agent_memory(global_v000)
        
        console.print(f"[green]✓ Initial scaffold created: global_v000[/green]")
        return True
    
    console.print("[yellow]Warning: ReCreate-Agent didn't create scaffold, using default[/yellow]")
    create_default_scaffold(workspace, domain, flat_config)
    return True


def run_single_evolution(
    instance,
    global_scaffold_dir: Path,
    output_dir: Path,
    adapter,
    agent_model: str,
    recreate_model: str,
    agent_temp: float,
    recreate_temp: float,
    domain: str,
    domain_config: dict,
    runs_recreate_dir: Path,
    ablation_settings: dict | None = None,
) -> BatchResult:
    """
    Run a single instance: Agent + Evaluate + ReCreate-Agent.
    
    All operations are isolated to output_dir.
    Does NOT modify global scaffold or current symlink.
    """
    from minisweagent.run.utils.save import save_traj
    from recreate_agent.adapters import UnifiedResult
    
    instance_id = instance.instance_id
    start_time = time.time()
    agent_wrapper = None
    container_id = None
    eval_result = None
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy global scaffold as working copy
        shutil.copy(global_scaffold_dir / "scaffold.yaml", output_dir / "scaffold.yaml")
        if (global_scaffold_dir / "agent_tools").exists():
            shutil.copytree(global_scaffold_dir / "agent_tools", output_dir / "agent_tools")
        else:
            (output_dir / "agent_tools").mkdir()
        if (global_scaffold_dir / "agent_memory").exists():
            shutil.copytree(global_scaffold_dir / "agent_memory", output_dir / "agent_memory")
        else:
            init_agent_memory(output_dir)
        
        # Load scaffold - handle YAML parsing errors gracefully
        try:
            scaffold = load_scaffold(global_scaffold_dir)
            if not scaffold:
                raise ValueError("Failed to load scaffold: empty or invalid YAML")
        except Exception as e:
            safe_print(f"  [red]{instance_id}: Error - {e}[/red]")
            return BatchResult(
                instance_id=instance_id,
                success=False, score=0.0,
                scaffold_changed=False, has_new_tools=False, has_new_memories=False,
                error=f"Scaffold loading error: {str(e)}", exit_status="YAMLError",
                duration=time.time() - start_time
            )
        
        # ===== Phase 1: Run Agent =====
        safe_print(f"  [dim]Starting {instance_id}...[/dim]")
        
        # Use adapter.run_agent uniformly
        try:
            exit_status, result, container_id, agent_wrapper = adapter.run_agent(
                instance, scaffold, agent_model, output_dir, 
                output_dir,  # tools_dir = output_dir (contains agent_tools/)
                agent_temp
            )
        except Exception as e:
            safe_print(f"  [red]{instance_id}: Agent error - {e}[/red]")
            # Cleanup container
            if agent_wrapper:
                try:
                    agent_wrapper.cleanup()
                except Exception:
                    pass
            return BatchResult(
                instance_id=instance_id,
                success=False, score=0.0,
                scaffold_changed=False, has_new_tools=False, has_new_memories=False,
                error=str(e), exit_status="Error",
                duration=time.time() - start_time
            )
        
        # Save trajectory
        traj_path = output_dir / f"{instance_id}.traj.json"
        if agent_wrapper and hasattr(agent_wrapper, 'agent') and agent_wrapper.agent:
            save_traj(agent_wrapper.agent, traj_path, exit_status=exit_status, result=result, instance_id=instance_id)
        
        # ===== Phase 2: Evaluate =====
        # AppWorld already evaluated in subprocess
        if eval_result is None:
            if container_id:
                eval_result = adapter.evaluate(container_id, instance, output_dir)
            else:
                eval_result = UnifiedResult(instance_id=instance_id, success=False, error="No container")
        
        # Save evaluation
        (output_dir / "evaluation.txt").write_text(eval_result.formatted_output)
        (output_dir / "eval_result.json").write_text(json.dumps({
            "success": eval_result.success,
            "score": eval_result.score,
            "error": eval_result.error,
            "details": eval_result.details,
        }, indent=2, default=str))
        
        # Display result in domain-appropriate format
        if domain == "swe":
            result_icon = "✓" if eval_result.success else "✗"
            tests_passed = eval_result.details.get('tests_passed', 0)
            tests_failed = eval_result.details.get('tests_failed', 0)
            status_str = f"{'Resolved' if eval_result.success else 'Failed'} ({tests_passed}P/{tests_failed}F)"
        elif domain == "data_science":
            # DA-Code: continuous score, not pass/fail
            result_icon = "●"
            status_str = f"Score: {eval_result.score:.2f}"
        elif domain in ("ds1000", "math"):
            result_icon = "✓" if eval_result.success else "✗"
            status_str = f"{'Passed' if eval_result.success else 'Failed'}"
        elif domain == "appworld":
            result_icon = "✓" if eval_result.success else "✗"
            pass_count = eval_result.details.get('pass_count', 0)
            fail_count = eval_result.details.get('fail_count', 0)
            status_str = f"{'Passed' if eval_result.success else 'Failed'} ({pass_count}/{pass_count + fail_count} assertions)"
        else:
            result_icon = "✓" if eval_result.success else "✗"
            status_str = f"score={eval_result.score:.2f}"
        safe_print(f"  {result_icon} {instance_id}: {status_str}")
        
        # ===== Phase 3: Run ReCreate-Agent (isolated) =====
        scaffold_changed = run_recreate_evolution_isolated(
            instance=instance,
            eval_result=eval_result,
            agent_exit_status=exit_status,
            working_dir=output_dir,
            base_scaffold_dir=global_scaffold_dir,
            traj_path=traj_path,
            container_id=container_id,
            recreate_model=recreate_model,
            recreate_temp=recreate_temp,
            domain=domain,
            domain_config=domain_config,
            runs_recreate_dir=runs_recreate_dir,
            ablation_settings=ablation_settings,
        )
        
        # Check for new tools/memories
        has_new_tools = False
        has_new_memories = False
        
        tools_dir = output_dir / "agent_tools"
        if tools_dir.exists():
            orig_tools = set((global_scaffold_dir / "agent_tools").iterdir()) if (global_scaffold_dir / "agent_tools").exists() else set()
            new_tools = set(tools_dir.iterdir())
            has_new_tools = len(new_tools) > len(orig_tools)
        
        memories_file = output_dir / "agent_memory" / "memories.yaml"
        if memories_file.exists():
            orig_mem = (global_scaffold_dir / "agent_memory" / "memories.yaml")
            if orig_mem.exists():
                has_new_memories = memories_file.read_text() != orig_mem.read_text()
        
        # Cleanup
        if agent_wrapper:
            try:
                agent_wrapper.cleanup()
            except Exception:
                pass
        
        return BatchResult(
            instance_id=instance_id,
            success=eval_result.success,
            score=eval_result.score,
            scaffold_changed=scaffold_changed,
            has_new_tools=has_new_tools,
            has_new_memories=has_new_memories,
            exit_status=exit_status,
            duration=time.time() - start_time
        )
        
    except Exception as e:
        safe_print(f"  [red]{instance_id}: Error - {e}[/red]")
        # Ensure container cleanup on exception
        if agent_wrapper is not None:
            try:
                agent_wrapper.cleanup()
            except Exception:
                pass
        return BatchResult(
            instance_id=instance_id,
            success=False, score=0.0,
            scaffold_changed=False, has_new_tools=False, has_new_memories=False,
            error=str(e), exit_status=type(e).__name__,
            duration=time.time() - start_time
        )


def run_recreate_evolution_isolated(
    instance,
    eval_result,
    agent_exit_status: str,
    working_dir: Path,
    base_scaffold_dir: Path,
    traj_path: Path,
    container_id: str,
    recreate_model: str,
    recreate_temp: float,
    domain: str,
    domain_config: dict,
    runs_recreate_dir: Path,
    ablation_settings: dict | None = None,
) -> bool:
    """
    Run ReCreate-Agent in isolation - modifies working_dir/scaffold.yaml only.
    Returns True if scaffold was modified.
    """
    from recreate_agent.recreate_agent import ReCreateAgent
    
    instance_id = instance.instance_id
    
    # Create Meta-Agent temp workspace
    recreate_temp_dir = working_dir / "recreate_temp"
    if recreate_temp_dir.exists():
        shutil.rmtree(recreate_temp_dir)
    recreate_temp_dir.mkdir()
    
    # Setup workspace structure
    meta_working = recreate_temp_dir / "working"
    meta_working.mkdir()
    shutil.copy(working_dir / "scaffold.yaml", meta_working / "scaffold.yaml")
    (recreate_temp_dir / "current").symlink_to("working")
    
    # Copy tools and prompts (with ablation filtering)
    src_dir = Path(__file__).parent.parent.parent / "src" / "recreate_agent"
    tools_dir = recreate_temp_dir / "tools"
    tools_dir.mkdir()
    
    # Selectively copy tools based on ablation settings
    ablation = ablation_settings or {}
    ablate_trajectory = not ablation.get("ablation_trajectory", True)
    ablate_eval_results = not ablation.get("ablation_eval_results", True)
    ablate_environment = not ablation.get("ablation_environment", True)
    
    for tool_file in (src_dir / "tools").iterdir():
        # read_trajectory.py serves two purposes
        if tool_file.name == "read_trajectory.py":
            if ablate_trajectory and ablate_eval_results:
                # Both ablated: don't copy
                continue
            elif ablate_trajectory:
                # Only trajectory ablated: create submission-only version
                create_submission_only_tool(tools_dir / tool_file.name)
                continue
        # Ablate environment: don't provide inspect_in_docker.py
        if ablate_environment and tool_file.name == "inspect_in_docker.py":
            continue
        shutil.copy(tool_file, tools_dir / tool_file.name)
    
    shutil.copytree(src_dir / "prompts", recreate_temp_dir / "prompts")
    
    # Save container ID for inspection (only if environment ablation is disabled)
    if container_id and not ablate_environment:
        (recreate_temp_dir / "container_id.txt").write_text(container_id)
    
    # Prepare results for Meta-Agent
    results_dir = recreate_temp_dir / "results" / instance_id
    results_dir.mkdir(parents=True)
    
    # Extract n_steps and total_cost from trajectory for Meta-Agent display
    n_steps = 0
    total_cost = 0.0
    traj_data = None
    if traj_path.exists():
        try:
            traj_data = json.loads(traj_path.read_text())
            info = traj_data.get("info", {})
            model_stats = info.get("model_stats", {})
            n_steps = model_stats.get("api_calls", 0)
            total_cost = model_stats.get("instance_cost", 0.0)
            # Fallback: count assistant messages if no api_calls
            if n_steps == 0:
                n_steps = len([m for m in traj_data.get("messages", []) if m.get("role") == "assistant"])
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Copy trajectory based on ablation settings
        if not ablate_trajectory:
            # Not ablated: copy full trajectory
            copied_traj = results_dir / traj_path.name
            shutil.copy(traj_path, copied_traj)
            # Auto-generate index file
            generate_trajectory_index(copied_traj, results_dir)
        elif not ablate_eval_results and traj_data:
            # Ablate trajectory but keep eval_results: create submission-only version
            minimal_traj = {
                "info": {
                    "submission": traj_data.get("info", {}).get("submission", ""),
                    "exit_status": traj_data.get("info", {}).get("exit_status", ""),
                    "model_stats": model_stats,
                },
                "messages": [],
            }
            (results_dir / traj_path.name).write_text(json.dumps(minimal_traj, indent=2))
    
    # Ablate eval_results: don't copy evaluation files
    if not ablate_eval_results:
        (results_dir / "evaluation.txt").write_text(eval_result.formatted_output)
        
        # Copy error files and test information
        for filename in ["test_output.txt", "eval_result.json", "expected_tests.txt", "test_patch.txt"]:
            src = working_dir / filename
            if src.exists():
                shutil.copy(src, results_dir / filename)
    
    # Save result.json (must include exit_status, n_steps, total_cost for Meta-Agent display)
    effective_exit_status = agent_exit_status if agent_exit_status else ("Resolved" if eval_result.success else "Failed")
    
    # Extract problem summary for Meta-Agent (truncate if too long)
    problem_summary = instance.problem_statement[:1500] if instance.problem_statement else ""
    if len(instance.problem_statement) > 1500:
        problem_summary += "\n... (truncated)"
    
    result_data = {
        "instance_id": instance_id,
        "domain": domain,
        "exit_status": effective_exit_status,
        "success": eval_result.success,
        "score": eval_result.score,
        "error": eval_result.error,
        "details": eval_result.details,
        "eval_result": eval_result.eval_result if eval_result.eval_result else eval_result.details,
        # Add execution stats for Meta-Agent display
        "n_steps": n_steps,
        "total_cost": total_cost,
        # Add problem summary for Meta-Agent context
        "problem_summary": problem_summary,
    }
    (results_dir / "result.json").write_text(json.dumps(result_data, indent=2, default=str))
    
    # Copy agent_tools and agent_memory
    if (working_dir / "agent_tools").exists():
        shutil.copytree(working_dir / "agent_tools", meta_working / "agent_tools")
    else:
        (meta_working / "agent_tools").mkdir()
    
    if (working_dir / "agent_memory").exists():
        shutil.copytree(working_dir / "agent_memory", meta_working / "agent_memory")
    else:
        (meta_working / "agent_memory").mkdir()
    
    # Run ReCreate-Agent with proper cleanup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"parallel_{instance_id}_{timestamp}"
    scaffold_changed = False
    
    # Merge ablation settings into domain_config for template rendering
    merged_domain_config = {**domain_config}
    if ablation_settings:
        merged_domain_config.update(ablation_settings)
    
    try:
        recreate_agent = ReCreateAgent(
            model_name=recreate_model,
            workspace_path=recreate_temp_dir,
            run_name=run_name,
            runs_recreate_dir=runs_recreate_dir,
            domain_config=merged_domain_config,
            model_kwargs={"temperature": recreate_temp},
        )
        
        exit_status, traj_path_out = recreate_agent.evolve(evolution_id=run_name, quiet=True)
        
        # Output ReCreate-Agent result
        recreate_steps = recreate_agent.model.n_calls
        recreate_cost = recreate_agent.model.cost
        status_icon = "✓" if exit_status in ["Submitted", "Resolved"] else "✗"
        safe_print(f"  ReCreate-Agent: {recreate_steps} steps | ${recreate_cost:.4f} | {status_icon} {exit_status}")
        
        # Check if scaffold changed
        evolved_scaffold = recreate_temp_dir / "working" / "scaffold.yaml"
        original_scaffold = working_dir / "scaffold.yaml"
        
        if evolved_scaffold.exists():
            # Validate YAML
            try:
                yaml.safe_load(evolved_scaffold.read_text())
                # Read original content BEFORE copying (important!)
                original_content = original_scaffold.read_text()
                evolved_content = evolved_scaffold.read_text()
                
                if evolved_content != original_content:
                    scaffold_changed = True
                    
                    # Generate diff BEFORE overwriting original
                    import difflib
                    orig_lines = original_content.splitlines(keepends=True)
                    new_lines = evolved_content.splitlines(keepends=True)
                    diff = difflib.unified_diff(orig_lines, new_lines, fromfile="before", tofile="after")
                    (working_dir / "scaffold_diff.txt").write_text("".join(diff))
                    
                    # Now copy the evolved scaffold
                    shutil.copy(evolved_scaffold, working_dir / "scaffold.yaml")
            except yaml.YAMLError:
                pass
        
        # Copy evolution summary if exists
        if (recreate_temp_dir / "working" / "evolution_summary.md").exists():
            shutil.copy(recreate_temp_dir / "working" / "evolution_summary.md", working_dir / "evolution_summary.md")
        
        # Copy evolved tools
        evolved_tools = recreate_temp_dir / "working" / "agent_tools"
        if evolved_tools.exists():
            for tool in evolved_tools.iterdir():
                if tool.is_dir():
                    dest = working_dir / "agent_tools" / tool.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(tool, dest)
        
        # Copy evolved memories
        evolved_memory = recreate_temp_dir / "working" / "agent_memory"
        if evolved_memory.exists():
            dest_memory = working_dir / "agent_memory"
            if dest_memory.exists():
                shutil.rmtree(dest_memory)
            shutil.copytree(evolved_memory, dest_memory)
    
    finally:
        # Always cleanup recreate_temp_dir
        if recreate_temp_dir.exists():
            shutil.rmtree(recreate_temp_dir)
    
    return scaffold_changed


def run_batch_synthesis(
    workspace: Path,
    batch_dir: Path,
    batch_results: dict,
    current_global_dir: Path,
    new_global_dir: Path,
    batch_idx: int,
    recreate_model: str,
    recreate_temp: float,
    domain: str,
    domain_config: dict,
    runs_recreate_dir: Path,
    ablation_settings: dict | None = None,
    EvolutionStats=None,
) -> bool:
    """
    Synthesis Meta-Agent: Analyze all batch modifications and create unified global scaffold.
    """
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.agents.default import NonTerminatingException, TerminatingException
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model
    
    console.print(f"\n[bold cyan]Batch {batch_idx} Synthesis: Analyzing {len(batch_results)} results[/bold cyan]")
    
    # Collect modifications
    modifications = []
    for instance_id, result in batch_results.items():
        instance_dir = batch_dir / instance_id
        if result.scaffold_changed:
            diff_file = instance_dir / "scaffold_diff.txt"
            summary_file = instance_dir / "evolution_summary.md"
            modifications.append({
                "instance_id": instance_id,
                "success": result.success,
                "score": result.score,
                "diff": diff_file.read_text() if diff_file.exists() else "",
                "summary": summary_file.read_text() if summary_file.exists() else "",
                "scaffold_path": instance_dir / "scaffold.yaml",
                "tools_path": instance_dir / "agent_tools",
                "memory_path": instance_dir / "agent_memory",
            })
    
    if not modifications:
        console.print(f"  [dim]No modifications in this batch, copying current global[/dim]")
        shutil.copytree(current_global_dir, new_global_dir)
        return False
    
    console.print(f"  [cyan]{len(modifications)} instances proposed modifications[/cyan]")
    
    # Create synthesis workspace
    synthesis_temp = batch_dir / "synthesis_temp"
    if synthesis_temp.exists():
        shutil.rmtree(synthesis_temp)
    synthesis_temp.mkdir()
    
    working_dir = synthesis_temp / "working"
    working_dir.mkdir()
    shutil.copy(current_global_dir / "scaffold.yaml", working_dir / "scaffold.yaml")
    (synthesis_temp / "current").symlink_to("working")
    
    # Copy tools (with ablation filtering)
    src_dir = Path(__file__).parent.parent.parent / "src" / "recreate_agent"
    tools_dir = synthesis_temp / "tools"
    tools_dir.mkdir()
    
    ablation = ablation_settings or {}
    ablate_trajectory = not ablation.get("ablation_trajectory", True)
    ablate_eval_results = not ablation.get("ablation_eval_results", True)
    ablate_environment = not ablation.get("ablation_environment", True)
    
    for tool_file in (src_dir / "tools").iterdir():
        # read_trajectory.py serves two purposes
        if tool_file.name == "read_trajectory.py":
            if ablate_trajectory and ablate_eval_results:
                continue
            elif ablate_trajectory:
                create_submission_only_tool(tools_dir / tool_file.name)
                continue
        if ablate_environment and tool_file.name == "inspect_in_docker.py":
            continue
        shutil.copy(tool_file, tools_dir / tool_file.name)
    
    shutil.copytree(src_dir / "prompts", synthesis_temp / "prompts")
    
    # Copy agent_tools and agent_memory
    if (current_global_dir / "agent_tools").exists():
        shutil.copytree(current_global_dir / "agent_tools", working_dir / "agent_tools")
    else:
        (working_dir / "agent_tools").mkdir()
    
    if (current_global_dir / "agent_memory").exists():
        shutil.copytree(current_global_dir / "agent_memory", working_dir / "agent_memory")
    else:
        (working_dir / "agent_memory").mkdir()
    
    # Create modifications directory for Meta-Agent to read
    mods_dir = synthesis_temp / "batch_modifications"
    mods_dir.mkdir()
    for mod in modifications:
        mod_dir = mods_dir / mod["instance_id"]
        mod_dir.mkdir()
        if mod["diff"]:
            (mod_dir / "diff.txt").write_text(mod["diff"])
        if mod["summary"]:
            (mod_dir / "summary.md").write_text(mod["summary"])
        shutil.copy(mod["scaffold_path"], mod_dir / "scaffold.yaml")
    
    # Load and render prompts
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "recreate_agent" / "prompts"
    system_template_raw = (prompts_dir / "meta_system.jinja2").read_text()
    instance_template_raw = (prompts_dir / "meta_batch_synthesis.jinja2").read_text()
    
    # Render with Jinja2 Template
    flat_config = flatten_domain_config(domain_config, ablation_settings)
    system_template = Template(system_template_raw).render(**flat_config)
    
    instance_template = Template(instance_template_raw).render(
        batch_idx=batch_idx,
        num_instances=len(batch_results),
        num_modifications=len(modifications),
        modifications=modifications,
    )
    
    config = AgentConfig(
        system_template=system_template,
        instance_template=instance_template,
        action_observation_template="<output>\n{{output.output}}\n</output>",
        format_error_template="Please provide EXACTLY ONE bash command in triple backticks.",
        step_limit=50,
        cost_limit=3.0,
    )
    
    model_kwargs = {
        "custom_llm_provider": "openai",
        "temperature": recreate_temp,
    }
    api_base = os.getenv("LLM_API_BASE")
    if api_base:
        model_kwargs["api_base"] = api_base
    model = get_model(recreate_model, {"model_kwargs": model_kwargs})
    
    env = LocalEnvironment(cwd=str(synthesis_temp), timeout=120, env={"PATH": "/usr/local/bin:/usr/bin:/bin"})
    agent = DefaultAgent(model, env, config_class=lambda **kw: config)
    
    # Run ReCreate-Agent
    try:
        agent.messages = []
        agent.add_message("system", system_template)
        agent.add_message("user", instance_template)
        
        exit_status = "Running"
        while True:
            print(f"\r  Synthesis: step {model.n_calls + 1} (${model.cost:.2f})...", end="", flush=True)
            try:
                agent.step()
            except NonTerminatingException as e:
                agent.add_message("user", str(e))
            except TerminatingException as e:
                exit_status = type(e).__name__
                break
            except Exception as e:
                exit_status = f"Error: {type(e).__name__}"
                console.print(f"\n[yellow]Warning: {e}[/yellow]")
                break
        
        print(f"\r  Synthesis: {model.n_calls} steps | ${model.cost:.4f} | {exit_status}          ")
        
    except Exception as e:
        console.print(f"[red]Error during synthesis: {e}[/red]")
        shutil.copytree(current_global_dir, new_global_dir)
        shutil.rmtree(synthesis_temp)
        return False
    
    # Save trajectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_dir = runs_recreate_dir / f"synthesis_batch{batch_idx:03d}_{timestamp}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_data = {
        "instance_id": f"synthesis_batch_{batch_idx}",
        "model": recreate_model,
        "exit_status": exit_status,
        "n_steps": model.n_calls,
        "cost": model.cost,
        "messages": agent.messages,
        "timestamp": datetime.now().isoformat(),
    }
    (traj_dir / "synthesis.traj.json").write_text(json.dumps(traj_data, indent=2, ensure_ascii=False))
    
    # Create new global version
    new_global_dir.mkdir(parents=True, exist_ok=True)
    
    evolved_scaffold = synthesis_temp / "working" / "scaffold.yaml"
    if evolved_scaffold.exists():
        try:
            yaml.safe_load(evolved_scaffold.read_text())
            shutil.copy(evolved_scaffold, new_global_dir / "scaffold.yaml")
        except yaml.YAMLError:
            console.print(f"  [yellow]Synthesis produced invalid YAML, using current global[/yellow]")
            shutil.copy(current_global_dir / "scaffold.yaml", new_global_dir / "scaffold.yaml")
    else:
        shutil.copy(current_global_dir / "scaffold.yaml", new_global_dir / "scaffold.yaml")
    
    # Copy synthesis summary
    synthesis_summary = synthesis_temp / "working" / "synthesis_summary.md"
    if synthesis_summary.exists():
        shutil.copy(synthesis_summary, new_global_dir / "synthesis_summary.md")
    
    # Merge tools and memories
    merge_tools_and_memories(current_global_dir, modifications, new_global_dir, synthesis_temp / "working")
    
    # Collect and save batch evolution stats (with cumulative tracking)
    if EvolutionStats is not None:
        new_version = int(new_global_dir.name.replace("global_v", ""))
        prev_version = int(current_global_dir.name.replace("global_v", ""))
        batch_stats = collect_batch_stats(
            batch_dir=batch_dir,
            batch_results=batch_results,
            runs_recreate_dir=runs_recreate_dir,
            batch_idx=batch_idx,
            new_version=new_version,
            prev_version=prev_version,
            EvolutionStats=EvolutionStats,
        )
        save_batch_evolution_stats(batch_stats, new_global_dir, prev_version_dir=current_global_dir)
    
    # Cleanup
    shutil.rmtree(synthesis_temp)
    
    console.print(f"  [green]✓ Created global_v{int(new_global_dir.name.replace('global_v', ''))}[/green]")
    return True
