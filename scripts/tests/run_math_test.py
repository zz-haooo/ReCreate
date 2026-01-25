#!/usr/bin/env python3
"""
Math Test Runner

Evaluate evolved scaffolds on Math test set.
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import typer
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj
from recreate_agent.adapters import get_adapter, UnifiedInstance, UnifiedResult
from recreate_agent.adapters.math_adapter import MathAdapter
from recreate_agent.evaluators.math import MathEvaluator, extract_boxed_answer

console = Console()
app = typer.Typer()

AGENT_CONFIG_FIELDS = {
    "system_template", "instance_template", "timeout_template",
    "format_error_template", "action_observation_template",
    "step_limit", "cost_limit"
}


def suppress_verbose_logging():
    for logger_name in [
        "minisweagent", "minisweagent.environment", "minisweagent.agents",
        "LiteLLM", "litellm", "httpx", "openai", "urllib3",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()


def process_single_instance(
    instance: UnifiedInstance,
    scaffold_config: dict,
    output_dir: Path,
    model_name: str,
    tools_dir: Path | None,
    temperature: float = 1.0,
    data_source: str = "math500",
) -> dict:
    """Process single Math instance."""
    adapter = MathAdapter(data_source=data_source)
    instance_id = instance.instance_id
    instance_output_dir = output_dir / instance_id
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    agent_wrapper = None
    exit_status = "Error"
    result_str = ""
    n_steps = 0
    total_cost = 0.0
    eval_result = UnifiedResult(instance_id=instance_id, success=False, error="Agent did not run")

    try:
        # Run agent
        exit_status, result_str, container_id, agent_wrapper = adapter.run_agent(
            instance, scaffold_config, model_name, instance_output_dir, tools_dir, temperature
        )
        
        # Save trajectory
        traj_path = instance_output_dir / f"{instance_id}.traj.json"
        if agent_wrapper and hasattr(agent_wrapper, 'agent') and agent_wrapper.agent:
            save_traj(
                agent_wrapper.agent,
                traj_path,
                exit_status=exit_status,
                result=result_str,
                instance_id=instance_id,
            )
        
        # Evaluate
        eval_result = adapter.evaluate(container_id, instance, instance_output_dir)
        
        # Extract stats
        if agent_wrapper and hasattr(agent_wrapper, 'agent') and agent_wrapper.agent:
            n_steps = agent_wrapper.agent.model.n_calls
            total_cost = agent_wrapper.agent.model.cost
            if hasattr(agent_wrapper.agent, 'messages'):
                traj_file = instance_output_dir / "trajectory.json"
                traj_file.write_text(json.dumps({"messages": agent_wrapper.agent.messages}, indent=2, ensure_ascii=False))

    except Exception as e:
        console.print(f"  [red]Error processing {instance_id}: {e}[/red]")
        eval_result.error = str(e)
        eval_result.success = False
    finally:
        if agent_wrapper:
            try:
                agent_wrapper.cleanup()
            except Exception:
                pass

    duration = (datetime.now() - start_time).total_seconds()

    return {
        "instance_id": instance_id,
        "success": eval_result.success,
        "passed": eval_result.success,
        "subject": instance.category,
        "level": instance.domain_data.get("level", 0),
        "expected_answer": instance.domain_data.get("answer", ""),
        "extracted_answer": eval_result.eval_result.get("extracted_answer", ""),
        "error": eval_result.error,
        "result": eval_result.eval_result.get("result", "unknown"),
        "n_steps": n_steps,
        "cost": total_cost,
        "duration": duration,
    }


@app.command()
def main(
    scaffold: Path = typer.Option(..., "--scaffold", "-s"),
    data_source: str = typer.Option("math500", "--data-source", "-d", 
                                     help="Data source: math500 | aime24 | aime25 | amc23"),
    subset: str = typer.Option(None, "--subset", help="Subject filter (math500 only)"),
    output: Path = typer.Option(..., "--output", "-o"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m"),
    workers: int = typer.Option(4, "--workers", "-j"),
    max_instances: int = typer.Option(None, "--max-instances", "-n"),
    skip_instances: int = typer.Option(0, "--skip", help="Skip first N instances (for test set)"),
    tools_dir: Path = typer.Option(None, "--tools-dir"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Agent temperature"),
    shuffle_file: Path = typer.Option(None, "--shuffle-file", help="Shuffle file path"),
):
    """Run Math test with specified scaffold."""
    
    suppress_verbose_logging()
    
    console.print("[cyan]Math Test[/cyan]")
    console.print(f"Scaffold: {scaffold}")
    console.print(f"Data source: {data_source}")
    console.print(f"Subject: {subset if subset else 'All'}")
    console.print(f"Temperature: {temperature}")
    
    if not scaffold.exists():
        console.print(f"[red]Scaffold not found: {scaffold}[/red]")
        raise typer.Exit(1)
    
    full_config = yaml.safe_load(scaffold.read_text())
    
    # Merge memory_template into system_template if present
    if "memory_template" in full_config and full_config["memory_template"]:
        memory_section = full_config["memory_template"].strip()
        system_tpl = full_config.get("system_template", "")
        if memory_section and memory_section not in system_tpl:
            full_config["system_template"] = f"{system_tpl}\n\n{memory_section}"
    
    scaffold_config = {k: v for k, v in full_config.items() if k in AGENT_CONFIG_FIELDS}
    
    adapter = MathAdapter(data_source=data_source)
    
    # Load dataset
    all_instances = adapter.load_dataset(subset=subset, shuffle_file=shuffle_file)
    
    if skip_instances > 0:
        console.print(f"[yellow]Skipping first {skip_instances} instances[/yellow]")
        instances = all_instances[skip_instances:]
    else:
        instances = all_instances
    
    if max_instances:
        instances = instances[:max_instances]
    
    if not instances:
        console.print("[red]No instances found for testing. Check dataset, subset, and skip/max configuration.[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(instances)} instances")
    
    # Subject distribution
    subject_counts = {}
    for inst in instances:
        subject_counts[inst.category] = subject_counts.get(inst.category, 0) + 1
    console.print("Subject distribution:")
    for subj, count in sorted(subject_counts.items()):
        console.print(f"  {subj}: {count}")

    output.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    results = []
    total_cost = 0.0
    total_steps = 0
    
    if workers > 1:
        console.print(f"\n[cyan]Parallel execution ({workers} workers)...[/cyan]")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_instance,
                    inst,
                    scaffold_config,
                    output,
                    model,
                    tools_dir,
                    temperature,
                    data_source,
                ): inst.instance_id
                for inst in instances
            }

            for future in Progress().track(as_completed(futures), total=len(instances), description="Testing"):
                instance_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_cost += result.get("cost", 0.0)
                    total_steps += result.get("n_steps", 0)
                    status_icon = "✓" if result.get("passed") else "✗"
                    console.print(f"  {status_icon} {instance_id} [{result.get('subject', '')}]: {result.get('n_steps', 0)} steps, ${result.get('cost', 0.0):.4f}, {result.get('result', 'unknown')}")
                except Exception as e:
                    console.print(f"  [red]⚠ {instance_id}: Error - {e}[/red]")
                    results.append({
                        "instance_id": instance_id,
                        "success": False, "passed": False,
                        "subject": "unknown", "level": 0,
                        "expected_answer": "", "extracted_answer": "",
                        "error": str(e), "result": "error",
                        "n_steps": 0, "cost": 0.0, "duration": 0.0,
                    })
    else:
        console.print(f"\n[cyan]Sequential execution...[/cyan]")
        for i, inst in enumerate(instances, 1):
            result = process_single_instance(
                inst, scaffold_config, output, model, tools_dir, temperature, data_source
            )
            results.append(result)
            total_cost += result.get("cost", 0.0)
            total_steps += result.get("n_steps", 0)
            status_icon = "✓" if result.get("passed") else "✗"
            console.print(f"[{i}/{len(instances)}] {status_icon} {inst.instance_id} [{result.get('subject', '')}]: {result.get('n_steps', 0)} steps, ${result.get('cost', 0.0):.4f}, {result.get('result', 'unknown')}")
    
    # Statistics
    passed_count = sum(1 for r in results if r.get("passed"))
    
    # Per-subject stats
    subject_results = {}
    for r in results:
        subj = r.get("subject", "unknown")
        if subj not in subject_results:
            subject_results[subj] = {"total": 0, "passed": 0}
        subject_results[subj]["total"] += 1
        if r.get("passed"):
            subject_results[subj]["passed"] += 1
    
    # Per-level stats (for MATH-500)
    level_results = {}
    for r in results:
        level = r.get("level", 0)
        if level:
            if level not in level_results:
                level_results[level] = {"total": 0, "passed": 0}
            level_results[level]["total"] += 1
            if r.get("passed"):
                level_results[level]["passed"] += 1

    summary = {
        "scaffold": str(scaffold),
        "data_source": data_source,
        "subset": subset,
        "model": model,
        "temperature": temperature,
        "skip_instances": skip_instances,
        "total": len(instances),
        "successful": len(results),
        "passed": passed_count,
        "pass_rate": passed_count / len(instances) if instances else 0,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "subject_results": subject_results,
        "level_results": level_results,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2))

    console.print(f"\n[green]✓ Math Test Complete[/green]")
    
    console.print("\nOverall Results")
    console.print(f"  Total: {summary['total']}")
    console.print(f"  Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
    console.print(f"  Total Cost: ${summary['total_cost']:.4f}")
    console.print(f"  Total Steps: {summary['total_steps']}")

    if subject_results:
        console.print("\nPer-Subject Statistics")
        console.print(f"{'Subject':<20} {'Count':<8} {'Passed':<8} {'Rate':<8}")
        console.print("-" * 50)
        for subj, stats in sorted(subject_results.items()):
            rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            console.print(f"{subj:<20} {stats['total']:<8} {stats['passed']:<8} {rate:.1f}%")
    
    if level_results:
        console.print("\nPer-Level Statistics")
        console.print(f"{'Level':<8} {'Count':<8} {'Passed':<8} {'Rate':<8}")
        console.print("-" * 40)
        for level, stats in sorted(level_results.items()):
            rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            console.print(f"{level:<8} {stats['total']:<8} {stats['passed']:<8} {rate:.1f}%")

    console.print(f"\nOutput: {output}")
    console.print(f"Summary: {output}/run_summary.json")


if __name__ == "__main__":
    app()
