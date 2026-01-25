#!/usr/bin/env python3
"""
DS-1000 Test Runner

Evaluate evolved scaffolds on DS-1000 test set.

Usage:
    python run_ds1000_test.py --scaffold path/to/scaffold.yaml --subset Pandas --output results/
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

import yaml
import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "datasets" / "dacode"))


def suppress_verbose_logging():
    for logger_name in [
        "minisweagent", "minisweagent.environment", "minisweagent.agents",
        "LiteLLM", "litellm", "httpx", "openai", "urllib3",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()


app = typer.Typer()
console = Console()
console_lock = Lock()


def safe_print(msg: str):
    with console_lock:
        console.print(msg)


AGENT_CONFIG_FIELDS = {
    "system_template", "instance_template", "timeout_template",
    "format_error_template", "action_observation_template",
    "step_limit", "cost_limit"
}


@dataclass
class DS1000TestResult:
    instance_id: str
    success: bool
    passed: bool
    library: str
    perturbation_type: str
    error: str = ""
    result_str: str = ""
    n_steps: int = 0
    cost: float = 0.0
    duration: float = 0.0
    generated_code: str = ""
    
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "success": self.success,
            "passed": self.passed,
            "library": self.library,
            "perturbation_type": self.perturbation_type,
            "error": self.error,
            "result": self.result_str,
            "n_steps": self.n_steps,
            "cost": self.cost,
            "duration": self.duration,
        }


def process_single_instance(
    instance,
    adapter,
    scaffold_config: dict,
    output_dir: Path,
    model: str,
    tools_dir: Path | None,
    temperature: float,
) -> DS1000TestResult:
    """Process single DS-1000 instance."""
    from recreate_agent.adapters import UnifiedResult
    
    instance_id = instance.instance_id
    start_time = time.time()
    
    instance_output = output_dir / instance_id
    instance_output.mkdir(parents=True, exist_ok=True)
    
    container_id = None
    agent_wrapper = None
    n_steps = 0
    cost = 0.0
    
    try:
        # Run agent
        exit_status, result, container_id, agent_wrapper = adapter.run_agent(
            instance=instance,
            scaffold=scaffold_config,
            model_name=model,
            output_dir=instance_output,
            tools_dir=tools_dir,
            temperature=temperature,
        )
        
        # Get steps and cost
        if agent_wrapper and hasattr(agent_wrapper, 'agent') and agent_wrapper.agent:
            if hasattr(agent_wrapper.agent, 'model'):
                n_steps = getattr(agent_wrapper.agent.model, 'n_calls', 0)
                cost = getattr(agent_wrapper.agent.model, 'cost', 0.0)
        
        # Save trajectory
        from minisweagent.run.utils.save import save_traj
        traj_path = instance_output / f"{instance_id}.traj.json"
        if agent_wrapper and hasattr(agent_wrapper, 'agent') and agent_wrapper.agent:
            save_traj(
                agent_wrapper.agent, traj_path,
                exit_status=exit_status, result=result, instance_id=instance_id
            )
        
        # Evaluate
        if container_id:
            eval_result = adapter.evaluate(container_id, instance, instance_output)
            passed = eval_result.success
            error = eval_result.error or ""
            result_str = "passed" if passed else f"failed: {error[:200]}"
        else:
            passed = False
            error = "No container ID"
            result_str = "failed: no container"
        
        return DS1000TestResult(
            instance_id=instance_id,
            success=True,
            passed=passed,
            library=instance.category,
            perturbation_type=instance.difficulty,
            error=error,
            result_str=result_str,
            n_steps=n_steps,
            cost=cost,
            duration=time.time() - start_time,
        )
        
    except Exception as e:
        return DS1000TestResult(
            instance_id=instance_id,
            success=False,
            passed=False,
            library=instance.category,
            perturbation_type=instance.difficulty,
            error=str(e),
            result_str=f"error: {type(e).__name__}",
            n_steps=n_steps,
            cost=cost,
            duration=time.time() - start_time,
        )
    finally:
        if agent_wrapper:
            try:
                agent_wrapper.cleanup()
            except Exception:
                pass


def format_result(r: DS1000TestResult) -> str:
    status_icon = "✓" if r.passed else "✗"
    return f"{status_icon} {r.instance_id} [{r.library}]: {r.n_steps} steps, ${r.cost:.4f}, {r.result_str[:50]}"


@app.command()
def main(
    scaffold: Path = typer.Option(..., "--scaffold", "-s", help="Scaffold file path"),
    subset: str = typer.Option("", "--subset", "--dataset", "-d", help="Library filter (Pandas/Numpy/Matplotlib/etc)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="Agent model"),
    workers: int = typer.Option(4, "--workers", "-j", help="Parallel workers"),
    max_instances: int = typer.Option(None, "--max-instances", "-n", help="Max instances"),
    skip_instances: int = typer.Option(0, "--skip", help="Skip first N instances"),
    tools_dir: Path = typer.Option(None, "--tools-dir", help="Tools directory"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Agent temperature"),
    shuffle_file: Path = typer.Option(None, "--shuffle-file", help="Shuffle file path"),
):
    """Run DS-1000 test with specified scaffold."""
    
    from recreate_agent.adapters.ds1000_adapter import DS1000Adapter
    
    suppress_verbose_logging()
    
    console.print("[cyan]DS-1000 Test[/cyan]")
    console.print(f"Scaffold: {scaffold}")
    console.print(f"Subset: {subset or 'All'}")
    console.print(f"Temperature: {temperature}")
    
    if not scaffold.exists():
        console.print(f"[red]Scaffold not found: {scaffold}[/red]")
        raise typer.Exit(1)
    
    # Load scaffold
    full_config = yaml.safe_load(scaffold.read_text())
    
    # Merge memory_template into system_template if present
    if "memory_template" in full_config and full_config["memory_template"]:
        memory_section = full_config["memory_template"].strip()
        system_tpl = full_config.get("system_template", "")
        if memory_section and memory_section not in system_tpl:
            full_config["system_template"] = f"{system_tpl}\n\n{memory_section}"
    
    scaffold_config = {k: v for k, v in full_config.items() if k in AGENT_CONFIG_FIELDS}
    
    # Fix common template variable errors
    if "action_observation_template" in scaffold_config:
        template = scaffold_config["action_observation_template"]
        if "{{ observation }}" in template or "{{observation}}" in template:
            template = template.replace("{{ observation }}", "{{ output.output }}")
            template = template.replace("{{observation}}", "{{output.output}}")
            scaffold_config["action_observation_template"] = template
        if "{{ output }}" in template and "{{ output.output }}" not in template:
            template = template.replace("{{ output }}", "{{ output.output }}")
            scaffold_config["action_observation_template"] = template
    
    # Initialize adapter
    adapter = DS1000Adapter()
    
    # Load dataset
    instances = adapter.load_dataset(subset=subset, shuffle_file=shuffle_file)
    
    if skip_instances > 0:
        console.print(f"[yellow]Skipping first {skip_instances} instances (evolution set)[/yellow]")
        instances = instances[skip_instances:]
    
    if max_instances:
        instances = instances[:max_instances]
    
    if not instances:
        console.print("[red]No matching instances found[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loaded {len(instances)} instances")
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Library distribution
    lib_counts = {}
    for inst in instances:
        lib = inst.category
        lib_counts[lib] = lib_counts.get(lib, 0) + 1
    console.print(f"Library distribution: {lib_counts}")
    
    # Run tests
    results: list[DS1000TestResult] = []
    completed = 0
    
    if workers > 1:
        console.print(f"\n[cyan]Parallel execution ({workers} workers)...[/cyan]")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_instance,
                    inst,
                    adapter,
                    scaffold_config,
                    output,
                    model,
                    tools_dir,
                    temperature,
                ): inst.instance_id
                for inst in instances
            }
            
            for future in as_completed(futures):
                instance_id = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    safe_print(f"[{completed}/{len(instances)}] {format_result(result)}")
                except Exception as e:
                    result = DS1000TestResult(
                        instance_id=instance_id,
                        success=False,
                        passed=False,
                        library="",
                        perturbation_type="",
                        error=str(e),
                        result_str=f"error: {type(e).__name__}",
                    )
                    results.append(result)
                    safe_print(f"[{completed}/{len(instances)}] ⚠ {instance_id}: {e}")
    else:
        console.print(f"\n[cyan]Sequential execution...[/cyan]")
        for i, inst in enumerate(instances, 1):
            result = process_single_instance(
                inst, adapter, scaffold_config, output, model, tools_dir, temperature
            )
            results.append(result)
            console.print(f"[{i}/{len(instances)}] {format_result(result)}")
    
    # Statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    passed = sum(1 for r in results if r.passed)
    total_cost = sum(r.cost for r in results)
    total_steps = sum(r.n_steps for r in results)
    
    # Per-library stats
    lib_results = {}
    for r in results:
        lib = r.library
        if lib not in lib_results:
            lib_results[lib] = {"total": 0, "passed": 0}
        lib_results[lib]["total"] += 1
        if r.passed:
            lib_results[lib]["passed"] += 1
    
    # Summary
    summary = {
        "scaffold": str(scaffold),
        "subset": subset,
        "model": model,
        "temperature": temperature,
        "skip_instances": skip_instances,
        "total": total,
        "successful": successful,
        "passed": passed,
        "pass_rate": passed / total if total > 0 else 0,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "library_results": lib_results,
        "results": [r.to_dict() for r in results],
        "timestamp": datetime.now().isoformat(),
    }
    
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    
    # DS-1000 format answers file
    answers = {}
    for r in results:
        instance_dir = output / r.instance_id
        solution_file = instance_dir / "solution.py"
        if solution_file.exists():
            answers[r.instance_id] = solution_file.read_text()
        else:
            traj_file = instance_dir / f"{r.instance_id}.traj.json"
            if traj_file.exists():
                try:
                    traj = json.loads(traj_file.read_text())
                    answers[r.instance_id] = traj.get("info", {}).get("submission", "")
                except:
                    answers[r.instance_id] = ""
            else:
                answers[r.instance_id] = ""
    
    answers_file = output / "ds1000_answers.json"
    answers_file.write_text(json.dumps(answers, indent=2))
    
    # Print results
    console.print(f"\n[green]✓ DS-1000 Test Complete[/green]")
    console.print(f"\n[bold]Overall Results[/bold]")
    console.print(f"  Total: {total}")
    console.print(f"  Passed: {passed} ({passed/total*100:.1f}%)" if total > 0 else "  Passed: 0")
    console.print(f"  Total Cost: ${total_cost:.4f}")
    console.print(f"  Total Steps: {total_steps}")
    
    console.print(f"\n[bold]Per-Library Statistics[/bold]")
    console.print(f"{'Library':<15} {'Count':>8} {'Passed':>8} {'Rate':>8}")
    console.print("-" * 45)
    for lib, stats in sorted(lib_results.items()):
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        console.print(f"{lib:<15} {stats['total']:>8} {stats['passed']:>8} {rate:>7.1f}%")
    
    console.print(f"\nOutput: {output}")
    console.print(f"Summary: {output}/run_summary.json")
    console.print(f"Answers: {output}/ds1000_answers.json")


if __name__ == "__main__":
    app()
