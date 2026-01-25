#!/usr/bin/env python3
"""
DA-Code Test Runner

Evaluate evolved scaffolds on DA-Code test set.

Usage:
    python run_dacode_test.py --scaffold path/to/scaffold.yaml --subset sa --output results/
"""

import json
import logging
import math
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


@dataclass
class DACodeTestResult:
    instance_id: str
    success: bool
    score: float
    error: str = ""
    metric_func: str = ""
    n_steps: int = 0
    cost: float = 0.0
    duration: float = 0.0


def run_single_test(
    instance,
    scaffold: dict,
    model: str,
    output_dir: Path,
    tools_dir: Path | None,
    temperature: float,
) -> DACodeTestResult:
    """Run single DA-Code test instance."""
    from recreate_agent.adapters.dacode_adapter import DACodeAdapter
    
    adapter = DACodeAdapter()
    instance_id = instance.instance_id
    instance_output_dir = output_dir / instance_id
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    agent_wrapper = None
    n_steps = 0
    cost = 0.0
    
    try:
        exit_status, result, container_id, agent_wrapper = adapter.run_agent(
            instance=instance,
            scaffold=scaffold,
            model_name=model,
            output_dir=instance_output_dir,
            tools_dir=tools_dir,
            temperature=temperature,
        )
        
        # Get agent stats and save trajectory
        if hasattr(agent_wrapper, 'agent') and agent_wrapper.agent:
            agent = agent_wrapper.agent
            if hasattr(agent, 'model'):
                n_steps = agent.model.n_calls
                cost = agent.model.cost
            if hasattr(agent, 'messages'):
                traj_file = instance_output_dir / "trajectory.json"
                traj_file.write_text(json.dumps({"messages": agent.messages}, indent=2, ensure_ascii=False))
        
        # Evaluate
        eval_result = adapter.evaluate(container_id, instance, instance_output_dir)
        
        duration = time.time() - start_time
        
        return DACodeTestResult(
            instance_id=instance_id,
            success=True,
            score=eval_result.score,
            error=eval_result.error,
            metric_func=eval_result.eval_result.get("metric_func", ""),
            n_steps=n_steps,
            cost=cost,
            duration=duration,
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return DACodeTestResult(
            instance_id=instance_id,
            success=False,
            score=0.0,
            error=str(e),
            duration=duration,
        )
    finally:
        if agent_wrapper and hasattr(agent_wrapper, 'cleanup'):
            try:
                agent_wrapper.cleanup()
            except Exception:
                pass


@app.command()
def main(
    scaffold: Path = typer.Option(..., "--scaffold", "-s", help="Scaffold file path"),
    subset: str = typer.Option("all", "--subset", help="Task type (sa, ml, visual, di, dm, all)"),
    skip_instances: int = typer.Option(0, "--skip-instances", help="Skip first N instances"),
    max_instances: int = typer.Option(10, "--max-instances", "-n", help="Max test instances"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="Model name"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature"),
    parallel: int = typer.Option(4, "--parallel", "-p", help="Parallelism"),
    tools_dir: Path = typer.Option(None, "--tools-dir", help="Tools directory"),
    shuffle_file: Path = typer.Option(None, "--shuffle-file", help="Shuffle file path"),
):
    """Run DA-Code test evaluation."""
    suppress_verbose_logging()
    
    from recreate_agent.adapters.dacode_adapter import DACodeAdapter
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./output/dacode_test_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if not scaffold.exists():
        console.print(f"[red]Error: Scaffold not found: {scaffold}[/red]")
        raise typer.Exit(1)
    
    scaffold_data = yaml.safe_load(scaffold.read_text())
    
    # Merge memory_template into system_template if present
    if "memory_template" in scaffold_data and scaffold_data["memory_template"]:
        memory_section = scaffold_data["memory_template"].strip()
        system_tpl = scaffold_data.get("system_template", "")
        if memory_section and memory_section not in system_tpl:
            scaffold_data["system_template"] = f"{system_tpl}\n\n{memory_section}"
    
    console.print(f"[cyan]Scaffold: {scaffold}[/cyan]")
    
    # Load dataset
    adapter = DACodeAdapter()
    instances = adapter.load_dataset(
        subset=subset,
        shuffle_file=shuffle_file,
    )
    
    if skip_instances > 0:
        instances = instances[skip_instances:]
    if max_instances > 0:
        instances = instances[:max_instances]
    
    console.print(f"[cyan]Instances: {len(instances)}[/cyan]")
    console.print(f"[cyan]Parallelism: {parallel}[/cyan]")
    console.print(f"[cyan]Output: {output_dir}[/cyan]")
    console.print()
    
    # Parallel execution
    results: list[DACodeTestResult] = []
    
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                run_single_test,
                instance,
                scaffold_data,
                model,
                output_dir,
                tools_dir,
                temperature,
            ): instance
            for instance in instances
        }
        
        for future in as_completed(futures):
            instance = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                safe_print(f" ● {result.instance_id}: "
                          f"Score={result.score:.2f} "
                          f"[{result.n_steps} steps, ${result.cost:.2f}]")
                
            except Exception as e:
                safe_print(f" ✗ {instance.instance_id}: Error - {e}")
    
    # Statistics (filter NaN values)
    total = len(results)
    valid_scores = [r.score for r in results if not math.isnan(r.score)]
    avg_score = sum(valid_scores) / max(1, len(valid_scores)) if valid_scores else 0.0
    total_cost = sum(r.cost for r in results)
    total_steps = sum(r.n_steps for r in results)
    
    console.print()
    console.print("=" * 60)
    console.print(f"[bold]Test Complete[/bold]")
    console.print(f"Average Score: {avg_score:.4f}")
    console.print(f"Total Instances: {total}")
    console.print(f"Total Steps: {total_steps}")
    console.print(f"Total Cost: ${total_cost:.2f}")
    console.print(f"Output: {output_dir}")
    console.print("=" * 60)
    
    # Save results
    summary = {
        "scaffold": str(scaffold),
        "subset": subset,
        "total": total,
        "avg_score": avg_score,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "results": [
            {
                "instance_id": r.instance_id,
                "score": r.score,
                "metric_func": r.metric_func,
                "error": r.error,
                "n_steps": r.n_steps,
                "cost": r.cost,
                "duration": r.duration,
            }
            for r in results
        ],
    }
    
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"[green]Results saved: {output_dir / 'summary.json'}[/green]")


if __name__ == "__main__":
    app()
