#!/usr/bin/env python3
"""
Parallel Evolution - Batch-based Multi-Domain Architecture

Key difference from serial evolution:
- Serial: Agent₁ → Meta₁ → Agent₂ → Meta₂ → ... (each Meta updates scaffold immediately)
- Parallel: [Agent₁→Meta₁, Agent₂→Meta₂, ...] → Synthesis → new global scaffold

Flow:
1. Create initial global scaffold (global_v000)
2. For each batch of N instances (parallel):
   - All Agents use the same global scaffold
   - Each ReCreate-Agent proposes modifications independently
3. Synthesis ReCreate-Agent reviews all batch modifications
   - Creates unified global_v001 scaffold
4. Repeat until all instances processed
5. Optional final review
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

# Suppress verbose logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("minisweagent.environment").setLevel(logging.WARNING)

try:
    import litellm
    litellm.suppress_debug_info = True
    logging.getLogger("litellm_model").setLevel(logging.ERROR)
    
    if hasattr(litellm, 'model_cost'):
        claude_cost = {
            "max_tokens": 8192,
            "max_input_tokens": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.000018,
            "output_cost_per_token": 0.000018,
        }
        for name in ["claude-4-5-sonnet-20250929", "claude-sonnet-4-5-20250929",
                     "claude-4-5-sonnet", "claude-sonnet-4-5", "claude-opus-4-5-20251101"]:
            litellm.model_cost[name] = claude_cost
        
        litellm.model_cost["gpt-5-mini"] = {
            "max_tokens": 16384,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.0000006,
        }
except Exception:
    pass

import typer

# Add source paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "dacode"))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "swebench"))

from recreate_agent.adapters import get_adapter
from recreate_agent.stats_collector import EvolutionStats

# Import from evolve_utils
from evolve_utils import (
    console,
    chunks,
    BatchResult,
    run_recreate_initial_creation,
    run_single_evolution,
    run_batch_synthesis,
)

app = typer.Typer()


@app.command()
def main(
    domain: str = typer.Option("swe", help="Domain: swe | data_science | ds1000 | math | appworld"),
    subset: str = typer.Option("", help="Subset filter (repo for SWE, app for AppWorld, task type for DA-Code)"),
    dataset_name: str = typer.Option("", help="Dataset name (for AppWorld: train/dev/test_normal/test_challenge)"),
    data_source: str = typer.Option("", help="Data source (for math: math500/aime24/aime25/amc23)"),
    shuffle_file: Path = typer.Option(None, help="Pre-shuffled dataset file"),
    max_instances: int = typer.Option(20, help="Max instances to process"),
    skip_instances: int = typer.Option(0, help="Skip first N instances"),
    batch_size: int = typer.Option(4, help="Parallel batch size"),
    max_retries: int = typer.Option(1, help="Max retries per instance (1=no retry)"),
    n_repeat: int = typer.Option(1, help="Number of evolution rounds on the same dataset"),
    agent_model: str = typer.Option("gpt-5-mini", help="Agent model"),
    recreate_model: str = typer.Option("claude-opus-4-5-20251101", help="Meta-Agent model"),
    agent_temp: float = typer.Option(1.0, help="Agent temperature"),
    recreate_temp: float = typer.Option(1.0, help="Meta-Agent temperature"),
    experiment_name: str = typer.Option("", help="Experiment name prefix"),
    output_dir: str = typer.Option(os.getenv("RECREATE_OUTPUT_DIR", "./output"), help="Output directory"),
    # Ablation parameters (all enabled by default)
    ablation_trajectory: bool = typer.Option(True, help="Enable trajectory analysis capability"),
    ablation_environment: bool = typer.Option(True, help="Enable environment inspection capability"),
    ablation_eval_results: bool = typer.Option(True, help="Enable evaluation results access"),
    ablation_modification_guidance: bool = typer.Option(True, help="Enable modification guidance (thinking framework, decision guidance)"),
):
    """Run parallel evolution with batch-based synthesis."""
    
    # Get adapter with domain-specific parameters
    adapter_kwargs = {}
    if domain == "appworld" and dataset_name:
        adapter_kwargs["dataset_name"] = dataset_name
    if domain == "math" and data_source:
        adapter_kwargs["data_source"] = data_source
    
    adapter = get_adapter(domain, **adapter_kwargs)
    
    # Setup directories (ensure absolute paths for Docker mount compatibility)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subset_name = subset if subset else "all"
    exp_name = f"{experiment_name}_{domain}_{subset_name}_{timestamp}" if experiment_name else f"{domain}_{subset_name}_{timestamp}"
    
    experiment_dir = Path(output_dir).resolve() / exp_name
    workspace = experiment_dir / "workspace"
    runs_dir = experiment_dir / "runs"
    runs_recreate_dir = experiment_dir / "runs_recreate"
    
    for d in [experiment_dir, workspace, runs_dir, runs_recreate_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Build ablation settings
    ablation_settings = {
        "ablation_trajectory": ablation_trajectory,
        "ablation_environment": ablation_environment,
        "ablation_eval_results": ablation_eval_results,
        "ablation_modification_guidance": ablation_modification_guidance,
    }
    
    console.print(f"\n[bold]Parallel Evolution - Batch-based Synthesis[/bold]")
    console.print(f"Domain: [cyan]{adapter.domain_name}[/cyan]")
    console.print(f"Subset: [cyan]{subset_name}[/cyan]")
    console.print(f"Batch Size: [cyan]{batch_size}[/cyan]")
    if n_repeat > 1:
        console.print(f"Repeat Rounds: [cyan]{n_repeat}[/cyan]")
    console.print(f"Experiment: {experiment_dir}\n")
    
    # Display ablation settings
    disabled = [k.replace("ablation_", "") for k, v in ablation_settings.items() if not v]
    if disabled:
        console.print(f"[yellow]Ablation mode: Disabled modules: {', '.join(disabled)}[/yellow]")
    
    # Phase 0: Create initial scaffold
    if not run_recreate_initial_creation(
        workspace, runs_recreate_dir, recreate_model, recreate_temp,
        adapter.domain_name, adapter.domain_description,
        adapter.get_initial_prompt_template(),
        adapter.get_recreate_agent_config(),
        ablation_settings,
    ):
        console.print("[red]Failed to create initial scaffold[/red]")
        raise typer.Exit(1)
    
    # Save config
    config = {
        "type": "parallel_evolution",
        "domain": domain,
        "subset": subset,
        "max_instances": max_instances,
        "skip_instances": skip_instances,
        "batch_size": batch_size,
        "max_retries": max_retries,
        "n_repeat": n_repeat,
        "agent_model": agent_model,
        "recreate_model": recreate_model,
        "agent_temp": agent_temp,
        "recreate_temp": recreate_temp,
        "ablation_settings": ablation_settings,
        "created_at": datetime.now().isoformat(),
    }
    (experiment_dir / "config.json").write_text(json.dumps(config, indent=4))
    
    # Load dataset
    all_instances = adapter.load_dataset(subset=subset, shuffle_file=shuffle_file)
    
    if skip_instances > 0:
        instances = all_instances[skip_instances:]
    else:
        instances = all_instances
    
    if max_instances > 0:
        instances = instances[:max_instances]
    
    console.print(f"Loaded {len(instances)} instances")
    num_batches = (len(instances) + batch_size - 1) // batch_size
    if n_repeat > 1:
        console.print(f"Will process in {num_batches} batches × {n_repeat} rounds = {num_batches * n_repeat} total batches\n")
    else:
        console.print(f"Will process in {num_batches} batches of up to {batch_size} instances\n")
    
    # Evolution log
    evolution_log = {
        "type": "parallel_evolution",
        "n_repeat": n_repeat,
        "global_versions": [{"version": 0, "type": "initial", "created_at": datetime.now().isoformat()}],
        "batches": [],
        "rounds": [],
    }
    
    global_version = 0
    all_results = []
    total_batch_idx = 0  # Global batch counter
    
    # Main loop - repeat rounds
    for repeat_idx in range(n_repeat):
        if n_repeat > 1:
            console.print(f"\n[bold magenta]══════ Round {repeat_idx + 1}/{n_repeat} ══════[/bold magenta]")
        
        round_results = []
        round_start_version = global_version
        
        # Process batches in this round
        for batch_in_round, batch_instances in enumerate(chunks(instances, batch_size)):
            batch_idx = total_batch_idx  # Use global batch number
            
            if n_repeat > 1:
                console.print(f"[bold cyan]━━━ Round {repeat_idx + 1} / Batch {batch_in_round} ({len(batch_instances)} instances) ━━━[/bold cyan]")
            else:
                console.print(f"[bold cyan]━━━ Batch {batch_idx} ({len(batch_instances)} instances) ━━━[/bold cyan]")
            
            current_global = workspace / f"global_v{global_version:03d}"
            batch_dir = workspace / f"batch_{batch_idx:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            console.print(f"Using scaffold: global_v{global_version:03d}")
            
            # ===== Phase 1: Parallel Agent + Meta-Agent =====
            batch_results: dict[str, BatchResult] = {}
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for instance in batch_instances:
                    instance_dir = batch_dir / instance.instance_id
                    future = executor.submit(
                        run_single_evolution,
                        instance=instance,
                        global_scaffold_dir=current_global,
                        output_dir=instance_dir,
                        adapter=adapter,
                        agent_model=agent_model,
                        recreate_model=recreate_model,
                        agent_temp=agent_temp,
                        recreate_temp=recreate_temp,
                        domain=domain,
                        domain_config=adapter.get_recreate_agent_config(),
                        runs_recreate_dir=runs_recreate_dir,
                        ablation_settings=ablation_settings,
                    )
                    futures[instance.instance_id] = future
                
                # Wait for all to complete with timeout handling
                completed = set()
                for instance_id, future in futures.items():
                    try:
                        result = future.result(timeout=600)  # 10 min timeout per instance
                        batch_results[instance_id] = result
                        completed.add(instance_id)
                    except FutureTimeoutError as e:
                        console.print(f"  [red]{instance_id}: Timeout or error - {e}[/red]")
                        # Cancel the future if it's still running
                        future.cancel()
                        result = BatchResult(
                            instance_id=instance_id,
                            success=False, score=0.0,
                            scaffold_changed=False, has_new_tools=False, has_new_memories=False,
                            error=f"Timeout after 600s: {str(e)}",
                            exit_status="Timeout",
                            duration=600.0
                        )
                        batch_results[instance_id] = result
                        completed.add(instance_id)
                    except Exception as e:
                        console.print(f"  [red]{instance_id}: Timeout or error - {e}[/red]")
                        # Cancel the future if it's still running
                        try:
                            future.cancel()
                        except Exception:
                            pass
                        result = BatchResult(
                            instance_id=instance_id,
                            success=False, score=0.0,
                            scaffold_changed=False, has_new_tools=False, has_new_memories=False,
                            error=str(e),
                            exit_status=type(e).__name__,
                            duration=0.0
                        )
                        batch_results[instance_id] = result
                        completed.add(instance_id)
                    
                    # Always add to all_results (both success and failure)
                    all_results.append({
                        "instance_id": instance_id,
                        "batch": batch_idx,
                        "round": repeat_idx + 1,
                        "success": result.success,
                        "score": result.score,
                        "scaffold_changed": result.scaffold_changed,
                    })
                
                # Ensure all futures are handled (cancel any remaining)
                for instance_id, future in futures.items():
                    if instance_id not in completed:
                        try:
                            future.cancel()
                        except Exception:
                            pass
            
            # Save batch info
            batch_info = {
                "batch_idx": batch_idx,
                "round": repeat_idx + 1,
                "base_version": global_version,
                "instances": [
                    {
                        "id": iid,
                        "success": r.success,
                        "score": r.score,
                        "scaffold_changed": r.scaffold_changed,
                        "duration": r.duration,
                    }
                    for iid, r in batch_results.items()
                ],
            }
            (batch_dir / "batch_info.json").write_text(json.dumps(batch_info, indent=2))
            
            # ===== Phase 2: Batch Synthesis =====
            new_version = global_version + 1
            new_global = workspace / f"global_v{new_version:03d}"
            
            modified = run_batch_synthesis(
                workspace=workspace,
                batch_dir=batch_dir,
                batch_results=batch_results,
                current_global_dir=current_global,
                new_global_dir=new_global,
                batch_idx=batch_idx,
                recreate_model=recreate_model,
                recreate_temp=recreate_temp,
                domain=domain,
                domain_config=adapter.get_recreate_agent_config(),
                runs_recreate_dir=runs_recreate_dir,
                ablation_settings=ablation_settings,
                EvolutionStats=EvolutionStats,
            )
            
            # Update current symlink
            current_link = workspace / "current"
            if current_link.exists() or current_link.is_symlink():
                current_link.unlink()
            current_link.symlink_to(f"global_v{new_version:03d}")
            
            # Update log
            evolution_log["global_versions"].append({
                "version": new_version,
                "type": "batch_synthesis",
                "source_batch": batch_idx,
                "round": repeat_idx + 1,
                "modified": modified,
                "created_at": datetime.now().isoformat(),
            })
            evolution_log["batches"].append(batch_info)
            
            global_version = new_version
        
            # Summary - domain-specific display
            batch_changed = sum(1 for r in batch_results.values() if r.scaffold_changed)
            if domain == "data_science":
                # DA-Code: show average score
                batch_avg_score = sum(r.score for r in batch_results.values()) / max(1, len(batch_results))
                console.print(f"  Batch {batch_idx}: Avg Score = {batch_avg_score:.2f}, {batch_changed} proposed changes")
            else:
                batch_success = sum(1 for r in batch_results.values() if r.success)
                console.print(f"  Batch {batch_idx}: {batch_success}/{len(batch_results)} success, {batch_changed} proposed changes")
            console.print("")
            
            total_batch_idx += 1
            
            # Record batch results
            for iid, r in batch_results.items():
                round_results.append({
                    "instance_id": iid,
                    "batch": batch_idx,
                    "success": r.success,
                    "score": r.score,
                })
        
        # Round summary
        if n_repeat > 1:
            round_success = sum(1 for r in round_results if r["success"])
            round_avg_score = sum(r["score"] for r in round_results) / max(1, len(round_results))
            console.print(f"[magenta]Round {repeat_idx + 1} complete: {round_success}/{len(round_results)} success, Avg Score = {round_avg_score:.2f}[/magenta]")
            
            evolution_log["rounds"].append({
                "round": repeat_idx + 1,
                "start_version": round_start_version,
                "end_version": global_version,
                "instances": len(round_results),
                "success": round_success,
                "avg_score": round_avg_score,
                "completed_at": datetime.now().isoformat(),
            })
    
    # Save evolution log
    (workspace / "evolution_log.json").write_text(json.dumps(evolution_log, indent=2, ensure_ascii=False))
    
    # Summary
    total_success = sum(1 for r in all_results if r["success"])
    avg_score = sum(r["score"] for r in all_results) / max(1, len(all_results))
    
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Domain: {domain}")
    if n_repeat > 1:
        console.print(f"Repeat Rounds: {n_repeat}")
        console.print(f"Total Instances: {len(all_results)} ({len(instances)} × {n_repeat} rounds)")
    else:
        console.print(f"Total Instances: {len(all_results)}")
    if domain == "data_science":
        # DA-Code: primary metric is average score
        console.print(f"[bold green]Average Score: {avg_score:.2f}[/bold green]")
        console.print(f"Valid Results: {total_success} ({total_success/max(1, len(all_results))*100:.1f}%)")
    else:
        console.print(f"Success: {total_success} ({total_success/max(1, len(all_results))*100:.1f}%)")
        if domain not in ("swe",):  # Non-SWE domains also show average score
            console.print(f"Average Score: {avg_score:.2f}")
    console.print(f"Global Versions: {global_version + 1} (v000 to v{global_version:03d})")
    console.print(f"Batches: {total_batch_idx}")
    
    # Save results
    (experiment_dir / "results.json").write_text(json.dumps({
        "domain": domain,
        "subset": subset,
        "n_repeat": n_repeat,
        "instances_per_round": len(instances),
        "total": len(all_results),
        "success": total_success,
        "avg_score": avg_score,
        "global_versions": global_version + 1,
        "batches": total_batch_idx,
        "results": all_results,
    }, indent=2))
    
    console.print(f"\nResults saved to {experiment_dir}")


if __name__ == "__main__":
    app()
