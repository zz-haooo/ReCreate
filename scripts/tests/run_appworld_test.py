#!/usr/bin/env python3
"""
AppWorld Test Runner

Evaluate evolved scaffolds on AppWorld tasks.
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from recreate_agent.adapters import UnifiedInstance
from recreate_agent.adapters.appworld_adapter import AppWorldAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="AppWorld Test Runner")
    
    parser.add_argument("--scaffold-path", type=str, required=True,
                        help="Scaffold YAML file path")
    parser.add_argument("--dataset", type=str, default="dev",
                        help="Dataset: train, dev, test_normal, test_challenge")
    parser.add_argument("--shuffle-file", type=str, default=None,
                        help="Shuffle file path")
    parser.add_argument("--skip-instances", type=int, default=0,
                        help="Skip first N instances (evolution set)")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Max test instances")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers")
    parser.add_argument("--subset", type=str, default="",
                        help="Subset filter (app name or difficulty_N)")
    parser.add_argument("--tools-dir", type=str, default=None,
                        help="Tools/memory directory (contains agent_tools/ and agent_memory/)")
    
    return parser.parse_args()


def load_scaffold(scaffold_path: str) -> dict:
    """Load scaffold configuration."""
    with open(scaffold_path, 'r') as f:
        scaffold = yaml.safe_load(f)
    
    # Merge memory_template into system_template if present
    if "memory_template" in scaffold and scaffold["memory_template"]:
        memory_section = scaffold["memory_template"].strip()
        system_tpl = scaffold.get("system_template", "")
        if memory_section and memory_section not in system_tpl:
            scaffold["system_template"] = f"{system_tpl}\n\n{memory_section}"
    
    return scaffold


def run_single_instance(
    instance: UnifiedInstance,
    scaffold: dict,
    model_name: str,
    temperature: float,
    output_dir: Path,
    tools_dir: Path | None = None,
) -> dict:
    """Run single AppWorld instance."""
    
    instance_id = instance.instance_id
    instance_output_dir = output_dir / instance_id
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "instance_id": instance_id,
        "success": False,
        "score": 0.0,
        "error": "",
        "duration": 0.0,
    }
    
    start_time = datetime.now()
    
    try:
        adapter = AppWorldAdapter()
        
        # Run agent
        exit_status, agent_result, container_id, env_wrapper = adapter.run_agent(
            instance=instance,
            scaffold=scaffold,
            model_name=model_name,
            output_dir=instance_output_dir,
            tools_dir=tools_dir,
            temperature=temperature,
        )
        
        # Save trajectory
        if hasattr(env_wrapper, 'agent') and hasattr(env_wrapper.agent, 'messages'):
            traj_file = instance_output_dir / f"{instance_id}.traj.json"
            traj_data = {
                "instance_id": instance_id,
                "exit_status": exit_status,
                "messages": env_wrapper.agent.messages,
                "info": {
                    "model_stats": {
                        "api_calls": getattr(env_wrapper.agent.model, 'n_calls', 0),
                        "instance_cost": getattr(env_wrapper.agent.model, 'cost', 0.0),
                    },
                    "exit_status": exit_status,
                },
            }
            traj_file.write_text(json.dumps(traj_data, indent=2, ensure_ascii=False))
        
        # Evaluate
        eval_result = adapter.evaluate(
            container_id=container_id,
            instance=instance,
            output_dir=instance_output_dir,
        )
        
        result["success"] = eval_result.success
        result["score"] = eval_result.score
        result["exit_status"] = exit_status
        result["eval_result"] = eval_result.eval_result
        
        # Save evaluation report
        report_file = instance_output_dir / "eval_report.md"
        report_file.write_text(eval_result.formatted_output)
        
        env_wrapper.cleanup()
        
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    result["duration"] = (datetime.now() - start_time).total_seconds()
    
    # Save result
    result_file = instance_output_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


def main():
    args = parse_args()
    
    scaffold = load_scaffold(args.scaffold_path)
    print(f"Loaded scaffold: {args.scaffold_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    adapter = AppWorldAdapter(dataset_name=args.dataset)
    
    shuffle_file = Path(args.shuffle_file) if args.shuffle_file else None
    instances = adapter.load_dataset(
        subset=args.subset,
        shuffle_file=shuffle_file,
    )
    
    print(f"Loaded {len(instances)} instances from {args.dataset}")
    
    # Apply skip and max limits
    if args.skip_instances > 0:
        instances = instances[args.skip_instances:]
        print(f"Skipped first {args.skip_instances} instances (evolution set)")
    
    if args.max_instances:
        remaining = args.max_instances - args.skip_instances
        if remaining > 0:
            instances = instances[:remaining]
        else:
            instances = []
    
    print(f"Testing {len(instances)} instances")
    
    if not instances:
        print("No instances to test!")
        return
    
    tools_dir = Path(args.tools_dir) if args.tools_dir else None
    
    # Run tests
    results = []
    success_count = 0
    
    if args.workers <= 1:
        # Sequential
        for i, instance in enumerate(instances):
            print(f"\n[{i+1}/{len(instances)}] Testing {instance.instance_id}...")
            result = run_single_instance(
                instance=instance,
                scaffold=scaffold,
                model_name=args.model,
                temperature=args.temperature,
                output_dir=output_dir,
                tools_dir=tools_dir,
            )
            results.append(result)
            
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {instance.instance_id}: score={result['score']:.2f}")
            
            if result["success"]:
                success_count += 1
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_single_instance,
                    instance=inst,
                    scaffold=scaffold,
                    model_name=args.model,
                    temperature=args.temperature,
                    output_dir=output_dir,
                    tools_dir=tools_dir,
                ): inst
                for inst in instances
            }
            
            for i, future in enumerate(as_completed(futures)):
                instance = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "instance_id": instance.instance_id,
                        "success": False,
                        "score": 0.0,
                        "error": str(e),
                    }
                
                results.append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"[{i+1}/{len(instances)}] {status} {instance.instance_id}: score={result.get('score', 0):.2f}")
                
                if result["success"]:
                    success_count += 1
    
    # Summary
    total = len(results)
    avg_score = sum(r.get("score", 0) for r in results) / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"AppWorld Test Complete")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Success: {success_count}")
    print(f"Success Rate: {success_count/total*100:.1f}%")
    print(f"Avg Score: {avg_score*100:.1f}%")
    
    # Save summary
    summary = {
        "scaffold_path": args.scaffold_path,
        "dataset": args.dataset,
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total > 0 else 0,
        "avg_score": avg_score,
        "results": results,
    }
    
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()
