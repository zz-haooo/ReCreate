"""
ResultCollector - Agent execution result collector.

Collects and analyzes Agent execution results, providing feedback data for ReCreate-Agent.
"""

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class InstanceResult:
    """Single instance execution result."""

    instance_id: str
    exit_status: str
    success: bool
    scaffold_version: int = 0

    n_steps: int = 0
    total_cost: float = 0.0
    duration_seconds: float = 0.0

    problem_summary: str = ""
    repo: str = ""
    problem_type: str = ""

    error_type: str | None = None
    error_message: str = ""
    error: str = ""

    key_issues: list[str] = field(default_factory=list)
    good_decisions: list[str] = field(default_factory=list)
    problematic_decisions: list[str] = field(default_factory=list)

    eval_result: dict = field(default_factory=dict)
    
    domain: str = "swe"
    score: float = 0.0
    details: dict = field(default_factory=dict)
    is_hard_failure: bool = False

    trajectory_path: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class AggregatedStats:
    """Aggregated statistics."""

    total_instances: int = 0
    success_count: int = 0
    success_rate: float = 0.0

    avg_steps: float = 0.0
    avg_cost: float = 0.0

    # Error distribution
    error_distribution: dict[str, int] = field(default_factory=dict)

    # Stats by scaffold version
    stats_by_version: dict[int, dict] = field(default_factory=dict)


class ResultCollector:
    """Result collector."""

    def __init__(self, workspace_path: Path | str):
        self.workspace = Path(workspace_path)
        self.results_dir = self.workspace / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def add_result(
        self,
        trajectory_path: Path | str,
        evaluation_result: dict[str, Any],
        scaffold_version: int = 0,
    ) -> InstanceResult:
        """Extract and save result from trajectory and evaluation."""
        traj_path = Path(trajectory_path)
        traj = json.loads(traj_path.read_text())

        info = traj.get("info", {})
        messages = traj.get("messages", [])

        # Extract basic info
        instance_id = evaluation_result.get("instance_id", traj_path.stem.replace(".traj", ""))

        result = InstanceResult(
            instance_id=instance_id,
            exit_status=info.get("exit_status", "Unknown"),
            success=evaluation_result.get("success", evaluation_result.get("resolved", False)),
            scaffold_version=scaffold_version,
            n_steps=info.get("model_stats", {}).get("api_calls", 0),
            total_cost=info.get("model_stats", {}).get("instance_cost", 0.0),
            problem_summary=self._extract_problem_summary(messages),
            repo=evaluation_result.get("repo", ""),
            error_type=self._classify_error(info, messages),
            trajectory_path=str(traj_path),
            details=evaluation_result.get("details", {}),
            eval_result=evaluation_result,
            score=evaluation_result.get("score", 0.0),
        )

        # Analyze trajectory
        analysis = self._analyze_trajectory(messages)
        result.key_issues = analysis.get("issues", [])
        result.good_decisions = analysis.get("good", [])
        result.problematic_decisions = analysis.get("problematic", [])

        # Save result
        self._save_result(result)

        return result

    def get_result(self, instance_id: str) -> InstanceResult | None:
        """Get single instance result."""
        result_file = self.results_dir / instance_id / "result.json"
        if result_file.exists():
            data = json.loads(result_file.read_text())
            return InstanceResult(**data)
        return None

    def get_all_results(self) -> list[InstanceResult]:
        """Get all results."""
        results = []
        for result_dir in self.results_dir.iterdir():
            if result_dir.is_dir():
                result_file = result_dir / "result.json"
                if result_file.exists():
                    data = json.loads(result_file.read_text())
                    results.append(InstanceResult(**data))
        return sorted(results, key=lambda r: r.created_at, reverse=True)

    def get_aggregated_stats(self, scaffold_version: int | None = None) -> dict:
        """Get aggregated stats (returns dict format for template rendering)."""
        results = self.get_all_results()

        if scaffold_version is not None:
            results = [r for r in results if r.scaffold_version == scaffold_version]

        if not results:
            return {
                "total_instances": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "avg_steps": 0.0,
                "avg_cost": 0.0,
                "error_distribution": {},
                "stats_by_version": {},
            }

        success_count = sum(1 for r in results if r.success)

        # Error distribution
        error_dist = {}
        for r in results:
            if r.error_type:
                error_dist[r.error_type] = error_dist.get(r.error_type, 0) + 1

        # Stats by version
        stats_by_version = {}
        for r in results:
            v = r.scaffold_version
            if v not in stats_by_version:
                stats_by_version[v] = {"total": 0, "success": 0}
            stats_by_version[v]["total"] += 1
            if r.success:
                stats_by_version[v]["success"] += 1

        return {
            "total_instances": len(results),
            "success_count": success_count,
            "success_rate": success_count / len(results) if results else 0.0,
            "avg_steps": sum(r.n_steps for r in results) / len(results),
            "avg_cost": sum(r.total_cost for r in results) / len(results),
            "error_distribution": error_dist,
            "stats_by_version": stats_by_version,
        }

    def get_recent_failures(self, limit: int = 5, scaffold_version: int | None = None) -> list[InstanceResult]:
        """Get recent failure cases."""
        results = self.get_all_results()

        if scaffold_version is not None:
            results = [r for r in results if r.scaffold_version == scaffold_version]

        failures = [r for r in results if not r.success]
        return failures[:limit]

    def get_recent_successes(self, limit: int = 5) -> list[InstanceResult]:
        """Get recent success cases."""
        results = self.get_all_results()
        successes = [r for r in results if r.success]
        return successes[:limit]

    def _save_result(self, result: InstanceResult) -> None:
        """Save result to file."""
        result_dir = self.results_dir / result.instance_id
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / "result.json"
        result_file.write_text(json.dumps(asdict(result), indent=2))

    def _extract_problem_summary(self, messages: list[dict]) -> str:
        """Extract problem summary."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Extract first 300 chars
                if len(content) > 300:
                    return content[:300] + "..."
                return content
        return "Unknown problem"

    def _classify_error(self, info: dict, messages: list[dict]) -> str | None:
        """Classify error type."""
        exit_status = info.get("exit_status", "")

        if exit_status == "Submitted":
            # Submitted but may have failed
            return None

        if exit_status == "LimitsExceeded":
            n_calls = info.get("model_stats", {}).get("api_calls", 0)
            cost = info.get("model_stats", {}).get("instance_cost", 0.0)
            if n_calls > 200:
                return "step_limit_exceeded"
            if cost > 2.5:
                return "cost_limit_exceeded"
            return "limits_exceeded"

        # Analyze error patterns in messages
        format_errors = 0
        timeouts = 0
        command_errors = 0

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if "EXACTLY ONE" in content:
                format_errors += 1
            if "timed out" in content:
                timeouts += 1
            if "Error" in content and "returncode" in content:
                command_errors += 1

        if format_errors > 3:
            return "format_error"
        if timeouts > 2:
            return "timeout"
        if command_errors > 5:
            return "command_errors"

        return exit_status.lower() if exit_status else None

    def _analyze_trajectory(self, messages: list[dict]) -> dict[str, list[str]]:
        """Analyze trajectory, extract key issues and decisions."""
        analysis = {"issues": [], "good": [], "problematic": []}

        step = 0
        prev_output_success = True

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                step += 1

                # Detect good decisions
                if "git blame" in content or "git log" in content:
                    analysis["good"].append(f"Step {step}: Used git history for investigation")
                elif "reproduce" in content.lower() and "python" in content:
                    analysis["good"].append(f"Step {step}: Created reproduction script")

                # Detect problematic decisions
                if not prev_output_success and "sed -i" in content:
                    analysis["problematic"].append(
                        f"Step {step}: Made edits after failed command"
                    )

            elif role == "user":
                # Detect execution result
                if "returncode>0" in content:
                    prev_output_success = True
                elif "returncode" in content:
                    prev_output_success = False

                # Detect issues
                if "EXACTLY ONE" in content:
                    analysis["issues"].append(f"Step {step}: Format error")
                elif "timed out" in content:
                    analysis["issues"].append(f"Step {step}: Command timeout")

        return analysis


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Result Collector CLI")
    parser.add_argument("command", choices=["stats", "failures", "list", "show"])
    parser.add_argument("--workspace", default="./recreate_workspace")
    parser.add_argument("--version", type=int, help="Filter by scaffold version")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--instance-id", help="Instance ID (for show command)")

    args = parser.parse_args()
    collector = ResultCollector(args.workspace)

    if args.command == "stats":
        stats = collector.get_aggregated_stats(args.version)
        print(f"Total instances: {stats['total_instances']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average steps: {stats['avg_steps']:.1f}")
        print(f"Average cost: ${stats['avg_cost']:.4f}")
        if stats["error_distribution"]:
            print("\nError distribution:")
            for error_type, count in stats["error_distribution"].items():
                print(f"  {error_type}: {count}")

    elif args.command == "failures":
        failures = collector.get_recent_failures(args.limit, args.version)
        print(f"Recent failures ({len(failures)}):")
        for f in failures:
            print(f"\n  {f.instance_id}")
            print(f"    Exit: {f.exit_status}, Error: {f.error_type}")
            print(f"    Steps: {f.n_steps}, Cost: ${f.total_cost:.4f}")
            if f.key_issues:
                print(f"    Issues: {', '.join(f.key_issues[:3])}")

    elif args.command == "list":
        results = collector.get_all_results()
        print(f"All results ({len(results)}):")
        for r in results[:20]:
            status = "✓" if r.success else "✗"
            print(f"  {status} {r.instance_id}: {r.exit_status} (v{r.scaffold_version})")

    elif args.command == "show":
        if not args.instance_id:
            print("Error: --instance-id is required")
            return
        result = collector.get_result(args.instance_id)
        if result:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"Result not found: {args.instance_id}")


if __name__ == "__main__":
    main()
