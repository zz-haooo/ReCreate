"""
ReCreateAgent - A ReCreate-Agent that creates and evolves Agents.

Core design: ReCreate-Agent is also an Agent, reusing mini-swe-agent's DefaultAgent architecture.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Template, StrictUndefined

from minisweagent.agents.default import DefaultAgent, AgentConfig
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj

from recreate_agent.scaffold import ScaffoldManager
from recreate_agent.result_collector import ResultCollector


RUNS_RECREATE_DIR = Path(os.getenv("RECREATE_RUNS_DIR", "./runs_recreate"))


@dataclass
class ReCreateAgentConfig(AgentConfig):
    """ReCreate-Agent configuration."""
    workspace_path: str = ""


class ReCreateAgent(DefaultAgent):
    """ReCreate-Agent implementation."""
    
    def __init__(
        self,
        model_name: str = "claude-opus-4-5-20251101",
        workspace_path: Path | str = "./recreate_workspace",
        run_name: str | None = None,
        runs_recreate_dir: Path | str | None = None,
        domain_config: dict | None = None,
        **model_kwargs,
    ):
        self.workspace = Path(workspace_path)
        self.model_name = model_name
        self.domain_config = domain_config or self._default_domain_config()
        
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"recreate_{model_name.replace('/', '_')}_{timestamp}"
        self.run_name = run_name
        
        if runs_recreate_dir is not None:
            self.traj_dir = Path(runs_recreate_dir) / run_name
        else:
            self.traj_dir = RUNS_RECREATE_DIR / run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        
        prompts_dir = self.workspace / "prompts"
        system_template_raw = (prompts_dir / "meta_system.jinja2").read_text()
        instance_template = (prompts_dir / "meta_instance.jinja2").read_text()
        
        flat_config = self._flatten_domain_config(self.domain_config)
        system_template = Template(system_template_raw).render(**flat_config)
        
        config = ReCreateAgentConfig(
            system_template=system_template,
            instance_template=instance_template,
            action_observation_template="<output>\n{{output.output}}\n</output>",
            format_error_template="Please provide EXACTLY ONE bash command in triple backticks.",
            step_limit=100,
            cost_limit=15.0,
            workspace_path=str(self.workspace),
        )
        
        actual_kwargs = model_kwargs.get("model_kwargs", model_kwargs) if isinstance(model_kwargs, dict) else {}
        default_model_kwargs = {
            "custom_llm_provider": "openai",
            "temperature": 1.0,
        }
        api_base = os.getenv("LLM_API_BASE")
        if api_base:
            default_model_kwargs["api_base"] = api_base
        default_model_kwargs.update(actual_kwargs)
        model = get_model(model_name, {"model_kwargs": default_model_kwargs})
        
        current_dir = self.workspace / "current"
        if not current_dir.exists():
            working_dir = self.workspace / "working"
            working_dir.mkdir(exist_ok=True)
            current_dir.symlink_to("working")
        
        env = LocalEnvironment(
            cwd=str(self.workspace),
            timeout=120,
            env={
                "TOOLS_DIR": str(self.workspace / "agent_tools"),
                "WORKSPACE": str(self.workspace),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
        )
        
        super().__init__(model, env, config_class=lambda **kw: config)
        
        self.scaffold_manager = ScaffoldManager(self.workspace)
        self.result_collector = ResultCollector(self.workspace)
    
    def evolve(
        self, 
        evolution_id: str = "evolve_001",
        progress_callback=None,
        quiet: bool = True,
    ) -> tuple[str, str]:
        """Execute one evolution cycle and save trajectory.
        
        Args:
            evolution_id: Evolution identifier
            progress_callback: Optional callback receiving (step, cost)
            quiet: If True, suppress progress output (default for parallel mode)
        """
        from minisweagent.agents.default import NonTerminatingException, TerminatingException
        from tenacity import RetryError
        import litellm
        
        stats = self.result_collector.get_aggregated_stats()
        recent_failures = self.result_collector.get_recent_failures(limit=5)
        recent_successes = self.result_collector.get_recent_successes(limit=2)
        
        task_context = self._build_task_context(stats, recent_failures, recent_successes)
        
        self.messages = []
        self.add_message("system", self.config.system_template)
        self.add_message("user", task_context)
        
        exit_status = "Running"
        step_count = 0
        
        while True:
            step_count += 1
            if progress_callback:
                progress_callback(step_count, self.model.cost)
            elif not quiet:
                print(f"\r  ReCreate-Agent executing {step_count} steps (${self.model.cost:.2f})...", end="", flush=True)
            
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                exit_status = type(e).__name__
                break
            except (RetryError, litellm.exceptions.BadRequestError) as e:
                exit_status = f"APIError: {str(e)[:100]}"
                break
        
        if not progress_callback and not quiet:
            status_icon = "✓" if exit_status in ["Submitted", "Resolved"] else "✗"
            print(f"\r  ReCreate-Agent: {step_count} steps | ${self.model.cost:.4f} | {status_icon} {exit_status}")
        
        traj_path = self.traj_dir / evolution_id / f"{evolution_id}.traj.json"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        
        traj_data = {
            "instance_id": evolution_id,
            "model": self.model_name,
            "exit_status": exit_status,
            "n_steps": step_count,
            "cost": self.model.cost,
            "messages": self.messages,
            "workspace": str(self.workspace),
            "timestamp": datetime.now().isoformat(),
        }
        traj_path.write_text(json.dumps(traj_data, indent=2, ensure_ascii=False))
        
        return exit_status, str(traj_path)
    
    def _build_task_context(self, stats, recent_failures: list, recent_successes: list) -> str:
        """Build task context."""
        flat_config = self._flatten_domain_config(self.domain_config)
        
        context = {
            "stats": self._format_stats(stats),
            "recent_failures": [self._format_result(r) for r in recent_failures],
            "recent_successes": [self._format_result(r) for r in recent_successes],
            **flat_config,
        }
        template = Template(self.config.instance_template, undefined=StrictUndefined)
        return template.render(**context)
    
    def _format_stats(self, stats) -> dict:
        """Format statistics data."""
        if isinstance(stats, dict):
            return stats
        if hasattr(stats, '__dict__'):
            return {
                "total_instances": getattr(stats, 'total_instances', 0),
                "successful": getattr(stats, 'success_count', 0),
                "failed": getattr(stats, 'total_instances', 0) - getattr(stats, 'success_count', 0),
                "success_rate": getattr(stats, 'success_rate', 0.0),
                "avg_steps": getattr(stats, 'avg_steps', 0.0),
                "avg_cost": getattr(stats, 'avg_cost', 0.0),
                "error_distribution": getattr(stats, 'error_distribution', {}),
            }
        return {}
    
    def _format_result(self, result) -> dict:
        """Format result object."""
        from recreate_agent.adapters.base import format_eval_result, DomainPromptConfig
        
        eval_data = {}
        if hasattr(result, 'eval_result') and result.eval_result:
            eval_data = result.eval_result
        elif hasattr(result, 'details') and result.details:
            eval_data = result.details
        
        domain = getattr(result, 'domain', 'swe')
        score = getattr(result, 'score', 0.0)
        formatted_result = self._build_formatted_result(domain, eval_data, score)
        
        return {
            "instance_id": result.instance_id,
            "exit_status": result.exit_status,
            "success": result.success,
            "error_type": result.error_type,
            "n_steps": result.n_steps,
            "total_cost": result.total_cost,
            "problem_summary": result.problem_summary[:500] if result.problem_summary else "",
            "key_issues": result.key_issues[:5] if result.key_issues else [],
            "good_decisions": result.good_decisions[:3] if result.good_decisions else [],
            "problematic_decisions": result.problematic_decisions[:3] if result.problematic_decisions else [],
            "eval_result": eval_data,
            "domain": domain,
            "score": score,
            "error": getattr(result, 'error', ''),
            "formatted_result": formatted_result,
            "extra_files": formatted_result.get("extra_files", []),
        }
    
    def _build_formatted_result(self, domain: str, eval_data: dict, score: float) -> dict:
        """Build unified formatted evaluation result based on domain."""
        result = {
            "primary_metric_name": "Score",
            "primary_metric_value": f"{score:.2f}",
            "secondary_metrics": [],
            "failure_list_name": "Errors",
            "failure_items": [],
            "extra_files": [("eval_result.json", "Evaluation details")],
        }
        
        if domain == "swe":
            passed = eval_data.get("tests_passed", 0)
            failed = eval_data.get("tests_failed", 0)
            result["primary_metric_name"] = "Tests"
            result["primary_metric_value"] = f"{passed}P/{failed}F"
            result["failure_list_name"] = "Still Failing Tests"
            result["failure_items"] = eval_data.get("fail_to_pass_failure", [])[:10]
            result["extra_files"] = [
                ("test_output.txt", "Raw test output with stack traces"),
                ("expected_tests.txt", "FAIL_TO_PASS and PASS_TO_PASS lists"),
                ("test_patch.txt", "Tests added by benchmark"),
            ]
            if eval_data.get("pass_to_pass_failure"):
                result["secondary_metrics"].append(
                    ("Regression", f"{len(eval_data.get('pass_to_pass_failure', []))} tests broke")
                )
        
        elif domain == "data_science":
            result["primary_metric_name"] = "Score"
            result["primary_metric_value"] = f"{score:.2f}"
            if eval_data.get("metric_func"):
                result["secondary_metrics"].append(("Metric", eval_data["metric_func"]))
        
        elif domain == "ds1000":
            result["primary_metric_name"] = "Result"
            result["primary_metric_value"] = "✓ Passed" if score >= 0.99 else "✗ Failed"
            if eval_data.get("error"):
                result["failure_items"] = [eval_data["error"]]
        
        elif domain == "math":
            result["primary_metric_name"] = "Correct"
            result["primary_metric_value"] = "Yes" if score >= 0.99 else "No"
            if eval_data.get("expected_answer"):
                result["secondary_metrics"].append(("Expected", eval_data["expected_answer"]))
            if eval_data.get("extracted_answer"):
                result["secondary_metrics"].append(("Agent Answer", eval_data["extracted_answer"]))
        
        elif domain == "appworld":
            pass_count = eval_data.get("pass_count", 0)
            fail_count = eval_data.get("fail_count", 0)
            total = pass_count + fail_count
            result["primary_metric_name"] = "Assertions"
            result["primary_metric_value"] = f"{pass_count}/{total}"
            result["failure_list_name"] = "Failed Requirements"
            failures = eval_data.get("failures", [])
            result["failure_items"] = [
                f.get("requirement", str(f)) if isinstance(f, dict) else str(f)
                for f in failures[:5]
            ]
            result["extra_files"] = [("evaluation/report.md", "Assertion results")]
        
        return result
    
    @staticmethod
    def _default_domain_config() -> dict:
        """Default domain configuration (SWE-bench)."""
        return {
            "domain": "swe",
            "domain_description": "Software engineering tasks from GitHub issues",
            "environment": {
                "codebase_path": "/testbed/",
                "working_dir": "/testbed/",
                "timeout_seconds": 60,
                "packages": [],
            },
            "evaluation": {
                "error_file": "test_output.txt",
                "error_file_desc": "raw test output with error messages, stack traces, and assertion failures",
                "success_criteria": "Tests pass",
            },
            "meta_tools": {
                "tools_path": "/workspace",
                "inspect_example": "cat /testbed/path/to/file.py",
            },
        }
    
    @staticmethod
    def _flatten_domain_config(config: dict) -> dict:
        """
        Flatten nested domain config for Jinja2 template rendering.
        
        Supports two sources:
        1. Old format: nested dict from get_meta_agent_config()
        2. New format: flat DomainPromptConfig fields
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
            "ablation_trajectory": config.get("ablation_trajectory", True),
            "ablation_environment": config.get("ablation_environment", True),
            "ablation_eval_results": config.get("ablation_eval_results", True),
            "ablation_modification_guidance": config.get("ablation_modification_guidance", True),
        }
        return result


def create_recreate_agent(
    workspace_path: Path | str = "./recreate_workspace",
    model_name: str | None = None,
    run_name: str | None = None,
) -> ReCreateAgent:
    import os
    if model_name is None:
        model_name = os.getenv("MSWEA_MODEL_NAME", "claude-opus-4-5-20251101")
    
    return ReCreateAgent(
        model_name=model_name,
        workspace_path=workspace_path,
        run_name=run_name,
    )
