"""
AgentRunner - Agent Executor

Responsibilities:
1. Load scaffold configuration
2. Run mini-swe-agent
3. Collect execution results and trajectories
"""

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

from recreate_agent.scaffold import ScaffoldManager, ScaffoldConfig
from recreate_agent.result_collector import ResultCollector


class AgentRunner:
    """Agent Executor.
    
    Uses ReCreate-Agent maintained scaffold configuration to run Agent.
    """
    
    def __init__(
        self,
        workspace_path: Path | str,
        model_name: str | None = None,
        use_docker: bool = True,
    ):
        self.workspace = Path(workspace_path)
        self.model_name = model_name
        self.use_docker = use_docker
        
        self.scaffold_manager = ScaffoldManager(self.workspace)
        self.result_collector = ResultCollector(self.workspace)
        
        self.results_dir = self.workspace / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_instance(
        self,
        instance: dict,
        scaffold: ScaffoldConfig | None = None,
        timeout: int = 60,
    ) -> dict:
        """Run single instance.
        
        Args:
            instance: Dataset instance (with instance_id, problem_statement, etc.)
            scaffold: Scaffold config, defaults to current version
            timeout: Command execution timeout
        
        Returns:
            Execution result dictionary
        """
        instance_id = instance.get("instance_id", f"instance_{int(time.time())}")
        task = instance.get("problem_statement", instance.get("task", ""))
        
        logger.info(f"Running instance: {instance_id}")
        
        if scaffold is None:
            scaffold = self.scaffold_manager.get_current()
        
        agent_config = self._build_agent_config(scaffold)
        instance_dir = self.results_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            agent, exit_status, result = self._run_agent(
                task=task,
                instance=instance,
                agent_config=agent_config,
                timeout=timeout,
            )
            
            traj_path = instance_dir / f"{instance_id}.traj.json"
            save_traj(
                agent,
                traj_path,
                exit_status=exit_status,
                result=result,
                instance_id=instance_id,
            )
            
            evaluation = self._evaluate_result(exit_status, result, instance)
            
            self.result_collector.add_result(
                trajectory_path=traj_path,
                evaluation_result=evaluation,
                scaffold_version=scaffold.version,
            )
            
            return {
                "instance_id": instance_id,
                "exit_status": exit_status,
                "success": evaluation.get("resolved", False),
                "trajectory_path": str(traj_path),
                "scaffold_version": scaffold.version,
            }
            
        except Exception as e:
            logger.error(f"Error running instance {instance_id}: {e}")
            
            error_file = instance_dir / "error.txt"
            error_file.write_text(str(e))
            
            return {
                "instance_id": instance_id,
                "exit_status": "Error",
                "success": False,
                "error": str(e),
                "scaffold_version": scaffold.version,
            }
    
    def run_batch(
        self,
        instances: list[dict],
        scaffold: ScaffoldConfig | None = None,
        max_instances: int | None = None,
    ) -> list[dict]:
        """Run batch of instances.
        
        Args:
            instances: Instance list
            scaffold: Scaffold config
            max_instances: Maximum instances to run
        
        Returns:
            Result list
        """
        if max_instances is not None:
            instances = instances[:max_instances]
        
        results = []
        for i, instance in enumerate(instances):
            logger.info(f"Processing {i+1}/{len(instances)}: {instance.get('instance_id', 'unknown')}")
            result = self.run_instance(instance, scaffold)
            results.append(result)
        
        return results
    
    def _build_agent_config(self, scaffold: ScaffoldConfig) -> dict:
        """Build agent config from scaffold."""
        return {
            "system_template": scaffold.system_template,
            "instance_template": scaffold.instance_template,
            "action_observation_template": scaffold.action_observation_template,
            "format_error_template": scaffold.format_error_template,
            "timeout_template": scaffold.timeout_template,
            "step_limit": scaffold.step_limit,
            "cost_limit": scaffold.cost_limit,
        }
    
    def _run_agent(
        self,
        task: str,
        instance: dict,
        agent_config: dict,
        timeout: int,
    ) -> tuple[DefaultAgent, str, str]:
        """Run agent."""
        import os
        
        model_name = self.model_name or os.getenv("MSWEA_MODEL_NAME", "gpt-5-mini")
        model = get_model({
            "model_name": model_name,
            "model_kwargs": {"temperature": 0.0},
        })
        
        if self.use_docker:
            image_name = self._get_docker_image(instance)
            env = DockerEnvironment(
                image=image_name,
                cwd="/testbed",
                timeout=timeout,
                env={
                    "PAGER": "cat",
                    "TOOLS_DIR": str(self.workspace / "agent_tools"),
                },
            )
        else:
            env = LocalEnvironment(
                cwd=str(self.workspace),
                timeout=timeout,
            )
        
        agent = DefaultAgent(model, env, **agent_config)
        
        try:
            exit_status, result = agent.run(task)
        finally:
            if self.use_docker and hasattr(env, 'cleanup'):
                env.cleanup()
        
        return agent, exit_status, result
    
    def _get_docker_image(self, instance: dict) -> str:
        """Get Docker image for instance."""
        if "image_name" in instance:
            return instance["image_name"]
        
        instance_id = instance.get("instance_id", "")
        id_docker_compatible = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    
    def _evaluate_result(
        self,
        exit_status: str,
        result: str,
        instance: dict,
    ) -> dict:
        """Evaluate execution result."""
        evaluation = {
            "instance_id": instance.get("instance_id", "unknown"),
            "resolved": False,
            "exit_status": exit_status,
        }
        
        if exit_status == "Submitted" and result and result.strip():
            evaluation["has_patch"] = True
            evaluation["resolved"] = False
        
        return evaluation


def run_single_instance(
    instance: dict,
    workspace_path: Path | str,
    model_name: str | None = None,
    use_docker: bool = False,
) -> dict:
    """Convenience function: run single instance."""
    runner = AgentRunner(
        workspace_path=workspace_path,
        model_name=model_name,
        use_docker=use_docker,
    )
    return runner.run_instance(instance)
