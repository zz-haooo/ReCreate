"""
SWE-bench Adapter - Software Engineering benchmark

Domain features:
- Solves real GitHub issues
- Docker environment contains specific version of codebase
- Submits patches via git diff
- Evaluation: Test case pass rate

Setup:
- Docker: Official SWE-bench images (sweb.eval.x86_64.*)
- Data: HuggingFace princeton-nlp/SWE-bench_Verified
"""

import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

from .base import DomainAdapter, DomainPromptConfig, UnifiedInstance, UnifiedResult


class SWEBenchAdapter(DomainAdapter):
    """Adapter for SWE-bench tasks."""
    
    @property
    def domain_name(self) -> str:
        return "swe"
    
    @property
    def domain_description(self) -> str:
        return """
SWE-bench: A benchmark for evaluating AI on real-world software engineering tasks.

The agent solves GitHub issues from real open-source projects (Django, Flask, Scikit-learn, Sympy, etc.).

Task types:
- Bug fixes: Identify and fix bugs based on issue descriptions
- Feature implementations: Add new functionality as described
- Refactoring: Improve code quality without changing behavior

Environment:
- Codebase location: `/testbed/` directory (a git repository)
- Agent must diagnose issues using: issue descriptions, stack traces, and code understanding
- Agent submits fixes via git diff

Evaluation: Agent's patch is tested against hidden test cases (FAIL_TO_PASS must pass, PASS_TO_PASS must not regress).
"""
    
    def load_dataset(
        self,
        subset: str = "",
        shuffle_file: Path | None = None,
        max_instances: int | None = None,
    ) -> list[UnifiedInstance]:
        """Load SWE-bench dataset."""
        from datasets import load_dataset as hf_load
        
        instances = list(hf_load("princeton-nlp/SWE-bench_Verified", split="test"))
        
        # Filter by repo if subset specified
        if subset:
            repo_filter = subset.replace("_evolve", "").replace("_test", "")
            instances = [i for i in instances if repo_filter.lower() in i["repo"].lower()]
        
        # Apply shuffle file ordering if provided
        if shuffle_file and shuffle_file.exists():
            shuffle_data = json.loads(shuffle_file.read_text())
            instance_order = [i["instance_id"] for i in shuffle_data.get("instances", [])]
            instance_map = {i["instance_id"]: i for i in instances}
            instances = [instance_map[iid] for iid in instance_order if iid in instance_map]
        
        # Limit instances
        if max_instances:
            instances = instances[:max_instances]
        
        # Convert to unified format
        return [
            UnifiedInstance(
                instance_id=i["instance_id"],
                problem_statement=i["problem_statement"],
                difficulty=i.get("difficulty", ""),
                category=i["repo"],
                domain_data=dict(i),
            )
            for i in instances
        ]
    
    def create_environment(
        self,
        instance: UnifiedInstance,
        output_dir: Path,
        tools_dir: Path | None = None,
    ) -> tuple[Any, str]:
        """Create SWE-bench Docker environment."""
        from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
        
        iid = instance.instance_id
        id_docker_compatible = iid.replace("__", "_1776_")
        image = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        
        env_config = {
            "image": image,
            "cwd": "/testbed",
            "timeout": 100,
            "run_args": [],
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
            "forward_env": ["GITHUB_TOKEN"],
            "pull_timeout": 300,
        }
        
        if tools_dir and tools_dir.exists():
            env_config["workspace_host_path"] = str(tools_dir)
        
        env = DockerEnvironment(config_class=DockerEnvironmentConfig, **env_config)
        env.execute("git config --global --add safe.directory /testbed")
        
        return env, env.container_id
    
    def run_agent(
        self,
        instance: UnifiedInstance,
        scaffold: dict,
        model_name: str,
        output_dir: Path,
        tools_dir: Path | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, str, str, Any]:
        """Run agent on SWE-bench task."""
        from minisweagent.agents.default import AgentConfig, DefaultAgent
        from minisweagent.agents.default import NonTerminatingException, TerminatingException
        from minisweagent.models import get_model
        
        task = instance.problem_statement
        
        model_kwargs = {
            "custom_llm_provider": "openai",
            "temperature": temperature,
        }
        api_base = os.getenv("LLM_API_BASE")
        if api_base:
            model_kwargs["api_base"] = api_base
        model = get_model(model_name, {"model_kwargs": model_kwargs})
        
        env, container_id = self.create_environment(instance, output_dir, tools_dir)
        
        agent_config_fields = {f.name for f in fields(AgentConfig)}
        filtered_scaffold = {k: v for k, v in scaffold.items() if k in agent_config_fields}
        config = AgentConfig(**filtered_scaffold)
        
        agent = DefaultAgent(model, env, config_class=lambda **kw: config)
        memory_template = scaffold.get("memory_template", "")
        agent.extra_template_vars |= {
            "task": task,
            "problem_statement": task,
            "memory_template": memory_template,
        }
        agent.messages = []
        agent.add_message("system", agent.render_template(config.system_template))
        agent.add_message("user", agent.render_template(config.instance_template))
        
        exit_status = "Running"
        result = None
        
        while True:
            try:
                output = agent.step()
                result = output.get("result")
            except NonTerminatingException as e:
                agent.add_message("user", str(e))
            except TerminatingException as e:
                agent.add_message("user", str(e))
                exit_status = type(e).__name__
                result = str(e)
                break
        
        class EnvWrapper:
            def __init__(wrapper_self, env_obj, agent_obj):
                wrapper_self.env = env_obj
                wrapper_self.agent = agent_obj
            def cleanup(wrapper_self):
                wrapper_self.env.cleanup()
        
        return exit_status, result or "", container_id, EnvWrapper(env, agent)
    
    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        """Evaluate using SWE-bench harness."""
        from recreate_agent.evaluators.swebench import (
            run_swebench_in_container, format_swebench_result_for_recreate_agent
        )
        
        proc = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c", "cd /testbed && git diff HEAD"],
            capture_output=True,
            text=False,
            timeout=30,
        )
        patch = proc.stdout.decode("utf-8", errors="ignore")
        
        result = run_swebench_in_container(
            container_id=container_id,
            instance=instance.domain_data,
            patch=patch,
            output_dir=output_dir,
        )
        
        return UnifiedResult(
            instance_id=instance.instance_id,
            success=result.resolved,
            score=1.0 if result.resolved else 0.0,
            error=result.error,
            eval_result={
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "fail_to_pass_success": result.fail_to_pass_success,
                "fail_to_pass_failure": result.fail_to_pass_failure,
                "pass_to_pass_success": result.pass_to_pass_success,
                "pass_to_pass_failure": result.pass_to_pass_failure,
                "report_json": result.report_json,
                "test_output": result.test_output[-10000:],
            },
            details={
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "fail_to_pass_success": result.fail_to_pass_success,
                "fail_to_pass_failure": result.fail_to_pass_failure,
            },
            formatted_output=format_swebench_result_for_recreate_agent(result),
        )
    
    def get_prompt_config(self) -> DomainPromptConfig:
        """Returns unified DomainPromptConfig"""
        return DomainPromptConfig(
            domain="swe",
            domain_display="SWE-bench",
            domain_description=self.domain_description,
            codebase_path="/testbed/",
            working_dir="/testbed/",
            timeout_seconds=100,
            submit_command="git add -A && git diff --cached && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            tools_path="/workspace/agent_tools",
            memory_path="/workspace/agent_memory",
            step_limit=250,
            cost_limit=10.0,
            
            # Scaffold template
            scaffold_system='''You are an expert software engineer solving GitHub issues.

## Response Format
THOUGHT: <analysis>

```bash
<ONE command>
```

## Rules
- ONE command per response. Do NOT try to do everything at once.
- No vim/nano. Use sed -i 's/old/new/' or python3 -c "..." for edits
- NEVER use heredoc (<<EOF) - it causes truncation errors
- NEVER read or modify test files''',
            
            scaffold_instance='''## Task
{{task}}

## Workflow (STEP BY STEP)
1. LOCATE: Find relevant files with `find` and `grep`
2. ANALYZE: Read and understand the code with `cat`
3. IMPLEMENT: Make targeted changes using sed or python -c
4. VERIFY: Run `git diff` to check your changes
5. SUBMIT: `git add -A && git diff --cached && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

Remember: Work step-by-step. Verify before submitting.''',
            
            scaffold_format_error='FORMAT ERROR. Use: THOUGHT: ... then ```bash <one command> ```',
            
            # ReCreate-Agent reference information
            extra_files=[("test_output.txt", "test output"), ("test_patch.txt", "benchmark tests")],
            submission_checks=["Empty patch? → heredoc fails", "Format error? → premature submit"],
            error_file_list=[("test_output.txt", "error traces"), ("test_patch.txt", "benchmark tests")],
            memory_examples=[("No heredoc", "Use sed or python -c"), ("Verify before submit", "Run git diff")],
            workflow_notes=["FAIL_TO_PASS = tests to fix", "PASS_TO_PASS = regression check"],
        )
    
    def get_initial_prompt_template(self) -> str:
        return "meta_initial_instance.jinja2"

