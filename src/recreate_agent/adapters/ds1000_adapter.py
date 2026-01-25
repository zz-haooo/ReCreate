"""
DS-1000 Adapter - Data Science Code Completion

Domain features:
- 1000 Python data science code completion problems
- Supported libraries: Pandas, Numpy, Matplotlib, Sklearn, Scipy, Pytorch, Tensorflow
- Evaluation: Execute generated code, check result correctness

Setup:
- Docker: datasets/ds1000/docker/Dockerfile
- Data: datasets/ds1000/data/ds1000.jsonl.gz
"""

import json
import os
import subprocess
from dataclasses import fields, dataclass, asdict
from pathlib import Path
from typing import Any

from .base import DomainAdapter, DomainPromptConfig, UnifiedInstance, UnifiedResult


class DS1000Adapter(DomainAdapter):
    """Adapter for DS-1000 data science code completion tasks."""
    
    def __init__(
        self,
        data_path: str | Path = "datasets/ds1000/data/ds1000.jsonl.gz",
        docker_image: str = "ds1000-image:latest",
    ):
        self.data_path = Path(data_path)
        self.docker_image = docker_image
        self._problems = None
    
    @property
    def problems(self) -> list[dict]:
        if self._problems is None:
            import gzip
            self._problems = [
                json.loads(l)
                for l in gzip.open(self.data_path, "rt").readlines()
            ]
        return self._problems
    
    @property
    def domain_name(self) -> str:
        return "ds1000"
    
    @property
    def domain_description(self) -> str:
        return """
DS-1000 is a benchmark for data science code generation with 1000 problems.

Task: Complete Python code snippets for data science operations.
Libraries: Pandas, Numpy, Matplotlib, Sklearn, Scipy, Pytorch, Tensorflow

The agent will receive a prompt with context and must generate the missing code.
The code will be tested by executing it against hidden test cases.

Working directory: /workspace/
The prompt contains all necessary context - no external files needed.
"""

    def load_dataset(
        self,
        subset: str = "",
        shuffle_file: Path | None = None,
        max_instances: int | None = None,
    ) -> list[UnifiedInstance]:
        """Load DS-1000 dataset."""
        if shuffle_file and shuffle_file.exists():
            data = json.loads(shuffle_file.read_text())
            instance_ids = [
                int(i["instance_id"].replace("ds1000_", ""))
                if isinstance(i, dict) else int(str(i).replace("ds1000_", ""))
                for i in data["instances"]
            ]
            problems = [self.problems[i] for i in instance_ids]
        else:
            problems = self.problems
        
        if subset:
            problems = [p for p in problems if subset.lower() in p["metadata"]["library"].lower()]
        
        instances = []
        for p in problems:
            problem_id = int(p["metadata"]["problem_id"])
            instances.append(UnifiedInstance(
                instance_id=f"ds1000_{problem_id:04d}",
                problem_statement=p["prompt"],
                difficulty=p["metadata"].get("perturbation_type", ""),
                category=p["metadata"]["library"],
                domain_data={
                    "problem_id": problem_id,
                    "library": p["metadata"]["library"],
                    "perturbation_type": p["metadata"]["perturbation_type"],
                    "code_context": p["code_context"],
                },
            ))
        
        if max_instances:
            instances = instances[:max_instances]
        
        return instances
    
    def create_environment(
        self,
        instance: UnifiedInstance,
        output_dir: Path,
        tools_dir: Path | None = None,
    ) -> tuple[Any, str]:
        """Create Docker environment for DS-1000 task."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_file = output_dir / "prompt.txt"
        prompt_file.write_text(instance.problem_statement)
        
        check_image = subprocess.run(
            ["docker", "images", "-q", self.docker_image],
            capture_output=True, text=True
        )
        if not check_image.stdout.strip():
            raise RuntimeError(
                f"Docker image '{self.docker_image}' not found. "
                f"Build it with: cd datasets/ds1000/docker && docker build -t {self.docker_image} ."
            )
        
        volume_mounts = ["-v", f"{output_dir.absolute()}:/workspace"]
        
        if tools_dir and tools_dir.exists():
            volume_mounts.extend(["-v", f"{tools_dir.absolute()}:/workspace/extras:ro"])
        
        docker_cmd = [
            "docker", "run", "-d",
            *volume_mounts,
            "-w", "/workspace",
            self.docker_image,
            "sleep", "3600",
        ]
        
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create Docker container: {result.stderr}")
        
        container_id = result.stdout.strip()[:12]
        if not container_id:
            raise RuntimeError(f"Docker returned empty container ID. stderr: {result.stderr}")
        
        return container_id, container_id
    
    def run_agent(
        self,
        instance: UnifiedInstance,
        scaffold: dict,
        model_name: str,
        output_dir: Path,
        tools_dir: Path | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, str, str, Any]:
        """Run agent to complete DS-1000 task."""
        from minisweagent.agents.default import DefaultAgent, AgentConfig
        from minisweagent.agents.default import NonTerminatingException, TerminatingException
        from minisweagent.models import get_model
        
        task = instance.problem_statement
        _, container_id = self.create_environment(instance, output_dir, tools_dir)
        
        @dataclass
        class DS1000EnvConfig:
            cwd: str = "/workspace"
            image: str = "ds1000-image:latest"
            timeout: int = 60
        
        class DS1000EnvWrapper:
            def __init__(wrapper_self, cid: str):
                wrapper_self.container_id = cid
                wrapper_self.config = DS1000EnvConfig()
            
            def execute(wrapper_self, command: str) -> dict:
                try:
                    proc = subprocess.run(
                        ["docker", "exec", wrapper_self.container_id, "bash", "-c", command],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    return {
                        "returncode": proc.returncode,
                        "output": proc.stdout + proc.stderr,
                    }
                except subprocess.TimeoutExpired:
                    return {"returncode": -1, "output": "Command timed out"}
            
            def get_template_vars(wrapper_self) -> dict:
                return asdict(wrapper_self.config)
            
            def cleanup(wrapper_self):
                subprocess.run(["docker", "rm", "-f", wrapper_self.container_id], 
                             capture_output=True)
        
        env_wrapper = DS1000EnvWrapper(container_id)
        
        agent_config_fields = {f.name for f in fields(AgentConfig)}
        filtered_scaffold = {k: v for k, v in scaffold.items() if k in agent_config_fields}
        config = AgentConfig(**filtered_scaffold)
        
        model_kwargs = {
            "custom_llm_provider": "openai",
            "temperature": temperature,
        }
        api_base = os.getenv("LLM_API_BASE")
        if api_base:
            model_kwargs["api_base"] = api_base
        model = get_model(model_name, {"model_kwargs": model_kwargs})
        
        agent = DefaultAgent(model, env_wrapper, config_class=lambda **kw: config)
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
        
        class FullEnvWrapper:
            def __init__(wrapper_self, env, agent_instance):
                wrapper_self._env = env
                wrapper_self.agent = agent_instance
                wrapper_self.container_id = env.container_id
            def cleanup(wrapper_self):
                wrapper_self._env.cleanup()
        
        return exit_status, result or "", container_id, FullEnvWrapper(env_wrapper, agent)
    
    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        """Evaluate DS-1000 solution in Docker container."""
        from recreate_agent.evaluators.ds1000 import DS1000Result, postprocess_code, format_ds1000_result_for_recreate_agent
        
        problem_id = instance.domain_data["problem_id"]
        problem = self.problems[problem_id]
        library = problem["metadata"]["library"]
        perturbation_type = problem["metadata"]["perturbation_type"]
        
        solution_file = output_dir / "solution.py"
        if solution_file.exists():
            generated_code = solution_file.read_text()
        else:
            py_files = list(output_dir.glob("*.py"))
            generated_code = py_files[0].read_text() if py_files else ""
        
        code = postprocess_code(generated_code)
        
        eval_script = f'''#!/usr/bin/env python3
import json
import sys

{problem["code_context"]}

code = {repr(code)}

result = {{"passed": False, "result": "", "error": ""}}
try:
    test_execution(code)
    {"test_string(code)" if "test_string(" in problem["code_context"] else "pass"}
    result["passed"] = True
    result["result"] = "passed"
except AssertionError as e:
    result["result"] = f"assertion_error: {{e}}"
    result["error"] = str(e)
except Exception as e:
    result["result"] = f"failed: {{type(e).__name__}}: {{e}}"
    result["error"] = str(e)

print("===EVAL_RESULT_START===")
print(json.dumps(result))
print("===EVAL_RESULT_END===")
'''
        
        eval_script_file = output_dir / "_ds1000_eval.py"
        eval_script_file.write_text(eval_script)
        
        passed = False
        result_str = "failed: unknown"
        error_msg = ""
        
        try:
            proc = subprocess.run(
                ["docker", "exec", container_id, "python3", "/workspace/_ds1000_eval.py"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            output = proc.stdout + proc.stderr
            
            if "===EVAL_RESULT_START===" in output and "===EVAL_RESULT_END===" in output:
                start = output.find("===EVAL_RESULT_START===") + len("===EVAL_RESULT_START===")
                end = output.find("===EVAL_RESULT_END===")
                result_json = output[start:end].strip()
                try:
                    eval_result = json.loads(result_json)
                    passed = eval_result.get("passed", False)
                    result_str = eval_result.get("result", "unknown")
                    error_msg = eval_result.get("error", "")
                except json.JSONDecodeError:
                    result_str = f"failed: could not parse result"
                    error_msg = result_str
            else:
                if proc.returncode == 0:
                    passed = True
                    result_str = "passed"
                else:
                    result_str = f"failed: exit_code={proc.returncode}"
                    error_msg = output[:1000] if output else "No output"
                    
        except subprocess.TimeoutExpired:
            result_str = "timed out"
            error_msg = "Evaluation timed out after 120 seconds"
        except Exception as e:
            result_str = f"failed: {type(e).__name__}: {e}"
            error_msg = str(e)
        
        try:
            eval_script_file.unlink()
        except:
            pass
        
        result = DS1000Result(
            instance_id=instance.instance_id,
            passed=passed,
            library=library,
            perturbation_type=perturbation_type,
            result=result_str,
            error_message=error_msg,
            generated_code=code,
        )
        
        eval_details_file = output_dir / "eval_result.json"
        eval_details_file.write_text(json.dumps({
            "passed": passed,
            "library": library,
            "perturbation_type": perturbation_type,
            "result": result_str,
            "error": error_msg,
            "generated_code_preview": code[:500] if code else "",
        }, indent=2))
        
        return UnifiedResult(
            instance_id=instance.instance_id,
            success=passed,
            score=1.0 if passed else 0.0,
            error=error_msg,
            eval_result={
                "passed": passed,
                "library": library,
                "result": result_str,
            },
            formatted_output=format_ds1000_result_for_recreate_agent(result),
        )
    
    def get_prompt_config(self) -> DomainPromptConfig:
        """Returns unified DomainPromptConfig"""
        return DomainPromptConfig(
            domain="ds1000",
            domain_display="DS-1000",
            domain_description=self.domain_description,
            codebase_path="/workspace/",
            working_dir="/workspace/",
            timeout_seconds=60,
            docker_image="ds1000-image:latest",
            packages=["pandas", "numpy", "matplotlib", "scipy", "sklearn", "pytorch", "tensorflow"],
            eval_file="eval_result.json",
            eval_file_desc="execution result with pass/fail status and error message",
            success_criteria="Generated code passes all test cases",
            primary_metric_name="Result",
            primary_metric_format="{'Passed' if score >= 0.99 else 'Failed'}",
            secondary_metrics=[],
            failure_list_name="Error",
            failure_list_key="error",
            submit_command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            submit_instructions="Test your solution before submitting",
            code_block_lang="bash",
            tools_path="/workspace/extras/agent_tools",
            memory_path="/workspace/extras/agent_memory",
            inspect_example="cat /workspace/solution.py",
            step_limit=30,
            cost_limit=3.0,
            anti_cheat_rules=[
                "NEVER access test cases or evaluation code",
                "Solve using ONLY: the prompt in /workspace/prompt.txt",
            ],
            common_commands={
                "read_prompt": "cat /workspace/prompt.txt",
                "write_code": "cat > /workspace/solution.py << 'EOF'\n...\nEOF",
            },
            scaffold_system='''You are completing Python code snippets for data science.

## Response Format
THOUGHT: <analysis>

```bash
<ONE command>
```

## Key Points
- Read the full prompt first: `cat /workspace/prompt.txt`
- Write ONLY the code snippet (result=...), not a full script
- Look for [insert] or <code> markers in the prompt''',
            scaffold_instance='''## Task
{{task}}

## Workflow
1. READ: `cat /workspace/prompt.txt` - understand the full context
2. ANALYZE: Identify what code to write (look for [insert] marker)
3. WRITE: Save your solution to /workspace/solution.py
4. SUBMIT: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

Remember: Write ONLY the requested snippet. Variables from context already exist.''',
            scaffold_format_error='''FORMAT ERROR. Use:
THOUGHT: <analysis>

```bash
<one command>
```''',
            extra_files=[("eval_result.json", "Pass/fail and error details")],
            
            submission_checks=[
                "Filled ALL [insert] markers?",
                "Only wrote the snippet, not full script?",
            ],
            
            error_file_list=[("eval_result.json", "pass/fail, error message")],
            
            memory_examples=[
                ("Write ONLY result= line", "Context code already exists, provide just the solution"),
                ("Variables exist", "Don't redefine a, df, x from context"),
            ],
            
            search_examples=["numpy reshape", "pandas fillna"],
            
            example_instance_id="ds1000_Pandas_001",
            
            workflow_notes=[
                "Tests code SNIPPETS, not full scripts",
                "Look for [insert] markers in prompt.txt",
            ],
        )
    
    def get_initial_prompt_template(self) -> str:
        return "meta_initial_instance.jinja2"

