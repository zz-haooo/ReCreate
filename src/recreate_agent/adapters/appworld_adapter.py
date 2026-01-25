"""
AppWorld Domain Adapter - Interactive API tasks with simulated apps.

AppWorld simulates 9 everyday apps for interactive task completion.
Agent interacts with app APIs through Python REPL to complete tasks.

Key design:
- Uses Docker for AppWorld execution (SafetyGuard affects global Python environment)
- Maintains persistent Python session in Docker, variables persist across steps
- Uses official prompt format with complete few-shot examples
"""

import json
import os
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from .base import DomainAdapter, UnifiedInstance, UnifiedResult, DomainPromptConfig


class AppWorldAdapter(DomainAdapter):
    """Adapter for AppWorld interactive API tasks."""
    
    # Official AppWorld Apps
    APPS = ["amazon", "phone", "file_system", "spotify", "venmo", "gmail", "splitwise", "simple_note", "todoist"]
    
    # Official dataset subsets
    DATASETS = ["train", "dev", "test_normal", "test_challenge"]
    
    def __init__(self, dataset_name: str = "test_normal"):
        """
        Initialize AppWorld adapter.
        
        Args:
            dataset_name: Official dataset name (train, dev, test_normal, test_challenge)
        """
        self.dataset_name = dataset_name
        self._appworld_available = None
        self._task_cache_file = None
        self._setup_appworld_root()
    
    def _setup_appworld_root(self):
        """Setup APPWORLD_ROOT environment variable."""
        if "APPWORLD_ROOT" not in os.environ:
            project_root = Path(__file__).parent.parent.parent.parent
            appworld_root = project_root / "datasets" / "appworld"
            if appworld_root.exists():
                os.environ["APPWORLD_ROOT"] = str(appworld_root)
    
    def _check_appworld(self):
        """Check if AppWorld is available."""
        if self._appworld_available is not None:
            return self._appworld_available
        
        # Check for pre-extracted task cache
        cache_file = Path(__file__).parent.parent.parent.parent / "datasets" / "appworld" / "config" / "task_cache.json"
        if cache_file.exists():
            self._appworld_available = True
            self._task_cache_file = cache_file
            return True
        
        # Fallback: try AppWorld SDK
        try:
            from appworld.common.path_store import path_store
            data_path = os.path.join(path_store.data, "tasks")
            self._appworld_available = os.path.exists(data_path)
            if not self._appworld_available:
                print(f"WARNING: AppWorld data not found at {data_path}.")
        except ImportError:
            self._appworld_available = False
            print("WARNING: AppWorld not installed. Run: pip install appworld")
        return self._appworld_available
    
    @property
    def domain_name(self) -> str:
        return "appworld"
    
    @property
    def domain_description(self) -> str:
        return """
AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents (ACL'24 Best Resource Paper)

The agent completes day-to-day tasks by interacting with 9 simulated apps (Amazon, Spotify, Venmo, Gmail, etc.) through 457 APIs on behalf of a supervisor.

Task types:
- Single-app tasks: Operations within one app (e.g., play a Spotify playlist)
- Multi-app tasks: Coordinated operations across apps (e.g., find email, lookup contact, send payment)
- Answer-seeking tasks: Retrieve specific information (e.g., count items, find dates)
- Action tasks: Perform operations (e.g., send money, create playlist)

Environment:
- Python REPL with API access via `apis.{app_name}.{api_name}(**params)`
- Supervisor app for credentials: `apis.supervisor.show_account_passwords()`
- API docs lookup: `apis.api_docs.show_api_doc(app_name, api_name)`

Evaluation: Database state-based assertions (not output-based). All required changes must be made correctly.
"""
    
    def load_dataset(
        self,
        subset: str = "",
        shuffle_file: Path | None = None,
        max_instances: int | None = None,
    ) -> list[UnifiedInstance]:
        """Load AppWorld dataset."""
        if not self._check_appworld():
            return []
        
        dataset = self.dataset_name
        if subset in self.DATASETS:
            dataset = subset
            subset = ""
        
        # Load from cache file (preferred)
        if self._task_cache_file:
            cache = json.loads(self._task_cache_file.read_text())
            tasks = cache.get(dataset, [])
        else:
            # Fallback to AppWorld SDK
            from appworld.task import load_task_ids, Task
            task_ids = load_task_ids(dataset_name=dataset)
            tasks = []
            for task_id in task_ids:
                try:
                    task = Task.load(task_id=task_id, load_ground_truth=(dataset in ["train", "dev"]), ground_truth_mode="partial")
                    supervisor = task.supervisor
                    required_apps = task.ground_truth.required_apps if task.ground_truth else []
                    difficulty = str(task.ground_truth.metadata.get("difficulty", "")) if task.ground_truth else ""
                    tasks.append({
                        "task_id": task_id,
                        "instruction": task.instruction,
                        "supervisor": {"first_name": supervisor.first_name, "last_name": supervisor.last_name, "email": supervisor.email, "phone_number": supervisor.phone_number},
                        "required_apps": required_apps,
                        "difficulty": difficulty,
                    })
                    task.close()
                except Exception:
                    continue
        
        # Apply filters
        difficulty_filter = None
        if subset and subset.startswith("difficulty_"):
            try:
                difficulty_filter = subset.split("_")[1]
            except (ValueError, IndexError):
                pass
            subset = ""
        
        app_filter = subset.lower() if subset and subset.lower() in [a.lower() for a in self.APPS] else None
        
        # Apply shuffle file ordering if provided
        if shuffle_file and shuffle_file.exists():
            shuffle_data = json.loads(shuffle_file.read_text())
            instance_order = [i["instance_id"] for i in shuffle_data.get("instances", [])]
            task_ids_set = set(instance_order)
            tasks = sorted([t for t in tasks if t["task_id"] in task_ids_set], key=lambda t: instance_order.index(t["task_id"]))
        
        instances = []
        for t in tasks:
            required_apps = t.get("required_apps", [])
            difficulty = t.get("difficulty", "")
            
            if difficulty_filter and difficulty != difficulty_filter:
                continue
            if app_filter and required_apps and app_filter not in [a.lower() for a in required_apps]:
                continue
            
            supervisor = t["supervisor"]
            problem_statement = f"""My name is: {supervisor['first_name']} {supervisor['last_name']}. My personal email is {supervisor['email']} and phone number is {supervisor['phone_number']}.

Task: {t['instruction']}"""
            
            instances.append(UnifiedInstance(
                instance_id=t["task_id"],
                problem_statement=problem_statement,
                difficulty=difficulty,
                category=required_apps[0] if required_apps else "multi-app",
                domain_data={
                    "instruction": t["instruction"],
                    "supervisor": supervisor,
                    "required_apps": required_apps,
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
        """Create AppWorld Docker environment."""
        import subprocess
        
        output_dir.mkdir(parents=True, exist_ok=True)
        task_id = instance.instance_id
        container_name = f"appworld_{task_id}_{os.getpid()}"
        
        # AppWorld data path (host)
        appworld_root = os.environ.get("APPWORLD_ROOT", "./datasets/appworld/data")
        
        # Start Docker container
        docker_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{appworld_root}/data:/workspace/appworld_data/data:ro",
            "-v", f"{output_dir}:/workspace/output",
            "-e", f"APPWORLD_ROOT=/workspace/appworld_data",
            "-e", f"TASK_ID={task_id}",
            "appworld-agent:latest",
            "sleep", "infinity"
        ]
        
        try:
            subprocess.run(docker_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start Docker container: {e.stderr.decode()}")
        
        # Initialize AppWorld in container
        init_script = f'''
import os
os.environ["APPWORLD_ROOT"] = "/workspace/appworld_data"
from appworld import AppWorld
from appworld.common.path_store import path_store
path_store.update_root("/workspace/appworld_data")

_appworld = AppWorld(
    task_id="{task_id}",
    experiment_name="docker_agent_{task_id}",
    max_interactions=1000,
    timeout_seconds=120,
)
apis = _appworld.apis
requester = _appworld.requester
print("AppWorld initialized successfully")
'''
        
        init_result = subprocess.run(
            ["docker", "exec", container_name, "python3", "-c", init_script],
            capture_output=True, text=True
        )
        if init_result.returncode != 0:
            # Cleanup container on error
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            raise RuntimeError(f"Failed to initialize AppWorld: {init_result.stderr}")
        
        return None, container_name
    
    def run_agent(
        self,
        instance: UnifiedInstance,
        scaffold: dict,
        model_name: str,
        output_dir: Path,
        tools_dir: Path | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, str, str, Any]:
        """
        Run agent on AppWorld task using Docker execution.
        
        Key design:
        - Uses Docker container to isolate AppWorld's SafetyGuard
        - Each step executes via docker exec calling Python code
        - Persistent session maintained through same experiment name
        """
        import subprocess
        import base64
        from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError
        from minisweagent.agents.default import NonTerminatingException, TerminatingException
        from minisweagent.models import get_model
        
        output_dir.mkdir(parents=True, exist_ok=True)
        task_id = instance.instance_id
        container_name = f"appworld_{task_id}_{os.getpid()}"
        
        # AppWorld data path
        appworld_root = os.environ.get("APPWORLD_ROOT", "./datasets/appworld/data")
        
        # Create experiments directory for this task (to share between agent and evaluation)
        experiments_dir = output_dir / "appworld_experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Start Docker container with AppWorld
        # Mount both data (readonly) and experiments (readwrite) directories
        docker_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{appworld_root}/data:/workspace/appworld_data/data:ro",
            "-v", f"{experiments_dir}:/workspace/appworld_data/experiments",
            "-v", f"{output_dir}:/workspace/output",
            "-e", f"APPWORLD_ROOT=/workspace/appworld_data",
            "appworld-agent:latest",
            "sleep", "infinity"
        ]
        
        try:
            subprocess.run(docker_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start Docker container: {e.stderr.decode()}")
        
        # Initialize AppWorld session in container
        init_script = '''
import os
os.environ["APPWORLD_ROOT"] = "/workspace/appworld_data"
from appworld import AppWorld
from appworld.common.path_store import path_store
path_store.update_root("/workspace/appworld_data")

_appworld = AppWorld(
    task_id="{task_id}",
    experiment_name="docker_agent_{task_id}",
    max_interactions=1000,
    timeout_seconds=120,
)
apis = _appworld.apis
requester = _appworld.requester
print("INIT_SUCCESS")
'''.format(task_id=task_id)
        
        # Write init script to container and run it
        init_b64 = base64.b64encode(init_script.encode()).decode()
        init_cmd = f"python3 -c \"import base64; exec(base64.b64decode('{init_b64}').decode())\""
        init_result = subprocess.run(
            ["docker", "exec", container_name, "bash", "-c", init_cmd],
            capture_output=True, text=True, timeout=60
        )
        
        if "INIT_SUCCESS" not in init_result.stdout:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            raise RuntimeError(f"AppWorld init failed: {init_result.stderr}")
        
        class AppWorldAgent(DefaultAgent):
            """Agent that parses Python code blocks."""
            
            def parse_action(self, response: dict) -> dict:
                content = response.get("content", "")
                
                # Extract Python code blocks
                actions = re.findall(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
                if len(actions) == 1:
                    return {"action": actions[0].strip(), **response}
                
                # Try generic code blocks
                actions = re.findall(r"```\s*\n(.*?)\n```", content, re.DOTALL)
                if len(actions) == 1:
                    return {"action": actions[0].strip(), **response}
                
                raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
        
        class AppWorldDockerEnv:
            """Environment that executes Python code in AppWorld Docker container."""
            
            def __init__(wrapper_self):
                wrapper_self.container_name = container_name
                wrapper_self.task_id = task_id
                wrapper_self.task_completed = False
            
            def execute(wrapper_self, code: str) -> dict[str, str]:
                """Execute Python code in container's AppWorld session."""
                if 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in code:
                    code = "apis.supervisor.complete_task()"
                
                # Create execution script that reuses existing AppWorld session
                exec_script = '''
import os
import base64
os.environ["APPWORLD_ROOT"] = "/workspace/appworld_data"
from appworld import AppWorld
from appworld.common.path_store import path_store
path_store.update_root("/workspace/appworld_data")

_appworld = AppWorld(
    task_id="{task_id}",
    experiment_name="docker_agent_{task_id}",
    max_interactions=1000,
    timeout_seconds=120,
)
apis = _appworld.apis
requester = _appworld.requester

_code = base64.b64decode("{code_b64}").decode()
try:
    _result = _appworld.execute(_code)
    print(_result)
    if _appworld.task_completed():
        print("__TASK_COMPLETED__")
except Exception as e:
    print(f"Error: {{e}}")
finally:
    _appworld.close()
'''.format(task_id=task_id, code_b64=base64.b64encode(code.encode()).decode())
                
                exec_b64 = base64.b64encode(exec_script.encode()).decode()
                exec_cmd = f"python3 -c \"import base64; exec(base64.b64decode('{exec_b64}').decode())\""
                
                try:
                    result = subprocess.run(
                        ["docker", "exec", container_name, "bash", "-c", exec_cmd],
                        capture_output=True, text=True, timeout=120
                    )
                    output = result.stdout
                    if "__TASK_COMPLETED__" in output:
                        wrapper_self.task_completed = True
                        output = output.replace("__TASK_COMPLETED__", "").strip()
                    
                    if result.returncode != 0 or result.stderr:
                        output += f"\nStderr: {result.stderr}" if result.stderr else ""
                    
                    return {"output": output, "returncode": result.returncode}
                except subprocess.TimeoutExpired:
                    return {"output": "Error: Execution timeout (120s)", "returncode": 1}
                except Exception as e:
                    return {"output": f"Error: {e}", "returncode": 1}
            
            def get_template_vars(wrapper_self) -> dict:
                return {"task_id": wrapper_self.task_id}
            
            @property
            def config(wrapper_self):
                return {"type": "AppWorldDockerEnv", "container_name": container_name}
            
            def cleanup(wrapper_self):
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        
        env_wrapper = AppWorldDockerEnv()
        
        # Setup model
        model_kwargs = {
            "custom_llm_provider": "openai",
            "temperature": temperature,
        }
        api_base = os.getenv("LLM_API_BASE")
        if api_base:
            model_kwargs["api_base"] = api_base
        model = get_model(model_name, {"model_kwargs": model_kwargs})
        
        # Setup agent config
        agent_config_fields = {f.name for f in fields(AgentConfig)}
        filtered_scaffold = {k: v for k, v in scaffold.items() if k in agent_config_fields}
        config = AgentConfig(**filtered_scaffold)
        
        agent = AppWorldAgent(model, env_wrapper, config_class=lambda **kw: config)
        
        # Setup messages
        task = instance.problem_statement
        # Add memory_template to template vars if defined in scaffold
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
        
        # Run agent loop
        try:
            while True:
                print(f"\rAgent executing step {model.n_calls + 1} (${model.cost:.2f})...", end="", flush=True)
                try:
                    output = agent.step()
                    result = output.get("result", "")
                    
                    # Check task completion
                    if env_wrapper.task_completed:
                        exit_status = "Submitted"
                        break
                    
                    # Check limits
                    if model.n_calls >= config.step_limit:
                        exit_status = "StepLimitExceeded"
                        break
                    
                    if model.cost >= config.cost_limit:
                        exit_status = "CostLimitExceeded"
                        break
                        
                except NonTerminatingException as e:
                    agent.add_message("user", str(e))
                except TerminatingException as e:
                    agent.add_message("user", str(e))
                    exit_status = type(e).__name__
                    result = str(e)
                    break
        finally:
            print()  # New line after step counter
        
        # Create wrapper for cleanup
        class EnvWrapper:
            def __init__(wrapper_self, container, agent_obj):
                wrapper_self.container_name = container
                wrapper_self.agent = agent_obj
            
            def cleanup(wrapper_self):
                subprocess.run(["docker", "rm", "-f", wrapper_self.container_name], capture_output=True)
        
        return exit_status, result or "", container_name, EnvWrapper(container_name, agent)
    
    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        """Evaluate AppWorld task using Docker evaluation (avoids SQLite threading issues)."""
        import subprocess
        import base64
        
        task_id = instance.instance_id
        experiment_name = f"docker_agent_{task_id}"
        appworld_root = os.environ.get("APPWORLD_ROOT", "./datasets/appworld/data")
        
        # Get experiments directory from output_dir (created during run_agent)
        experiments_dir = output_dir / "appworld_experiments"
        if not experiments_dir.exists():
            # Fallback: create it (might happen if evaluate is called independently)
            experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create evaluation script
        eval_script = '''
import os
import json
os.environ["APPWORLD_ROOT"] = "/workspace/appworld_data"
from appworld.evaluator import evaluate_task
from appworld.common.path_store import path_store
path_store.update_root("/workspace/appworld_data")

try:
    test_tracker = evaluate_task(
        task_id="{task_id}",
        experiment_name="{experiment_name}",
        suppress_errors=True,
        save_report=True,
    )
    
    result = {{
        "success": test_tracker.success,
        "pass_percentage": getattr(test_tracker, 'pass_percentage', 100 if test_tracker.success else 0),
        "passes": [dict(p) for p in getattr(test_tracker, 'passes', [])],
        "failures": [dict(f) for f in getattr(test_tracker, 'failures', [])],
    }}
    print("EVAL_RESULT:" + json.dumps(result))
except Exception as e:
    print("EVAL_ERROR:" + str(e))
'''.format(task_id=task_id, experiment_name=experiment_name)
        
        # Run evaluation in a new Docker container with the same experiments directory
        eval_container = f"appworld_eval_{task_id}_{os.getpid()}"
        try:
            # Start container - mount the experiments directory from run_agent
            subprocess.run([
                "docker", "run", "-d",
                "--name", eval_container,
                "-v", f"{appworld_root}/data:/workspace/appworld_data/data:ro",
                "-v", f"{experiments_dir}:/workspace/appworld_data/experiments",
                "-e", "APPWORLD_ROOT=/workspace/appworld_data",
                "appworld-agent:latest",
                "sleep", "60"
            ], check=True, capture_output=True)
            
            # Execute evaluation
            eval_b64 = base64.b64encode(eval_script.encode()).decode()
            eval_cmd = f"python3 -c \"import base64; exec(base64.b64decode('{eval_b64}').decode())\""
            result = subprocess.run(
                ["docker", "exec", eval_container, "bash", "-c", eval_cmd],
                capture_output=True, text=True, timeout=60
            )
            
            output = result.stdout
            
            if "EVAL_RESULT:" in output:
                import json
                result_json = json.loads(output.split("EVAL_RESULT:")[1].strip())
                success = result_json["success"]
                score = result_json["pass_percentage"] / 100.0
                passes = [{"requirement": p.get("requirement", "N/A")} for p in result_json.get("passes", [])]
                failures = [{"requirement": f.get("requirement", "N/A"), "trace": str(f.get("trace", ""))[:500]} for f in result_json.get("failures", [])]
            elif "EVAL_ERROR:" in output:
                error_msg = output.split("EVAL_ERROR:")[1].strip()
                return UnifiedResult(
                    instance_id=instance.instance_id,
                    success=False,
                    score=0.0,
                    error=error_msg,
                    formatted_output=f"## Evaluation Error\n\n{error_msg}",
                )
            else:
                return UnifiedResult(
                    instance_id=instance.instance_id,
                    success=False,
                    score=0.0,
                    error=f"Unexpected output: {output[:500]}",
                    formatted_output=f"## Evaluation Error\n\nUnexpected output: {output[:500]}",
                )
                
        except Exception as e:
            return UnifiedResult(
                instance_id=instance.instance_id,
                success=False,
                score=0.0,
                error=str(e),
                formatted_output=f"## Evaluation Error\n\n{str(e)}",
            )
        finally:
            # Cleanup evaluation container
            subprocess.run(["docker", "rm", "-f", eval_container], capture_output=True)
        
        formatted_output = f"""## AppWorld Evaluation Result

**Status**: {"✅ Success" if success else "❌ Failed"}
**Score**: {score:.2%}
**Tests Passed**: {len(passes)}
**Tests Failed**: {len(failures)}

### Passed Requirements:
{chr(10).join(f"- {p.get('requirement', 'N/A')}" for p in passes) or "None"}

### Failed Requirements:
"""
        for f in failures:
            formatted_output += f"\n- {f.get('requirement', 'N/A')}"
            if f.get('trace'):
                formatted_output += f"\n  ```\n  {f.get('trace')[:300]}\n  ```"
        
        return UnifiedResult(
            instance_id=instance.instance_id,
            success=success,
            score=score,
            eval_result={
                "passes": passes,
                "failures": failures,
                "pass_count": len(passes),
                "fail_count": len(failures),
            },
            details={"pass_count": len(passes), "fail_count": len(failures)},
            formatted_output=formatted_output,
        )
    
    def get_initial_prompt_template(self) -> str:
        return "meta_initial_instance.jinja2"
    
    def get_prompt_config(self) -> DomainPromptConfig:
        """Returns unified DomainPromptConfig"""
        return DomainPromptConfig(
            domain="appworld",
            domain_display="AppWorld",
            domain_description=self.domain_description,
            codebase_path="/workspace/",
            working_dir="/workspace/",
            timeout_seconds=120,
            docker_image="appworld-agent:latest",
            packages=["apis (AppWorld API client)", "pendulum", "json", "datetime", "re"],
            eval_file="evaluation/report.md",
            eval_file_desc="Database state assertion results - shows which requirements passed/failed",
            success_criteria="All database state assertions pass (no collateral damage)",
            primary_metric_name="Assertions",
            primary_metric_format="{passed}/{total}",
            secondary_metrics=[],
            failure_list_name="Failed Requirements",
            failure_list_key="failures",
            submit_command="apis.supervisor.complete_task()",
            submit_instructions="Call complete_task() after finishing. Pass answer= for answer-seeking tasks.",
            code_block_lang="python",
            tools_path="/workspace/extras/agent_tools",
            memory_path="/workspace/extras/agent_memory",
            inspect_example="print(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_playlist_library'))",
            step_limit=50,
            cost_limit=10.0,
            anti_cheat_rules=[
                "NEVER access ground_truth, evaluation, or solution files",
                "NEVER look up answers directly - complete tasks autonomously",
                "Only interact through official APIs",
            ],
            common_commands={
                "list_apps": "print(apis.api_docs.show_app_descriptions())",
                "get_passwords": "passwords = apis.supervisor.show_account_passwords()",
                "login": "result = apis.{app}.login(username=email, password=pwd)",
            },
            scaffold_system='''I am your supervisor. You are an AI Assistant completing my day-to-day tasks fully autonomously.

You will interact with apps (spotify, venmo, amazon, etc) using their APIs in a *multi-step conversation* with a Python REPL.

## Key APIs
```python
print(apis.api_docs.show_app_descriptions())  # List apps
print(apis.api_docs.show_api_descriptions(app_name='spotify'))  # List APIs
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))  # API spec
```

## A. General
- Act fully on your own. You have full access to operate across my accounts.
- Never invent or guess values - look them up via APIs.
- When I omit details, choose any valid value (e.g., pick any available card).
- Avoid collateral damage. Only perform what I explicitly ask for.

## B. App-specific
- Personal info (credentials, addresses, cards) is in the Supervisor app.
- References to friends/family refer to people in my phone's contacts list.
- "file system" means the file system app, NOT machine's OS. Never use os module.
- Get current date/time from `datetime.now()` or phone app, not your internal clock.
- Paginated APIs: loop through ALL page_index until empty result.

## C. Code
- Write small chunks of code, ONE chunk per step.
- Only use the provided app APIs, not Python packages for app interaction.
- Always check API spec (show_api_doc) before calling unfamiliar APIs.
- Variables persist across steps (passwords, access_token, etc.)

## D. Task completion
- Call `apis.supervisor.complete_task(answer=result)` for answer-seeking tasks.
- Call `apis.supervisor.complete_task()` for action tasks.
- Answers must be minimal: "10" not "ten".''',
            scaffold_instance='''Using these APIs, generate code to solve the task:

{{ problem_statement }}''',
            memory_examples=[
                ("Save access_token", "token = result['access_token'] after login"),
                ("Handle pagination", "Loop page_index until empty result"),
                ("Friends/family = phone contacts", "References to friends/family refer to people in phone's contacts list"),
            ],
            workflow_notes=[
                "Variables persist across steps. Use apis.{app}.{method}() to call APIs.",
                "CRITICAL: When user omits details, agent should choose any valid value (e.g., pick any available card)",
                "CRITICAL: Avoid collateral damage - only perform what is explicitly asked",
                "CRITICAL: 'file system' means the file system app, NOT machine's OS. Never use os module.",
                "CRITICAL: Only use the provided app APIs, not Python packages for app interaction",
                "Date/time can use datetime.now() or phone app API - both are valid",
                "Step limit should be 50 for complex AppWorld tasks",
            ],
            submission_checks=[
                "Called apis.supervisor.complete_task()?",
                "Used exact API names from show_api_descriptions?",
                "Avoided collateral damage (only did what was asked)?",
            ],
        )
    
    def get_scaffold_template(self) -> dict:
        """Get scaffold template (dynamically created by ReCreate-Agent)"""
        return {}
