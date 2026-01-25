"""
DA-Code Adapter - Data Science Tasks

Domain features:
- Solves data analysis tasks (statistical analysis, ML, visualization, etc.)
- Unified Docker environment
- Evaluation: Output matching score (0-1)
- Success metric: Average score

Setup:
- Docker: datasets/dacode/docker/Dockerfile
- Data: datasets/dacode/da_code/configs/task/
"""

import json
import os
from dataclasses import fields, dataclass, asdict
from pathlib import Path
from typing import Any

from .base import DomainAdapter, DomainPromptConfig, UnifiedInstance, UnifiedResult


class DACodeAdapter(DomainAdapter):
    """Adapter for DA-Code data science tasks."""
    
    def __init__(
        self,
        source_dir: str | Path = "datasets/dacode/da_code/source/source",
        task_config_dir: str | Path = "datasets/dacode/da_code/configs/task",
        gold_dir: str | Path = "datasets/dacode/da_code/gold/gold",
        eval_config_dir: str | Path = "datasets/dacode/da_code/configs/eval",
    ):
        self.source_dir = Path(source_dir)
        self.task_config_dir = Path(task_config_dir)
        self.gold_dir = Path(gold_dir)
        self.eval_config_dir = Path(eval_config_dir)
    
    @property
    def domain_name(self) -> str:
        return "data_science"
    
    @property
    def domain_description(self) -> str:
        return """
The agent will solve data science tasks in a Python environment with Docker.

Tasks include:
- Statistical Analysis (SA): hypothesis testing, regression, correlation
- Machine Learning (ML): classification, clustering, regression models
- Data Visualization (Visual): charts, plots, graphs
- Data Manipulation (DM): cleaning, transformation, aggregation
- Data Insight (DI): exploration, pattern discovery

Working directory: /workspace/ (contains data files like CSV, JSON)
Environment: Docker with pandas, numpy, scipy, sklearn, matplotlib
Output: result.csv, result.txt, plot.png, or JSON format
"""
    
    def load_dataset(
        self,
        subset: str = "",
        shuffle_file: Path | None = None,
        max_instances: int | None = None,
    ) -> list[UnifiedInstance]:
        """Load DA-Code dataset."""
        task_type = subset if subset else "all"
        
        if task_type == "all":
            task_file = self.task_config_dir / "all.jsonl"
        else:
            task_file = self.task_config_dir / f"{task_type}.jsonl"
        
        if not task_file.exists():
            raise FileNotFoundError(f"Task config file not found: {task_file}")
        
        tasks = []
        for line in task_file.read_text().strip().split("\n"):
            if line.strip():
                tasks.append(json.loads(line))
        
        if shuffle_file and shuffle_file.exists():
            shuffle_data = json.loads(shuffle_file.read_text())
            instance_order = [i["instance_id"] for i in shuffle_data.get("instances", [])]
            task_map = {t["id"]: t for t in tasks}
            tasks = [task_map[iid] for iid in instance_order if iid in task_map]
        
        if max_instances:
            tasks = tasks[:max_instances]
        
        return [
            UnifiedInstance(
                instance_id=t["id"],
                problem_statement=t["instruction"],
                difficulty=t.get("hardness", ""),
                category=t.get("type", ""),
                domain_data={
                    **t,
                    "source_dir": str(self.source_dir / t["id"]),
                },
            )
            for t in tasks
        ]
    
    def create_environment(
        self,
        instance: UnifiedInstance,
        output_dir: Path,
        tools_dir: Path | None = None,
    ) -> tuple[Any, str]:
        """Create DA-Code Docker environment."""
        from recreate_agent.evaluators.dacode_environment import DACodeEnvironment, DACodeInstance
        
        dacode_instance = DACodeInstance(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            task_type=instance.category,
            hardness=instance.difficulty,
            source_dir=instance.domain_data.get("source_dir", ""),
        )
        
        env = DACodeEnvironment(dacode_instance, output_dir, extras_dir=tools_dir)
        container_id = env.setup()
        
        return env, container_id
    
    def run_agent(
        self,
        instance: UnifiedInstance,
        scaffold: dict,
        model_name: str,
        output_dir: Path,
        tools_dir: Path | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, str, str, Any]:
        """Run agent on DA-Code task."""
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
        
        @dataclass
        class DACodeEnvConfig:
            cwd: str = "/workspace"
            image: str = "da_agent-image:latest"
            timeout: int = 60
        
        class DACodeEnvWrapper:
            def __init__(wrapper_self, dacode_env):
                wrapper_self._env = dacode_env
                wrapper_self.container_id = dacode_env.container_id
                wrapper_self.config = DACodeEnvConfig()
            
            def execute(wrapper_self, command: str) -> dict:
                return wrapper_self._env.execute(command)
            
            def get_template_vars(wrapper_self) -> dict:
                return asdict(wrapper_self.config)
            
            def cleanup(wrapper_self):
                pass
        
        env_wrapper = DACodeEnvWrapper(env)
        
        agent_config_fields = {f.name for f in fields(AgentConfig)}
        filtered_scaffold = {k: v for k, v in scaffold.items() if k in agent_config_fields}
        config = AgentConfig(**filtered_scaffold)
        
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
            def __init__(wrapper_self, dacode_env, agent_instance):
                wrapper_self._env = dacode_env
                wrapper_self.agent = agent_instance
                wrapper_self.container_id = dacode_env.container_id
            def cleanup(wrapper_self):
                wrapper_self._env.cleanup()
        
        return exit_status, result or "", env.container_id, FullEnvWrapper(env, agent)
    
    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        """Evaluate using DA-Code metrics."""
        from recreate_agent.evaluators.dacode import (
            DACodeEvaluator, format_dacode_result_for_recreate_agent
        )
        
        evaluator = DACodeEvaluator(
            gold_dir=self.gold_dir,
            eval_config_dir=self.eval_config_dir,
        )
        
        instance_dict = {
            "id": instance.instance_id,
            "type": instance.category,
            "hardness": instance.difficulty,
        }
        
        result = evaluator.evaluate_in_container(container_id, instance_dict, output_dir)
        
        return UnifiedResult(
            instance_id=instance.instance_id,
            success=result.error == "",
            score=result.score,
            error=result.error,
            details={
                "metric_func": result.metric_func,
                "output_files": result.output_files,
                "expected_files": result.expected_files,
                "ml_metric": result.ml_metric,
                "ml_raw_score": result.ml_raw_score,
            },
            eval_result={
                "score": result.score,
                "metric_func": result.metric_func,
                "metric_definition": result.metric_definition,
                "output_files": result.output_files,
                "expected_files": result.expected_files,
                "ml_metric": result.ml_metric,
                "ml_raw_score": result.ml_raw_score,
                "error": result.error,
                "raw_result": result.raw_result,
            },
            formatted_output=format_dacode_result_for_recreate_agent(result),
        )
    
    def get_prompt_config(self) -> DomainPromptConfig:
        """Returns unified DomainPromptConfig"""
        return DomainPromptConfig(
            domain="data_science",
            domain_display="DA-Code",
            domain_description=self.domain_description,
            codebase_path="/workspace/",
            working_dir="/workspace/",
            timeout_seconds=60,
            docker_image="da_agent-image:latest",
            packages=["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn"],
            eval_file="eval_result.json",
            eval_file_desc="evaluation result JSON with score and error details",
            success_criteria="Score (0.0-1.0) based on output quality; higher is better; test set reports average score",
            primary_metric_name="Score",
            primary_metric_format="{score:.2f}",
            secondary_metrics=[("Metric", "{metric_func}")],
            failure_list_name="Errors",
            failure_list_key="error",
            submit_command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            submit_instructions="Ensure output files exist before submitting",
            code_block_lang="bash",
            tools_path="/workspace/extras/agent_tools",
            memory_path="/workspace/extras/agent_memory",
            inspect_example="cat /workspace/solution.py",
            step_limit=50,
            cost_limit=5.0,
            anti_cheat_rules=[
                "NEVER read gold/answer files or evaluation scripts",
                "Solve using ONLY: task description and input data",
            ],
            common_commands={
                "view_data": "head file.csv",
                "run_script": "python3 script.py",
            },
            scaffold_system='''You are a Data Scientist. Solve tasks using Python scripts.

## Response Format
THOUGHT: <analysis>

```bash
<ONE command>
```

## Rules
- ONE command per response. Do NOT try to do everything at once.
- Write scripts using: echo 'code' > script.py or python3 -c "code"
- NEVER use heredoc (<<EOF, <<'EOF') - it causes truncation errors''',
            
            scaffold_instance='''## Task
{{task}}

## Workflow
1. EXPLORE: Check files with `ls`, `head`, `cat`
2. ANALYZE: Understand data structure and requirements
3. IMPLEMENT: Write and run Python scripts step-by-step
4. VERIFY: Check output exists and format matches requirements
5. SUBMIT: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

Remember:
- For ML tasks: train on train.csv, predict ONLY on test.csv
- Check sample_submission.csv for exact output format''',
            scaffold_format_error='''FORMAT ERROR. Use:
THOUGHT: <analysis>

```bash
<one command>
```''',
            extra_files=[("eval_result.json", "Score and evaluation details")],
            
            submission_checks=[
                "Output file name correct? → Check task/README for exact filename",
                "Row count = test set rows? → Common error: predicting on train+test",
                "Format matches sample_submission.csv? → Check columns and structure",
            ],
            
            error_file_list=[("eval_result.json", "score, metric_func, error details")],
            
            memory_examples=[
                ("Train/Test split", "Train on train.csv, predict ONLY on test.csv"),
                ("Check sample_submission.csv", "Shows output format, columns, row count"),
                ("Pandas groupby", "Returns GroupBy object, call agg()/sum()/mean()"),
            ],
            
            search_examples=["sklearn pipeline", "pandas read_csv encoding"],
            
            example_instance_id="sa_001_2",
            
            workflow_notes=[
                "DA-Code uses CONTINUOUS scores (0.0-1.0), not pass/fail",
                "For ML tasks: train.csv → model → predict test.csv only",
            ],
        )
    
    def get_initial_prompt_template(self) -> str:
        return "meta_initial_instance.jinja2"

