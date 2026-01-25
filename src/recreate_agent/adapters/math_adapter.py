"""
Math Domain Adapter - Mathematical reasoning tasks (MATH-500, AIME, AMC, etc.)
"""

import json
import os
import subprocess
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any

from .base import DomainAdapter, UnifiedInstance, UnifiedResult, DomainPromptConfig


class MathAdapter(DomainAdapter):
    """Adapter for mathematical reasoning tasks (MATH-500, AIME, AMC, etc.)."""
    
    # Project root directory (for resolving relative paths)
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    
    # Data source mapping
    # math500: from HuggingFace (HuggingFaceH4/MATH-500)
    # aime/amc: local files if available
    DATA_SOURCES = {
        "math500": "huggingface:HuggingFaceH4/MATH-500",
        "aime24": "datasets/math/data/aime24.json",
        "aime25": "datasets/math/data/aime25.json",
        "amc23": "datasets/math/data/amc23.json",
    }
    
    def __init__(
        self,
        data_source: str = "math500",
        docker_image: str = "math-env:latest",
    ):
        if data_source not in self.DATA_SOURCES:
            raise ValueError(f"Unknown data source: {data_source}. Available: {list(self.DATA_SOURCES.keys())}")
        self.data_source = data_source
        self.docker_image = docker_image
        self._problems = None
    
    @property
    def problems(self) -> list[dict]:
        """Lazy load math problems."""
        if self._problems is None:
            source = self.DATA_SOURCES[self.data_source]
            
            if source.startswith("huggingface:"):
                # Load from HuggingFace
                from datasets import load_dataset as hf_load
                hf_name = source.replace("huggingface:", "")
                dataset = hf_load(hf_name, split="test")
                self._problems = [dict(item) for item in dataset]
            else:
                # Load from local file
                data_file = self.PROJECT_ROOT / source
                if not data_file.exists():
                    raise FileNotFoundError(f"Data file not found: {data_file}")
                
                if data_file.suffix == ".jsonl":
                    self._problems = [json.loads(l) for l in data_file.read_text().splitlines() if l.strip()]
                else:
                    content = data_file.read_text()
                    if content.strip().startswith("["):
                        self._problems = json.loads(content)
                    else:
                        self._problems = [json.loads(l) for l in content.splitlines() if l.strip()]
        return self._problems
    
    @property
    def domain_name(self) -> str:
        return "math"
    
    @property
    def domain_description(self) -> str:
        return """
Mathematical Reasoning Benchmark for evaluating AI on math problem solving.

Data sources available:
- MATH-500: 500 problems from MATH dataset (algebra, geometry, number theory, etc.)
- AIME 2024/2025: American Invitational Mathematics Examination problems
- AMC 2023: American Mathematics Competition problems

Task: Solve mathematical problems step by step using logical reasoning.
Output format: Final answer must be in \\boxed{ANSWER} format.

The agent can use Python/SymPy for symbolic computation to assist problem solving.

Working directory: /workspace/
Problem file: /workspace/problem.txt
Solution file: /workspace/solution.txt
"""
    
    def load_dataset(
        self,
        subset: str = "",
        shuffle_file: Path | None = None,
        max_instances: int | None = None,
    ) -> list[UnifiedInstance]:
        """Load math problems."""
        all_problems = self.problems
        indexed_problems = list(enumerate(all_problems))
        
        if shuffle_file and shuffle_file.exists():
            data = json.loads(shuffle_file.read_text())
            id_to_info = {}
            for orig_idx, p in indexed_problems:
                instance_id = f"{self.data_source}_{orig_idx:04d}"
                id_to_info[instance_id] = (orig_idx, p)
            
            selected = []
            for item in data["instances"]:
                iid = item["instance_id"] if isinstance(item, dict) else str(item)
                if iid in id_to_info:
                    selected.append(id_to_info[iid])
        else:
            selected = indexed_problems
            if subset and self.data_source == "math500":
                selected = [(idx, p) for idx, p in selected 
                           if subset.lower() in p.get("subject", "").lower()]
        
        instances = []
        for orig_idx, p in selected:
            if self.data_source == "math500":
                problem_text = p.get("problem", "")
                answer = p.get("answer", "")
                subject = p.get("subject", "")
                level = p.get("level", 0)
                problem_id = p.get("unique_id", f"{orig_idx}")
            else:
                # AMC23 uses "question", AIME uses "problem"
                problem_text = p.get("problem", p.get("question", ""))
                answer = p.get("expected_answer", p.get("answer", ""))
                subject = p.get("source", self.data_source)
                level = 0
                problem_id = p.get("id", f"{self.data_source}_{orig_idx}")
            
            instances.append(UnifiedInstance(
                instance_id=f"{self.data_source}_{orig_idx:04d}",
                problem_statement=problem_text,
                difficulty=str(level) if level else "",
                category=subject,
                domain_data={
                    "answer": answer,
                    "solution": p.get("solution", p.get("reference_solution", "")),
                    "problem_id": problem_id,
                    "source": self.data_source,
                    "level": level,
                    "subject": subject,
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
        """Create Docker environment for math task."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        problem_file = output_dir / "problem.txt"
        problem_file.write_text(instance.problem_statement)
        
        check_image = subprocess.run(
            ["docker", "images", "-q", self.docker_image],
            capture_output=True, text=True
        )
        if not check_image.stdout.strip():
            raise RuntimeError(
                f"Docker image '{self.docker_image}' not found. "
                f"Build it with: cd math/docker && docker build -t {self.docker_image} ."
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
        """Run agent to solve math problem."""
        from minisweagent.agents.default import DefaultAgent, AgentConfig
        from minisweagent.models import get_model
        
        task = instance.problem_statement
        _, container_id = self.create_environment(instance, output_dir, tools_dir)
        
        @dataclass
        class MathEnvConfig:
            cwd: str = "/workspace"
            image: str = "math-env:latest"
            timeout: int = 60
        
        class MathEnvWrapper:
            def __init__(wrapper_self, cid: str):
                wrapper_self.container_id = cid
                wrapper_self.config = MathEnvConfig()
                wrapper_self.agent = None
            
            def execute(wrapper_self, command: str) -> dict:
                try:
                    proc = subprocess.run(
                        ["docker", "exec", wrapper_self.container_id, "bash", "-c", command],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    return {"returncode": proc.returncode, "output": proc.stdout + proc.stderr}
                except subprocess.TimeoutExpired:
                    return {"returncode": -1, "output": "Command timed out"}
            
            def get_template_vars(wrapper_self) -> dict:
                return asdict(wrapper_self.config)
            
            def cleanup(wrapper_self):
                subprocess.run(["docker", "rm", "-f", wrapper_self.container_id], capture_output=True)
        
        env_wrapper = MathEnvWrapper(container_id)
        
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
        env_wrapper.agent = agent
        
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
                agent.step()
                last_msg = agent.messages[-1] if agent.messages else None
                if last_msg and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in last_msg.get("content", ""):
                    exit_status = "Submitted"
                    break
            except Exception as e:
                if "NonTerminatingException" in str(type(e).__name__):
                    continue
                elif "TerminatingException" in str(type(e).__name__) or "MaxStepsExceeded" in str(type(e).__name__):
                    exit_status = str(type(e).__name__)
                    break
                else:
                    exit_status = f"Error: {type(e).__name__}: {str(e)[:100]}"
                    break
        
        solution_file = output_dir / "solution.txt"
        result = solution_file.read_text() if solution_file.exists() else ""
        
        return exit_status, result, container_id, env_wrapper
    
    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        """Evaluate math solution in Docker container."""
        expected_answer = instance.domain_data.get("answer", "")
        source = instance.domain_data.get("source", self.data_source)
        subject = instance.domain_data.get("subject", "")
        level = instance.domain_data.get("level", 0)
        
        solution_file = output_dir / "solution.txt"
        if not solution_file.exists():
            try:
                subprocess.run(
                    ["docker", "cp", f"{container_id}:/workspace/solution.txt", str(solution_file)],
                    capture_output=True, text=True
                )
            except:
                pass
        
        generated_text = solution_file.read_text() if solution_file.exists() else ""
        
        eval_script = f'''#!/usr/bin/env python3
import json
import re
import warnings
warnings.filterwarnings("ignore")

def last_boxed_only_string(string):
    idx = string.rfind("\\\\boxed")
    if "\\\\boxed " in string:
        return "\\\\boxed " + string.split("\\\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{{":
            num_left_braces_open += 1
        if string[i] == "}}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1].replace("$", "").replace("fbox", "boxed")

def remove_boxed(s):
    if "\\\\boxed " in s:
        return s[len("\\\\boxed "):]
    if s.startswith("\\\\boxed{{") and s.endswith("}}"):
        return s[len("\\\\boxed{{"):-1]
    return s

def extract_answer(text):
    boxed = last_boxed_only_string(text)
    if boxed:
        try:
            return remove_boxed(boxed)
        except:
            return ""
    lines = text.strip().split("\\n")
    return lines[-1].strip() if lines else ""

def normalize(answer):
    answer = answer.replace(",", "")
    answer = re.sub(r"\\\\text{{(.*?)}}", r"\\1", answer)
    answer = re.sub(r"\\\\textbf{{(.*?)}}", r"\\1", answer)
    answer = answer.replace("$", "").strip()
    return answer

def is_equiv(x1, x2):
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        if normalize(x1) == normalize(x2):
            return True
        try:
            n1 = float(normalize(x1))
            n2 = float(normalize(x2))
            if abs(n1 - n2) < 1e-6:
                return True
        except:
            pass
        try:
            p1 = parse_latex(x1)
            p2 = parse_latex(x2)
            diff = sympy.simplify(p1 - p2)
            if diff == 0:
                return True
            if sympy.Abs(diff) < 0.001:
                return True
        except:
            pass
        return False
    except ImportError:
        return normalize(x1) == normalize(x2)

try:
    with open("/workspace/solution.txt", "r") as f:
        generated = f.read()
except:
    generated = ""

expected = {repr(expected_answer)}
extracted = extract_answer(generated)

result = {{"passed": False, "expected_answer": expected, "extracted_answer": extracted, "result": "failed"}}

if not extracted:
    result["result"] = "no_answer"
else:
    try:
        if is_equiv(expected, extracted):
            result["passed"] = True
            result["result"] = "passed"
    except Exception as e:
        result["result"] = f"error: {{str(e)[:100]}}"

print("===EVAL_RESULT_START===")
print(json.dumps(result))
print("===EVAL_RESULT_END===")
'''
        
        eval_script_file = output_dir / "_math_eval.py"
        eval_script_file.write_text(eval_script)
        
        passed = False
        extracted_answer = ""
        result_str = "failed"
        error_msg = ""
        
        try:
            proc = subprocess.run(
                ["docker", "exec", container_id, "python3", "/workspace/_math_eval.py"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            output = proc.stdout + proc.stderr
            
            if "===EVAL_RESULT_START===" in output and "===EVAL_RESULT_END===" in output:
                start = output.find("===EVAL_RESULT_START===") + len("===EVAL_RESULT_START===")
                end = output.find("===EVAL_RESULT_END===")
                result_json = output[start:end].strip()
                try:
                    eval_result = json.loads(result_json)
                    passed = eval_result.get("passed", False)
                    extracted_answer = eval_result.get("extracted_answer", "")
                    result_str = eval_result.get("result", "unknown")
                except json.JSONDecodeError:
                    result_str = f"parse_error: {result_json[:100]}"
                    error_msg = result_str
            else:
                result_str = f"no_output: {output[:200]}"
                error_msg = result_str
                
        except subprocess.TimeoutExpired:
            result_str = "timeout"
            error_msg = "Evaluation timed out"
        except Exception as e:
            result_str = f"error: {type(e).__name__}: {str(e)[:100]}"
            error_msg = str(e)
        
        try:
            eval_script_file.unlink()
        except:
            pass
        
        eval_details = {
            "instance_id": instance.instance_id,
            "passed": passed,
            "expected_answer": expected_answer,
            "extracted_answer": extracted_answer,
            "result": result_str,
            "subject": subject,
            "level": level,
            "source": source,
            "error": error_msg,
        }
        eval_details_file = output_dir / "eval_result.json"
        eval_details_file.write_text(json.dumps(eval_details, indent=2))
        
        formatted_output = f"""
{'=' * 60}
MATH EVALUATION RESULT
{'=' * 60}

**Instance**: {instance.instance_id}
**Subject**: {subject}
**Level**: {level}
**Passed**: {'✓ YES' if passed else '✗ NO'}

## Answers
Expected: {expected_answer}
Extracted: {extracted_answer if extracted_answer else '(none)'}

## Result
{result_str}
"""
        
        return UnifiedResult(
            instance_id=instance.instance_id,
            success=passed,
            score=1.0 if passed else 0.0,
            error=error_msg,
            eval_result=eval_details,
            formatted_output=formatted_output,
        )
    
    def get_initial_prompt_template(self) -> str:
        return "meta_initial_instance.jinja2"
    
    def get_prompt_config(self) -> DomainPromptConfig:
        """Return unified DomainPromptConfig."""
        return DomainPromptConfig(
            domain="math",
            domain_display="Math",
            domain_description=self.domain_description,
            codebase_path="/workspace/",
            working_dir="/workspace/",
            timeout_seconds=60,
            docker_image="math-env:latest",
            packages=["sympy", "numpy", "scipy", "matplotlib"],
            eval_file="eval_result.json",
            eval_file_desc="evaluation result with pass/fail and answer comparison",
            success_criteria="Final answer matches expected answer (mathematically equivalent)",
            primary_metric_name="Correct",
            primary_metric_format="{'Yes' if score >= 0.99 else 'No'}",
            secondary_metrics=[
                ("Expected", "{expected_answer}"),
                ("Agent Answer", "{extracted_answer}"),
            ],
            failure_list_name="Error",
            failure_list_key="error",
            submit_command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            submit_instructions="Write your final answer with \\boxed{ANSWER} to solution.txt before submitting",
            code_block_lang="bash",
            tools_path="/workspace/extras/agent_tools",
            memory_path="/workspace/extras/agent_memory",
            inspect_example="cat /workspace/solution.txt",
            step_limit=20,
            cost_limit=2.0,
            anti_cheat_rules=[
                "NEVER access answer files or evaluation scripts",
                "Solve problems using mathematical reasoning only",
            ],
            common_commands={
                "read_problem": "cat /workspace/problem.txt",
                "write_solution": "cat > /workspace/solution.txt << 'EOF'\n...\nEOF",
            },
            scaffold_system='''You are solving mathematical problems. Use Python/SymPy for computation.

## Response Format
THOUGHT: <reasoning>

```bash
<ONE command>
```

## Files
- Problem: /workspace/problem.txt
- Solution: /workspace/solution.txt (write \\boxed{ANSWER} here)''',
            scaffold_instance='''## Math Problem
{{task}}

## Workflow (STEP BY STEP)
1. READ: `cat /workspace/problem.txt` - understand the full problem
2. SOLVE: Use Python/SymPy for computation if needed
3. VERIFY: Double-check your answer before writing
4. WRITE: Save solution with \\boxed{ANSWER} to solution.txt
5. SUBMIT: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

Remember: Verify your answer BEFORE submitting. One command per response.''',
            scaffold_format_error='''FORMAT ERROR. Use:
THOUGHT: <reasoning>

```bash
<one command>
```''',
            extra_files=[("eval_result.json", "expected vs extracted answer")],
            submission_checks=[
                "Answer in \\boxed{}? → REQUIRED for extraction",
                "Verify calculation? → Substitute back or use SymPy",
            ],
            error_file_list=[("eval_result.json", "expected/extracted answers")],
            memory_examples=[
                ("\\boxed{} format", "Final answer MUST be in \\boxed{...}"),
                ("Verify before submit", "Substitute answer back to check"),
                ("SymPy for computation", "Use sympy.simplify(), solve(), etc."),
            ],
            search_examples=["sympy solve", "modular arithmetic python"],
            example_instance_id="AIME24_001",
            workflow_notes=[
                "Answers checked for MATHEMATICAL EQUIVALENCE",
                "\\boxed{} format is REQUIRED",
            ],
        )
    
    def get_recreate_agent_config(self) -> dict:
        """Backward compatible interface."""
        config = self.get_prompt_config()
        return {
            "domain": config.domain,
            "domain_description": config.domain_description,
            "environment": {
                "codebase_path": config.codebase_path,
                "working_dir": config.working_dir,
                "timeout_seconds": config.timeout_seconds,
                "packages": config.packages,
            },
            "evaluation": {
                "error_file": config.eval_file,
                "error_file_desc": config.eval_file_desc,
                "success_criteria": config.success_criteria,
            },
            "scaffold_template": {
                "system_template": config.scaffold_system,
                "instance_template": config.scaffold_instance,
                "format_error_template": config.scaffold_format_error,
            },
            "rules": {"anti_cheat": config.anti_cheat_rules},
            "submission": {
                "command": config.submit_command,
                "instructions": config.submit_instructions,
            },
            "common_commands": config.common_commands,
            "suggested_limits": {
                "step_limit": config.step_limit,
                "cost_limit": config.cost_limit,
            },
            "meta_tools": {
                "tools_path": config.tools_path,
                "memory_path": config.memory_path,
                "inspect_example": config.inspect_example,
            },
            "code_block_lang": config.code_block_lang,
            "submission_checks": config.submission_checks,
            "error_file_list": config.error_file_list,
            "memory_examples": config.memory_examples,
            "search_examples": config.search_examples,
            "example_instance_id": config.example_instance_id,
            "workflow_notes": config.workflow_notes,
        }
