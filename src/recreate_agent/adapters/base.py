"""
Domain Adapter Base - Unified domain adapter base class and data structures.

To add a new domain:
1. Create a new adapter file in adapters/
2. Create an evaluator file in evaluators/
3. Add Docker files as needed
4. Register in __init__.py
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



@dataclass
class UnifiedInstance:
    """Unified instance format across all domains."""
    instance_id: str
    problem_statement: str
    difficulty: str = ""
    category: str = ""
    domain_data: dict = field(default_factory=dict)
    
    @property
    def display_name(self) -> str:
        return f"{self.instance_id} ({self.category})"


@dataclass  
class UnifiedResult:
    """Unified evaluation result across all domains."""
    instance_id: str
    success: bool
    score: float = 0.0
    error: str = ""
    details: dict = field(default_factory=dict)
    eval_result: dict = field(default_factory=dict)
    formatted_output: str = ""


@dataclass
class DomainPromptConfig:
    """
    Unified domain prompt configuration.
    
    All domains return the same format, templates render unified structure,
    no more `if domain == 'xxx'` branch logic.
    
    Design principles:
    1. Each field has default value, new domains can fill gradually
    2. Uses strings and simple types for Jinja2 rendering
    3. Domain-specific info passed through extra_* fields
    """
    
    domain: str
    domain_display: str
    domain_description: str = ""
    
    codebase_path: str = "/workspace/"
    working_dir: str = "/workspace/"
    timeout_seconds: int = 60
    docker_image: str = ""
    packages: list[str] = field(default_factory=list)
    
    eval_file: str = "eval_result.json"
    eval_file_desc: str = ""
    success_criteria: str = ""
    
    primary_metric_name: str = "Score"
    primary_metric_format: str = "{value}"
    secondary_metrics: list[tuple[str, str]] = field(default_factory=list)
    failure_list_name: str = "Failures"
    failure_list_key: str = "failures"
    
    submit_command: str = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    submit_instructions: str = ""
    
    code_block_lang: str = "bash"
    
    tools_path: str = "/workspace/agent_tools"
    memory_path: str = "/workspace/agent_memory"
    inspect_example: str = "ls /workspace/"
    
    step_limit: int = 50
    cost_limit: float = 5.0
    
    anti_cheat_rules: list[str] = field(default_factory=list)
    common_commands: dict[str, str] = field(default_factory=dict)
    
    scaffold_system: str = ""
    scaffold_instance: str = ""
    scaffold_format_error: str = ""
    
    extra_files: list[tuple[str, str]] = field(default_factory=list)
    submission_checks: list[str] = field(default_factory=list)
    
    error_file_list: list[tuple[str, str]] = field(default_factory=list)
    memory_examples: list[tuple[str, str]] = field(default_factory=list)
    search_examples: list[str] = field(default_factory=list)
    example_instance_id: str = "example_instance_001"
    workflow_notes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dict for Jinja2 template rendering."""
        return {
            "domain": self.domain,
            "domain_display": self.domain_display,
            "domain_description": self.domain_description,
            "codebase_path": self.codebase_path,
            "working_dir": self.working_dir,
            "timeout_seconds": self.timeout_seconds,
            "docker_image": self.docker_image,
            "packages": self.packages,
            "eval_file": self.eval_file,
            "eval_file_desc": self.eval_file_desc,
            "success_criteria": self.success_criteria,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_format": self.primary_metric_format,
            "secondary_metrics": self.secondary_metrics,
            "failure_list_name": self.failure_list_name,
            "failure_list_key": self.failure_list_key,
            "submit_command": self.submit_command,
            "submit_instructions": self.submit_instructions,
            "code_block_lang": self.code_block_lang,
            "tools_path": self.tools_path,
            "memory_path": self.memory_path,
            "inspect_example": self.inspect_example,
            "step_limit": self.step_limit,
            "cost_limit": self.cost_limit,
            "anti_cheat_rules": self.anti_cheat_rules,
            "common_commands": self.common_commands,
            "scaffold_system": self.scaffold_system,
            "scaffold_instance": self.scaffold_instance,
            "scaffold_format_error": self.scaffold_format_error,
            "extra_files": self.extra_files,
            "submission_checks": self.submission_checks,
            "error_file_list": self.error_file_list,
            "memory_examples": self.memory_examples,
            "search_examples": self.search_examples,
            "example_instance_id": self.example_instance_id,
            "workflow_notes": self.workflow_notes,
            **self.extra,
        }



class DomainAdapter(ABC):
    """
    Domain adapter abstract base class.
    
    Each domain must implement:
    1. domain_name - Domain identifier
    2. domain_description - Domain description
    3. load_dataset() - Load dataset
    4. create_environment() - Create execution environment
    5. run_agent() - Run agent
    6. evaluate() - Evaluate results
    7. get_prompt_config() - Return unified prompt config
    """
    
    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Domain identifier (e.g., 'swe', 'data_science')."""
        pass
    
    @property
    @abstractmethod
    def domain_description(self) -> str:
        """Human-readable description for ReCreate-Agent prompts."""
        pass
    
    @abstractmethod
    def load_dataset(
        self,
        subset: str = "",
        shuffle_file: Path | None = None,
        max_instances: int | None = None,
    ) -> list[UnifiedInstance]:
        """Load and filter dataset."""
        pass
    
    @abstractmethod
    def create_environment(
        self,
        instance: UnifiedInstance,
        output_dir: Path,
        tools_dir: Path | None = None,
    ) -> tuple[Any, str]:
        """
        Create execution environment for an instance.
        
        Returns: (environment_wrapper, container_id)
        """
        pass
    
    @abstractmethod
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
        Run agent on an instance.
        
        Returns: (exit_status, result, container_id, agent_wrapper)
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        container_id: str,
        instance: UnifiedInstance,
        output_dir: Path,
    ) -> UnifiedResult:
        """Evaluate agent's output."""
        pass
    
    @abstractmethod
    def get_prompt_config(self) -> DomainPromptConfig:
        """
        Return unified prompt configuration.
        
        This is the new unified interface, replacing get_meta_agent_config().
        """
        pass
    
    def get_recreate_agent_config(self) -> dict:
        """
        Backward compatible interface - returns dict format config.
        
        New code should use get_prompt_config() returning DomainPromptConfig.
        Now includes all new DomainPromptConfig fields for Jinja2 template rendering.
        """
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
            "rules": {
                "anti_cheat": config.anti_cheat_rules,
            },
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
    
    def get_initial_prompt_template(self) -> str:
        """Get the template name for initial scaffold creation."""
        return "meta_initial_instance.jinja2"



def format_eval_result(
    config: DomainPromptConfig,
    eval_result: dict,
    score: float,
) -> dict:
    """
    Format evaluation result based on DomainPromptConfig.
    
    Returns unified format dict for template rendering.
    """
    primary_value = config.primary_metric_format.format(
        value=score,
        score=score,
        passed=eval_result.get("tests_passed", eval_result.get("pass_count", 0)),
        failed=eval_result.get("tests_failed", eval_result.get("fail_count", 0)),
        total=eval_result.get("tests_passed", 0) + eval_result.get("tests_failed", 0),
    )
    
    secondary = []
    for name, fmt in config.secondary_metrics:
        try:
            value = fmt.format(**eval_result)
            secondary.append((name, value))
        except KeyError:
            pass
    
    # Get failure list
    failures = eval_result.get(config.failure_list_key, [])
    if isinstance(failures, list):
        failure_items = [
            f.get("requirement", f.get("test", str(f))) if isinstance(f, dict) else str(f)
            for f in failures[:10]
        ]
    else:
        failure_items = []
    
    return {
        "primary_metric_name": config.primary_metric_name,
        "primary_metric_value": primary_value,
        "secondary_metrics": secondary,
        "failure_list_name": config.failure_list_name,
        "failure_items": failure_items,
        "extra_files": config.extra_files,
    }
