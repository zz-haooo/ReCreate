"""
Domain Adapters Package

Unified domain adapter interface, supporting multiple benchmarks:
- SWE-bench: Software engineering
- DA-Code: Data science  
- DS-1000: Data science code completion
- Math: Mathematical reasoning
- AppWorld: Interactive API tasks

To add a new domain:
1. Create new adapter file in adapters/ (implement DomainAdapter and get_prompt_config)
2. Create evaluator file in evaluators/
3. Prepare Docker environment
4. Register adapter in _adapter_classes below
"""

from .base import (
    DomainAdapter,
    DomainPromptConfig,
    UnifiedInstance,
    UnifiedResult,
    format_eval_result,
)

# Lazy import adapters to avoid circular imports and improve startup time
# Format: "domain_id": "module.path:ClassName"
# 
# Each domain has its own adapter file implementing the full DomainAdapter interface
#
_adapter_classes = {
    "swe": "recreate_agent.adapters.swe_adapter:SWEBenchAdapter",
    "data_science": "recreate_agent.adapters.dacode_adapter:DACodeAdapter",
    "ds1000": "recreate_agent.adapters.ds1000_adapter:DS1000Adapter",
    "math": "recreate_agent.adapters.math_adapter:MathAdapter",
    "appworld": "recreate_agent.adapters.appworld_adapter:AppWorldAdapter",
}


def get_adapter(domain: str, **kwargs) -> DomainAdapter:
    """
    Get domain adapter.
    
    Args:
        domain: Domain identifier (swe, data_science, ds1000, math, appworld)
        **kwargs: Arguments passed to adapter constructor
    
    Returns:
        DomainAdapter instance
    
    Example:
        >>> adapter = get_adapter("swe")
        >>> config = adapter.get_prompt_config()
        >>> print(config.domain_display)  # "SWE-bench"
    """
    if domain not in _adapter_classes:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(_adapter_classes.keys())}")
    
    # Dynamic import
    import importlib
    module_path, class_name = _adapter_classes[domain].rsplit(":", 1)
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    
    # Special parameter handling
    if domain == "appworld" and "dataset_name" in kwargs:
        return adapter_class(dataset_name=kwargs["dataset_name"])
    if domain == "math" and "data_source" in kwargs:
        return adapter_class(data_source=kwargs["data_source"])
    
    return adapter_class()


def list_domains() -> list[str]:
    """List all available domains."""
    return list(_adapter_classes.keys())


def get_all_prompt_configs() -> dict[str, DomainPromptConfig]:
    """Get all domain PromptConfigs (for documentation and debugging)."""
    configs = {}
    for domain in _adapter_classes:
        adapter = get_adapter(domain)
        if hasattr(adapter, 'get_prompt_config'):
            configs[domain] = adapter.get_prompt_config()
    return configs


def get_domain_paths(domain: str) -> dict:
    """
    Get domain-related paths.
    
    Args:
        domain: Domain identifier
    
    Returns:
        Dict containing:
        - config_dir: Config directory
        - docker_dir: Docker directory
    """
    from pathlib import Path
    
    # Domain name mapping (domain_id -> folder_name)
    folder_map = {
        "swe": "swe",
        "data_science": "dacode",
        "ds1000": "ds1000",
        "math": "math",
        "appworld": "appworld",
    }
    
    folder_name = folder_map.get(domain)
    if not folder_name:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Calculate datasets directory path
    # adapters/__init__.py -> src/recreate_agent/adapters/
    # datasets is in project root
    project_root = Path(__file__).parent.parent.parent.parent
    domains_dir = project_root / "datasets" / folder_name
    
    return {
        "config_dir": domains_dir / "config",
        "docker_dir": domains_dir / "docker",
    }


# Exports
__all__ = [
    # Base classes and data structures
    "DomainAdapter",
    "DomainPromptConfig",
    "UnifiedInstance", 
    "UnifiedResult",
    "format_eval_result",
    # Factory functions
    "get_adapter",
    "list_domains",
    "get_all_prompt_configs",
    "get_domain_paths",
]
