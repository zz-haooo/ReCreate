"""
Evolve utilities - Modular components for parallel evolution.

Modules:
- utils: Common utilities (safe_print, YAML loading, chunks)
- trajectory: Trajectory analysis and indexing
- stats: Statistics collection and saving
- scaffold_ops: Scaffold operations (loading, creation, merging)
- evolution: Core evolution logic (single evolution, synthesis)
"""

from .utils import (
    safe_print,
    clean_yaml_content,
    safe_load_yaml,
    chunks,
    BatchResult,
    console,
    console_lock,
)

from .trajectory import (
    generate_trajectory_index,
    flatten_domain_config,
    analyze_recreate_agent_traj,
)

from .stats import (
    collect_batch_stats,
    load_cumulative_stats,
    save_batch_evolution_stats,
)

from .scaffold_ops import (
    load_scaffold,
    init_agent_memory,
    create_submission_only_tool,
    create_default_scaffold,
    indent_yaml,
    create_symlink,
    merge_tools_and_memories,
)

from .evolution import (
    run_recreate_initial_creation,
    run_single_evolution,
    run_recreate_evolution_isolated,
    run_batch_synthesis,
)

__all__ = [
    # utils
    "safe_print",
    "clean_yaml_content",
    "safe_load_yaml",
    "chunks",
    "BatchResult",
    "console",
    "console_lock",
    # trajectory
    "generate_trajectory_index",
    "flatten_domain_config",
    "analyze_recreate_agent_traj",
    # stats
    "collect_batch_stats",
    "load_cumulative_stats",
    "save_batch_evolution_stats",
    # scaffold_ops
    "load_scaffold",
    "init_agent_memory",
    "create_submission_only_tool",
    "create_default_scaffold",
    "indent_yaml",
    "create_symlink",
    "merge_tools_and_memories",
    # evolution
    "run_recreate_initial_creation",
    "run_single_evolution",
    "run_recreate_evolution_isolated",
    "run_batch_synthesis",
]
