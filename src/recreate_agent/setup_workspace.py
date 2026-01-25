#!/usr/bin/env python3
"""
Setup Workspace - Initialize ReCreate-Agent workspace.

Creates necessary directory structure, copies tools and initial scaffold.
"""

import argparse
import shutil
from pathlib import Path

from recreate_agent import package_dir
from recreate_agent.scaffold import ScaffoldManager


def setup_workspace(
    workspace_path: Path,
    domain: str = "swe",
    force: bool = False,
) -> None:
    """Initialize workspace."""
    workspace = Path(workspace_path)

    if workspace.exists() and not force:
        print(f"Workspace already exists: {workspace}")
        print("Use --force to overwrite")
        return

    # Create directory structure
    print(f"Creating workspace at: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "history").mkdir(exist_ok=True)
    (workspace / "results").mkdir(exist_ok=True)
    (workspace / "tools").mkdir(exist_ok=True)

    # Copy ReCreate-Agent tool scripts (both .py and .sh)
    tools_src = package_dir / "tools"
    tools_dst = workspace / "tools"

    for tool_file in list(tools_src.glob("*.py")) + list(tools_src.glob("*.sh")):
        if tool_file.name != "__init__.py":
            dst = tools_dst / tool_file.name
            shutil.copy(tool_file, dst)
            dst.chmod(0o755)
            print(f"  Copied: {tool_file.name}")
    
    # Create Agent tools directory (categorized)
    agent_tools_dir = workspace / "agent_tools"
    agent_tools_dir.mkdir(exist_ok=True)
    for category in ["analysis", "testing", "debugging", "utils"]:
        (agent_tools_dir / category).mkdir(exist_ok=True)
    print(f"  Created: agent_tools/ (for Agent-created tools)")

    # Initialize default scaffold (dynamically created by ReCreate-Agent during evolution)
    manager = ScaffoldManager(workspace)
    scaffold = manager._create_default()
    scaffold.domain = domain
    manager.save(scaffold, backup=False)
    print(f"  Created default scaffold for domain: {domain}")

    # Copy prompt templates (for reference)
    prompts_src = package_dir / "prompts"
    prompts_dst = workspace / "prompts"
    prompts_dst.mkdir(exist_ok=True)

    for prompt_file in prompts_src.glob("*.jinja2"):
        shutil.copy(prompt_file, prompts_dst / prompt_file.name)
        print(f"  Copied prompt: {prompt_file.name}")

    print("\n✓ Workspace setup complete!")
    print(f"\nStructure:")
    print(f"  {workspace}/")
    print(f"  ├── current_scaffold.yaml  # Agent scaffold config")
    print(f"  ├── history/               # Scaffold version history")
    print(f"  ├── results/               # Agent execution results")
    print(f"  ├── tools/                 # ReCreate-Agent tools (scaffold_editor, read_trajectory, tool_manager)")
    print(f"  ├── agent_tools/           # Agent-created tools (categorized)")
    print(f"  │   ├── analysis/")
    print(f"  │   ├── testing/")
    print(f"  │   ├── debugging/")
    print(f"  │   └── utils/")
    print(f"  └── prompts/               # Prompt templates (reference)")


def main():
    parser = argparse.ArgumentParser(description="Setup ReCreate-Agent workspace")
    parser.add_argument(
        "workspace",
        nargs="?",
        default="./recreate_workspace",
        help="Workspace directory path",
    )
    parser.add_argument(
        "--domain",
        choices=["swe", "math", "data_science", "general"],
        default="swe",
        help="Domain for initial scaffold",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing workspace",
    )

    args = parser.parse_args()
    setup_workspace(Path(args.workspace), args.domain, args.force)


if __name__ == "__main__":
    main()
