"""
ScaffoldManager - Scaffold version management.

Manages Agent scaffold configuration, including:
- Four components: system_template, instance_template, tools, observation_template
- Version history management
- Export to mini-swe-agent compatible format
"""

import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScaffoldConfig:
    """Scaffold configuration data structure."""

    # Metadata
    version: int = 0
    created_at: str = ""
    updated_at: str = ""
    domain: str = "general"  # swe, math, data_science, etc.

    # Four components
    system_template: str = ""
    instance_template: str = ""
    action_observation_template: str = ""
    format_error_template: str = ""

    # Extended config
    timeout_template: str = ""
    step_limit: int = 250
    cost_limit: float = 3.0

    # Tool config (reserved)
    tools_enabled: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class ScaffoldManager:
    """Scaffold version manager."""

    def __init__(self, workspace_path: Path | str):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.current_file = self.workspace / "current_scaffold.yaml"
        self.history_dir = self.workspace / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def get_current(self) -> ScaffoldConfig:
        """Get current scaffold configuration."""
        if self.current_file.exists():
            data = yaml.safe_load(self.current_file.read_text())
            return ScaffoldConfig(**data)
        return self._create_default()

    def save(self, scaffold: ScaffoldConfig, backup: bool = True) -> None:
        """Save scaffold configuration (with automatic version management)."""
        if backup and self.current_file.exists():
            current = self.get_current()
            self._backup_version(current)

        scaffold.version += 1
        scaffold.updated_at = datetime.now().isoformat()

        self.current_file.write_text(
            yaml.dump(asdict(scaffold), allow_unicode=True, default_flow_style=False, sort_keys=False)
        )

    def load_version(self, version: int) -> ScaffoldConfig | None:
        """Load specific version of scaffold."""
        version_file = self.history_dir / f"v{version:04d}.yaml"
        if version_file.exists():
            data = yaml.safe_load(version_file.read_text())
            return ScaffoldConfig(**data)
        return None

    def rollback(self, version: int) -> bool:
        """Rollback to specific version."""
        old_config = self.load_version(version)
        if old_config is None:
            return False
        self.save(old_config, backup=True)
        return True

    def list_versions(self) -> list[dict[str, Any]]:
        """List all versions."""
        versions = []
        for version_file in sorted(self.history_dir.glob("v*.yaml")):
            data = yaml.safe_load(version_file.read_text())
            versions.append({
                "version": data.get("version", 0),
                "updated_at": data.get("updated_at", ""),
                "domain": data.get("domain", "unknown"),
            })
        # Add current version
        if self.current_file.exists():
            current = self.get_current()
            versions.append({
                "version": current.version,
                "updated_at": current.updated_at,
                "domain": current.domain,
                "current": True,
            })
        return versions

    def export_to_agent_config(self, scaffold: ScaffoldConfig | None = None) -> dict[str, Any]:
        """Export to mini-swe-agent compatible config format."""
        if scaffold is None:
            scaffold = self.get_current()

        return {
            "agent": {
                "system_template": scaffold.system_template,
                "instance_template": scaffold.instance_template,
                "action_observation_template": scaffold.action_observation_template,
                "format_error_template": scaffold.format_error_template,
                "timeout_template": scaffold.timeout_template,
                "step_limit": scaffold.step_limit,
                "cost_limit": scaffold.cost_limit,
            }
        }

    def load_from_yaml(self, yaml_path: Path | str) -> ScaffoldConfig:
        """Load scaffold configuration from YAML file."""
        data = yaml.safe_load(Path(yaml_path).read_text())

        # Support mini-swe-agent config format
        if "agent" in data:
            agent_config = data["agent"]
            return ScaffoldConfig(
                system_template=agent_config.get("system_template", ""),
                instance_template=agent_config.get("instance_template", ""),
                action_observation_template=agent_config.get("action_observation_template", ""),
                format_error_template=agent_config.get("format_error_template", ""),
                timeout_template=agent_config.get("timeout_template", ""),
                step_limit=agent_config.get("step_limit", 250),
                cost_limit=agent_config.get("cost_limit", 3.0),
            )

        # Direct scaffold format
        return ScaffoldConfig(**data)

    def _backup_version(self, scaffold: ScaffoldConfig) -> None:
        """Backup current version to history directory."""
        version_file = self.history_dir / f"v{scaffold.version:04d}.yaml"
        version_file.write_text(
            yaml.dump(asdict(scaffold), allow_unicode=True, default_flow_style=False, sort_keys=False)
        )

    def _create_default(self) -> ScaffoldConfig:
        """Create default scaffold."""
        return ScaffoldConfig(
            domain="general",
            system_template="You are a helpful assistant.",
            instance_template="Your task: {{task}}",
            action_observation_template="<output>{{output.output}}</output>",
            format_error_template="Please provide exactly ONE bash command in triple backticks.",
        )


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scaffold Manager CLI")
    parser.add_argument("command", choices=["show", "list", "export", "rollback", "init"])
    parser.add_argument("--workspace", default="./recreate_workspace", help="Workspace directory")
    parser.add_argument("--version", type=int, help="Version number (for rollback)")
    parser.add_argument("--output", help="Output file (for export)")
    parser.add_argument("--from-yaml", help="Initialize from existing YAML config")

    args = parser.parse_args()
    manager = ScaffoldManager(args.workspace)

    if args.command == "show":
        scaffold = manager.get_current()
        print(yaml.dump(asdict(scaffold), allow_unicode=True, default_flow_style=False))

    elif args.command == "list":
        versions = manager.list_versions()
        print("Available versions:")
        for v in versions:
            current = " (current)" if v.get("current") else ""
            print(f"  v{v['version']:04d} - {v['updated_at']} - {v['domain']}{current}")

    elif args.command == "export":
        config = manager.export_to_agent_config()
        output = yaml.dump(config, allow_unicode=True, default_flow_style=False)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Exported to {args.output}")
        else:
            print(output)

    elif args.command == "rollback":
        if args.version is None:
            print("Error: --version is required for rollback")
            return
        if manager.rollback(args.version):
            print(f"Rolled back to version {args.version}")
        else:
            print(f"Version {args.version} not found")

    elif args.command == "init":
        if args.from_yaml:
            scaffold = manager.load_from_yaml(args.from_yaml)
            manager.save(scaffold, backup=False)
            print(f"Initialized from {args.from_yaml}")
        else:
            scaffold = manager._create_default()
            manager.save(scaffold, backup=False)
            print("Initialized with default scaffold")


if __name__ == "__main__":
    main()
