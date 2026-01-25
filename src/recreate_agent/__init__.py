"""ReCreate-Agent: An agent that creates and evolves other agents."""

__version__ = "0.1.0"

from pathlib import Path

package_dir = Path(__file__).resolve().parent
workspace_dir = package_dir.parent.parent / "recreate_workspace"

__all__ = ["package_dir", "workspace_dir", "__version__"]

