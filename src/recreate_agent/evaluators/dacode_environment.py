"""
DA-Code Environment Adapter

Creates Docker environment for data science tasks with data copying.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import docker


DACODE_IMAGE = "da_agent-image:latest"
DACODE_WORK_DIR = "/workspace"
STARTUP_DELAY = 2


@dataclass 
class DACodeInstance:
    """DA-Code task instance - compatible with SWE-bench instance format."""
    instance_id: str
    problem_statement: str      # task["instruction"]
    
    task_type: str = ""         # statistical analysis, ml, etc.
    hardness: str = ""          # Easy/Medium/Hard
    source_dir: str = ""        # Source data directory
    
    @classmethod
    def from_task(cls, task: dict, source_base: str | Path) -> "DACodeInstance":
        """Create from DA-Code task config."""
        return cls(
            instance_id=task["id"],
            problem_statement=task["instruction"],
            task_type=task.get("type", ""),
            hardness=task.get("hardness", ""),
            source_dir=str(Path(source_base) / task["id"]),
        )


class DACodeEnvironment:
    """
    DA-Code Docker environment.
    
    Key differences from SWE-bench:
    - Uses single shared image (da_agent-image)
    - Copies task data to /workspace/
    - No git, just file outputs
    """
    
    def __init__(
        self,
        instance: DACodeInstance,
        output_dir: Path,
        timeout: int = 300,  # 5 minutes for ML tasks
        extras_dir: Path | None = None,  # For agent_tools and agent_memory
    ):
        self.instance = instance
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.extras_dir = Path(extras_dir) if extras_dir else None
        self.container_id: str | None = None
        self._client = docker.from_env()
        self._container = None
    
    def setup(self) -> str:
        """
        Setup environment:
        1. Create Docker container with volume mount
        2. Copy source data to /workspace/
        
        Returns:
            container_id
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear output directory, but preserve scaffold and agent resources
        # These are needed by parallel_evolve.py for ReCreate-Agent phase
        preserve = {"scaffold.yaml", "agent_tools", "agent_memory"}
        for f in self.output_dir.iterdir():
            if f.name in preserve:
                continue
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                import shutil
                shutil.rmtree(f)
        
        # Create volume mounts
        volumes = {
            str(self.output_dir.absolute()): {
                "bind": DACODE_WORK_DIR,
                "mode": "rw"
            }
        }
        
        # Mount extras (agent_tools, agent_memory) if provided
        # Use rw mode to allow Agent to write memories
        if self.extras_dir and self.extras_dir.exists():
            volumes[str(self.extras_dir.absolute())] = {
                "bind": "/workspace/extras",
                "mode": "rw"
            }
        
        # Create container with volume mount
        self._container = self._client.containers.run(
            image=DACODE_IMAGE,
            volumes=volumes,
            detach=True,
            tty=True,
            stdin_open=True,
            name=f"dacode_{self.instance.instance_id}_{int(time.time())}",
        )
        
        self.container_id = self._container.id
        time.sleep(STARTUP_DELAY)
        
        # Copy source data
        self._copy_source_data()
        
        return self.container_id
    
    def _copy_source_data(self):
        """Copy task source data to container's /workspace/."""
        source_dir = Path(self.instance.source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Copy using docker cp
        subprocess.run(
            ["docker", "cp", f"{source_dir}/.", f"{self.container_id}:{DACODE_WORK_DIR}/"],
            check=True,
            capture_output=True,
        )
    
    def execute(self, command: str) -> dict:
        """Execute command in container with timeout."""
        if not self._container:
            raise RuntimeError("Container not started")
        
        try:
            # Use subprocess to call docker exec with timeout support
            result = subprocess.run(
                ["docker", "exec", "-w", DACODE_WORK_DIR, self.container_id, "bash", "-c", command],
                capture_output=True,
                timeout=self.timeout,
                text=True,
            )
            return {
                "returncode": result.returncode,
                "output": result.stdout + result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "output": f"Error: Command timed out after {self.timeout} seconds",
            }
        except Exception as e:
            return {
                "returncode": -1,
                "output": f"Error: {e}",
            }
    
    def get_output_files(self) -> list[str]:
        """List files in output directory."""
        return [f.name for f in self.output_dir.iterdir() if f.is_file()]
    
    def cleanup(self):
        """Stop and remove container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
                self._container.remove()
            except Exception:
                pass
            self._container = None
            self.container_id = None
    
    def keep_alive(self):
        """Keep container running (for ReCreate-Agent inspection)."""
        pass  # Container is kept alive by default


def load_dacode_dataset(
    task_type: str = "sa",
    source_dir: str | Path = "datasets/dacode/da_code/source/source",
    task_config_dir: str | Path = "datasets/dacode/da_code/configs/task",
) -> list[DACodeInstance]:
    """
    Load DA-Code dataset.
    
    Args:
        task_type: Task type filter (sa/ml/visual/di/dm/dw/all)
        source_dir: Source data directory
        task_config_dir: Task config directory
    
    Returns:
        List of DACodeInstance
    """
    import json
    
    task_config_dir = Path(task_config_dir)
    source_dir = Path(source_dir)
    
    # Determine which jsonl file to load
    if task_type == "all":
        task_file = task_config_dir / "all.jsonl"
    else:
        task_file = task_config_dir / f"{task_type}.jsonl"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task config file not found: {task_file}")
    
    # Load tasks
    instances = []
    for line in task_file.read_text().strip().split("\n"):
        if line.strip():
            task = json.loads(line)
            instance = DACodeInstance.from_task(task, source_dir)
            instances.append(instance)
    
    return instances

