#!/usr/bin/env python3
"""
Execute commands in the same Docker environment as the Agent.

Usage:
    # Use existing container (recommended, faster!)
    python inspect_in_docker.py --command "<command>"
    
    # Use trajectory file to specify image (starts new container)
    python inspect_in_docker.py --trajectory <trajectory_file> --command "<command>"
    
    # Specify image directly (starts new container)
    python inspect_in_docker.py --image <image_name> --command "<command>"

Examples:
    # View Agent-modified code (using existing container)
    python inspect_in_docker.py --command "cat /testbed/django/core/validators.py | head -50"
    
    # View original code (starts new container)
    python inspect_in_docker.py \
        --trajectory results/django__django-10097/django__django-10097.traj.json \
        --command "cat /testbed/django/core/validators.py | head -50"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_env_config_from_trajectory(traj_path: Path) -> dict:
    """Extract Docker environment config from trajectory file."""
    traj = json.loads(traj_path.read_text())
    env_config = traj.get("info", {}).get("config", {}).get("environment", {})
    
    if not env_config.get("image"):
        raise ValueError(f"No Docker image config found in trajectory: {traj_path}")
    
    return env_config


def get_container_id() -> str:
    """Get container ID from container_id.txt."""
    container_file = Path("container_id.txt")
    if container_file.exists():
        return container_file.read_text().strip()
    return ""


def run_in_existing_container(container_id: str, command: str, cwd: str = "/testbed", timeout: int = 60) -> dict:
    """Execute command in existing container (using docker exec)."""
    docker_cmd = [
        "docker", "exec",
        "-w", cwd,
        container_id,
        "bash", "-c", command
    ]
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "success": False
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }


def run_in_new_container(image: str, command: str, cwd: str = "/testbed", timeout: int = 60) -> dict:
    """Start new container and execute command."""
    docker_cmd = [
        "docker", "run", "--rm",
        "-w", cwd,
        image,
        "bash", "-c", command
    ]
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "success": False
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }


def main():
    parser = argparse.ArgumentParser(
        description="Execute commands in the same Docker environment as the Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--trajectory", "-t", type=Path, help="Trajectory file path (starts new container)")
    group.add_argument("--image", "-i", help="Docker image name (starts new container)")
    group.add_argument("--container", help="Container ID (use existing container)")
    
    parser.add_argument("--command", "-c", required=True, help="Command to execute")
    parser.add_argument("--cwd", default="/testbed", help="Working directory (default: /testbed)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="JSON format output")
    
    args = parser.parse_args()
    
    # Prefer existing container
    container_id = args.container or get_container_id()
    
    if container_id:
        # Use existing container (fast!)
        print(f"Using existing container: {container_id[:12]}...", file=sys.stderr)
        result = run_in_existing_container(container_id, args.command, args.cwd, args.timeout)
    elif args.trajectory:
        # Get image from trajectory, start new container
        env_config = get_env_config_from_trajectory(args.trajectory)
        image = env_config["image"]
        cwd = env_config.get("cwd", args.cwd)
        print(f"Starting new container from image: {image}", file=sys.stderr)
        result = run_in_new_container(image, args.command, cwd, args.timeout)
    elif args.image:
        # Use specified image directly
        print(f"Starting new container from image: {args.image}", file=sys.stderr)
        result = run_in_new_container(args.image, args.command, args.cwd, args.timeout)
    else:
        print("Error: No container available. Use --trajectory or --image to specify.", file=sys.stderr)
        sys.exit(1)
    
    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["stdout"]:
            print(result["stdout"])
        if result["stderr"]:
            print(f"STDERR: {result['stderr']}", file=sys.stderr)
        sys.exit(result["returncode"])


if __name__ == "__main__":
    main()
