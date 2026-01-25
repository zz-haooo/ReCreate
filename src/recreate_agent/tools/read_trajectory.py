#!/usr/bin/env python3
"""
Read Trajectory - Read, index, and analyze agent trajectories.

Provides lightweight indexing for quick step-based navigation.

Usage:
    # Generate index file (saves to trajectory_index.json)
    python read_trajectory.py index /path/to/trajectory.json
    
    # View specific step context (anchor navigation)
    python read_trajectory.py context /path/to/trajectory.json --step 12 --window 2
    
    # View trajectory summary
    python read_trajectory.py summary /path/to/trajectory.json
    
    # View submission analysis
    python read_trajectory.py submission /path/to/trajectory.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


class TrajectoryIndexer:
    """Trajectory indexer - builds event→step lightweight mapping."""

    def __init__(self, trajectory_path: Path):
        self.path = Path(trajectory_path)
        self.data = json.loads(self.path.read_text())
        self.messages = self.data.get("messages", [])
        self.info = self.data.get("info", {})
        self.dir = self.path.parent

    def build_index(self) -> dict:
        """Build trajectory index."""
        index = {
            "files": {"viewed": {}, "edited": {}, "searched": {}},
            "errors": [],
            "tests": {"ran": [], "passed": [], "failed": []},
            "expected": {"fail_to_pass": [], "pass_to_pass": []},
            "meta": {
                "total_steps": 0,
                "exit_status": self.info.get("exit_status", "unknown"),
                "trajectory_file": self.path.name,
            },
        }

        step = 0
        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                step += 1
                self._index_assistant_action(content, step, index)
            elif role == "user" and step > 0:
                self._index_environment_feedback(content, step, index)

        index["meta"]["total_steps"] = step

        # Load expected tests from expected_tests.txt
        self._load_expected_tests(index)

        return index

    def _index_assistant_action(self, content: str, step: int, index: dict):
        """Extract file operations and searches from assistant messages."""
        # File viewing: cat, head, tail, less, sed -n
        view_patterns = [
            r"cat\s+([^\s|>]+\.py)",
            r"cat\s+(/[^\s|>]+)",
            r"head\s+(?:-[n\d\s]+)?([^\s|]+\.py)",
            r"tail\s+(?:-[n\d\s]+)?([^\s|]+\.py)",
            r"less\s+([^\s]+)",
            r"sed\s+-n\s+['\"][^'\"]+['\"]\s+([^\s`]+)",
        ]
        for pattern in view_patterns:
            for match in re.finditer(pattern, content):
                path = match.group(1).strip("'\"` ")
                if path and "/" in path and not path.startswith("-"):
                    index["files"]["viewed"].setdefault(path, []).append(step)

        # File editing: sed -i, echo >, cat >
        edit_patterns = [
            r"sed\s+-i[^\s]*\s+['\"][^'\"]+['\"]\s+([^\s`]+)",
            r">\s*(/[^\s;&|`]+)",
            r"cat\s*>\s*(/[^\s<`]+)",
            r"echo\s+['\"].*['\"]\s*>\s*(/[^\s`]+)",
        ]
        for pattern in edit_patterns:
            for match in re.finditer(pattern, content):
                path = match.group(1).strip("'\"` ")
                if path and "/" in path:
                    index["files"]["edited"].setdefault(path, []).append(step)

        # Search: grep, find, rg
        search_patterns = [
            r"grep\s+(?:-[nrRilw]+\s+)*['\"]([^'\"]+)['\"]",
            r"grep\s+(?:-[nrRilw]+\s+)+([^\s'\"]+)\s+",
        ]
        for pattern in search_patterns:
            for match in re.finditer(pattern, content):
                term = match.group(1).strip()
                if term and len(term) > 2 and not term.startswith("-"):
                    index["files"]["searched"].setdefault(term, []).append(step)

        # Test runs: pytest
        if "pytest" in content.lower() or "python -m pytest" in content:
            index["tests"]["ran"].append(step)

    def _index_environment_feedback(self, content: str, step: int, index: dict):
        """Extract errors and test results from environment feedback."""
        # Error detection
        error_patterns = [
            (r"(SyntaxError):\s*(.{0,50})", "SyntaxError"),
            (r"(NameError):\s*(.{0,50})", "NameError"),
            (r"(TypeError):\s*(.{0,50})", "TypeError"),
            (r"(AttributeError):\s*(.{0,50})", "AttributeError"),
            (r"(ImportError):\s*(.{0,50})", "ImportError"),
            (r"(FileNotFoundError):\s*(.{0,50})", "FileNotFoundError"),
            (r"(KeyError):\s*(.{0,50})", "KeyError"),
            (r"(ValueError):\s*(.{0,50})", "ValueError"),
            (r"(IndentationError):\s*(.{0,50})", "IndentationError"),
        ]
        for pattern, error_type in error_patterns:
            match = re.search(pattern, content)
            if match:
                index["errors"].append({
                    "step": step,
                    "type": error_type,
                    "preview": match.group(2)[:50] if match.lastindex >= 2 else "",
                })
                break

        # Format errors
        if "EXACTLY ONE" in content or "format" in content.lower() and "error" in content.lower():
            index["errors"].append({"step": step, "type": "FormatError", "preview": ""})

        # Test results (pytest output)
        if "PASSED" in content:
            index["tests"]["passed"].append(step)
        if "FAILED" in content:
            index["tests"]["failed"].append(step)

    def _load_expected_tests(self, index: dict):
        """Load expected tests from expected_tests.txt."""
        expected_file = self.dir / "expected_tests.txt"
        if not expected_file.exists():
            return

        content = expected_file.read_text()
        current_section = None

        for line in content.splitlines():
            line = line.strip()
            if "FAIL_TO_PASS" in line:
                current_section = "fail_to_pass"
            elif "PASS_TO_PASS" in line:
                current_section = "pass_to_pass"
            elif line and not line.startswith("#") and current_section:
                index["expected"][current_section].append(line)

    def save_index(self, output_path: Path = None) -> Path:
        """Save index to file."""
        index = self.build_index()
        if output_path is None:
            output_path = self.dir / "trajectory_index.json"
        output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        return output_path


class TrajectoryReader:
    """Agent trajectory reader."""

    def __init__(self, trajectory_path: Path):
        self.path = Path(trajectory_path)
        self.data = json.loads(self.path.read_text())
        self.messages = self.data.get("messages", [])
        self.info = self.data.get("info", {})

    def context(self, step: int, window: int = 2) -> str:
        """View context around specific step (anchor navigation)."""
        start = max(1, step - window)
        end = step + window
        return self.steps(start, end)

    def steps(self, start: int, end: int) -> str:
        """Show specific step range."""
        lines = []
        step = 0
        in_range = False

        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "assistant":
                step += 1
                in_range = start <= step <= end

            if in_range:
                if role == "assistant":
                    lines.append(f"\n{'='*60}")
                    lines.append(f"STEP {step}")
                    lines.append("=" * 60)
                lines.append(f"[{role.upper()}]")
                # Moderate truncation
                if len(content) > 3000:
                    lines.append(content[:1500] + f"\n... ({len(content) - 3000} chars) ...\n" + content[-1500:])
                else:
                    lines.append(content)

            if step > end:
                break

        return "\n".join(lines) if lines else f"No steps found in range {start}-{end}"

    def summary(self) -> str:
        """Generate trajectory summary."""
        indexer = TrajectoryIndexer(self.path)
        index = indexer.build_index()

        lines = [
            "=" * 60,
            f"Trajectory Summary: {self.path.name}",
            "=" * 60,
            "",
            f"## Meta",
            f"- Total Steps: {index['meta']['total_steps']}",
            f"- Exit Status: {index['meta']['exit_status']}",
            "",
        ]

        # File index
        if any(index["files"].values()):
            lines.append("## Files")
            for action, files in index["files"].items():
                if files:
                    lines.append(f"  {action}:")
                    for path, steps in list(files.items())[:5]:
                        lines.append(f"    {path}: steps {steps}")
            lines.append("")

        # Error index
        if index["errors"]:
            lines.append(f"## Errors ({len(index['errors'])})")
            for e in index["errors"][:5]:
                lines.append(f"  - Step {e['step']}: {e['type']} {e['preview']}")
            lines.append("")

        # Test index
        if any(index["tests"].values()):
            lines.append("## Tests")
            if index["tests"]["ran"]:
                lines.append(f"  - Ran at steps: {index['tests']['ran']}")
            if index["tests"]["passed"]:
                lines.append(f"  - Passed at steps: {index['tests']['passed']}")
            if index["tests"]["failed"]:
                lines.append(f"  - Failed at steps: {index['tests']['failed']}")
            lines.append("")

        # Expected tests
        if index["expected"]["fail_to_pass"]:
            lines.append("## Expected Tests (from expected_tests.txt)")
            lines.append(f"  - FAIL_TO_PASS: {len(index['expected']['fail_to_pass'])} tests")
            for t in index["expected"]["fail_to_pass"][:3]:
                lines.append(f"      {t}")
            lines.append(f"  - PASS_TO_PASS: {len(index['expected']['pass_to_pass'])} tests")

        return "\n".join(lines)

    def submission(self) -> str:
        """Check submission content."""
        lines = ["=" * 60, "Submission Analysis", "=" * 60, ""]

        # Read from agent.patch
        patch_file = self.path.parent / "agent.patch"
        if patch_file.exists():
            patch = patch_file.read_text()
            lines.append(f"## Patch ({len(patch)} chars)")
            if patch.strip():
                file_count = patch.count("diff --git")
                lines.append(f"Files modified: {file_count}")
                lines.append("")
                lines.append(patch[:3000] if len(patch) > 3000 else patch)
            else:
                lines.append("Warning: Patch is EMPTY")
        else:
            # Read from info.submission
            submission = self.info.get("submission", "")
            if submission:
                lines.append(f"## Patch ({len(submission)} chars)")
                lines.append(submission[:3000])
            else:
                lines.append("## No Patch Found")

        # Last step analysis
        lines.append("")
        lines.append("## Last Step")
        total_steps = sum(1 for m in self.messages if m.get("role") == "assistant")
        lines.append(self.steps(total_steps, total_steps))

        return "\n".join(lines)

    def failures(self) -> str:
        """Analyze failure reasons."""
        indexer = TrajectoryIndexer(self.path)
        index = indexer.build_index()

        lines = ["=" * 60, "Failure Analysis", "=" * 60, ""]

        if index["errors"]:
            lines.append(f"## Errors ({len(index['errors'])})")
            for e in index["errors"]:
                lines.append(f"  - Step {e['step']}: {e['type']} {e['preview']}")
                lines.append(f"    → Use: python read_trajectory.py context {self.path} --step {e['step']}")
            lines.append("")

        # Test failures
        if index["tests"]["failed"]:
            lines.append(f"## Test Failures at steps: {index['tests']['failed']}")
            for step in index["tests"]["failed"][:3]:
                lines.append(f"    → Use: python read_trajectory.py context {self.path} --step {step}")
            lines.append("")

        # Last 3 steps
        total_steps = index["meta"]["total_steps"]
        lines.append("## Last 3 Steps")
        lines.append(self.steps(max(1, total_steps - 2), total_steps))

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Read and index Agent trajectories")
    parser.add_argument("command", choices=["index", "context", "summary", "steps", "failures", "submission"])
    parser.add_argument("path", help="Path to trajectory JSON file")
    parser.add_argument("--step", type=int, help="Step number for context command")
    parser.add_argument("--window", type=int, default=2, help="Context window size (default: 2)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="Step range for steps command")
    parser.add_argument("--output", "-o", help="Output path for index file")

    args = parser.parse_args()

    try:
        if args.command == "index":
            indexer = TrajectoryIndexer(args.path)
            output = Path(args.output) if args.output else None
            saved_path = indexer.save_index(output)
            index = indexer.build_index()
            print(f"Index saved to: {saved_path}")
            print(f"\nSummary:")
            print(f"  - Total steps: {index['meta']['total_steps']}")
            print(f"  - Files viewed: {len(index['files']['viewed'])}")
            print(f"  - Files edited: {len(index['files']['edited'])}")
            print(f"  - Errors: {len(index['errors'])}")
            print(f"  - Tests ran: {len(index['tests']['ran'])}")

        elif args.command == "context":
            if not args.step:
                print("Error: --step is required for context command", file=sys.stderr)
                sys.exit(1)
            reader = TrajectoryReader(args.path)
            print(reader.context(args.step, args.window))

        elif args.command == "summary":
            reader = TrajectoryReader(args.path)
            print(reader.summary())

        elif args.command == "steps":
            if not args.range:
                print("Error: --range is required for steps command", file=sys.stderr)
                sys.exit(1)
            reader = TrajectoryReader(args.path)
            print(reader.steps(args.range[0], args.range[1]))

        elif args.command == "failures":
            reader = TrajectoryReader(args.path)
            print(reader.failures())

        elif args.command == "submission":
            reader = TrajectoryReader(args.path)
            print(reader.submission())

    except FileNotFoundError:
        print(f"Error: File not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
