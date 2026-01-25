#!/usr/bin/env python3
"""
Scaffold Editor - Generic scaffold file editing tool.

Similar to OpenHands' str_replace_editor, but for ReCreate-Agent managing Agent scaffolds.
Commands: view, create, str_replace, insert, append

Usage:
    # View file
    python scaffold_editor.py view /workspace/current_scaffold.yaml
    
    # Create file
    python scaffold_editor.py create /workspace/current_scaffold.yaml --content "..."
    
    # Replace content
    python scaffold_editor.py str_replace /workspace/current_scaffold.yaml --old "..." --new "..."
    
    # Insert content
    python scaffold_editor.py insert /workspace/current_scaffold.yaml --line 10 --content "..."
    
    # Append content
    python scaffold_editor.py append /workspace/current_scaffold.yaml --content "..."
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


class ScaffoldEditor:
    """Generic scaffold file editor."""

    def __init__(self, workspace: Path | None = None):
        self.workspace = Path(workspace) if workspace else Path("/workspace")
        self.history_dir = self.workspace / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def view(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """View file content with optional line range."""
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist"

        if file_path.is_dir():
            # List directory contents
            items = []
            for item in sorted(file_path.iterdir()):
                if not item.name.startswith("."):
                    prefix = "📁 " if item.is_dir() else "📄 "
                    items.append(f"{prefix}{item.name}")
            return f"Directory: {path}\n" + "\n".join(items)

        content = file_path.read_text()
        lines = content.splitlines()

        if start_line is not None:
            start_idx = max(0, start_line - 1)
            end_idx = len(lines) if end_line is None or end_line == -1 else min(len(lines), end_line)
            lines = lines[start_idx:end_idx]
            # Add line numbers
            numbered_lines = [f"{start_idx + i + 1:4d}| {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered_lines)

        # Default: show full content with line numbers
        numbered_lines = [f"{i + 1:4d}| {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def create(self, path: str, content: str) -> str:
        """Create new file (fails if file exists)."""
        file_path = Path(path)
        if file_path.exists():
            return f"Error: File '{path}' already exists. Use str_replace to modify it."

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"✓ Created: {path} ({len(content)} chars)"

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace string in file (must have unique match)."""
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist"

        content = file_path.read_text()

        # Check match count
        count = content.count(old_str)
        if count == 0:
            return f"Error: String not found in '{path}'.\nSearched for:\n{old_str[:200]}..."
        if count > 1:
            return f"Error: Found {count} matches. Please provide more context to make it unique."

        # Backup
        self._backup(file_path)

        # Replace
        new_content = content.replace(old_str, new_str, 1)
        file_path.write_text(new_content)

        return f"✓ Replaced in {path}\n  Old: {len(old_str)} chars → New: {len(new_str)} chars"

    def insert(self, path: str, line_number: int, content: str) -> str:
        """Insert content after specified line."""
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist"

        lines = file_path.read_text().splitlines()
        if line_number < 0 or line_number > len(lines):
            return f"Error: Line {line_number} out of range (file has {len(lines)} lines)"

        # Backup
        self._backup(file_path)

        # Insert
        new_lines = content.splitlines()
        lines = lines[:line_number] + new_lines + lines[line_number:]
        file_path.write_text("\n".join(lines) + "\n")

        return f"✓ Inserted {len(new_lines)} lines after line {line_number} in {path}"

    def append(self, path: str, content: str) -> str:
        """Append content to end of file."""
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist"

        # Backup
        self._backup(file_path)

        # Append
        with open(file_path, "a") as f:
            if not content.startswith("\n"):
                f.write("\n")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

        return f"✓ Appended {len(content)} chars to {path}"

    def undo(self, path: str) -> str:
        """Undo most recent modification to file."""
        file_path = Path(path)
        backup_path = self._get_latest_backup(file_path)

        if backup_path is None:
            return f"Error: No backup found for '{path}'"

        shutil.copy(backup_path, file_path)
        backup_path.unlink()
        return f"✓ Restored {path} from backup"

    def _backup(self, file_path: Path):
        """Create file backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = self.history_dir / backup_name
        shutil.copy(file_path, backup_path)

    def _get_latest_backup(self, file_path: Path) -> Path | None:
        """Get latest backup file."""
        pattern = f"{file_path.name}.*.bak"
        backups = sorted(self.history_dir.glob(pattern), reverse=True)
        return backups[0] if backups else None


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold Editor - Scaffold file editing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  view        View file or directory contents
  create      Create new file
  str_replace Replace string in file
  insert      Insert content after specified line
  append      Append content to end of file
  undo        Undo most recent modification

Examples:
  %(prog)s view /workspace/scaffold.yaml
  %(prog)s view /workspace/scaffold.yaml --range 1 50
  %(prog)s create /workspace/new.yaml --content "key: value"
  %(prog)s str_replace /workspace/scaffold.yaml --old "old text" --new "new text"
  %(prog)s insert /workspace/scaffold.yaml --line 10 --content "new line"
  %(prog)s append /workspace/scaffold.yaml --content "additional content"
  %(prog)s undo /workspace/scaffold.yaml
""",
    )

    parser.add_argument("command", choices=["view", "create", "str_replace", "insert", "append", "undo"])
    parser.add_argument("path", help="File or directory path")
    parser.add_argument("--workspace", default="/workspace", help="Workspace root directory")
    parser.add_argument("--content", help="File content (create/insert/append)")
    parser.add_argument("--old", help="Old string to replace (str_replace)")
    parser.add_argument("--new", default="", help="New string (str_replace)")
    parser.add_argument("--line", type=int, help="Line number (insert)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="Line range (view)")

    args = parser.parse_args()
    editor = ScaffoldEditor(workspace=args.workspace)

    try:
        if args.command == "view":
            start, end = args.range if args.range else (None, None)
            result = editor.view(args.path, start, end)
        elif args.command == "create":
            if not args.content:
                print("Error: --content is required for create command", file=sys.stderr)
                sys.exit(1)
            result = editor.create(args.path, args.content)
        elif args.command == "str_replace":
            if not args.old:
                print("Error: --old is required for str_replace command", file=sys.stderr)
                sys.exit(1)
            result = editor.str_replace(args.path, args.old, args.new)
        elif args.command == "insert":
            if args.line is None or not args.content:
                print("Error: --line and --content are required for insert command", file=sys.stderr)
                sys.exit(1)
            result = editor.insert(args.path, args.line, args.content)
        elif args.command == "append":
            if not args.content:
                print("Error: --content is required for append command", file=sys.stderr)
                sys.exit(1)
            result = editor.append(args.path, args.content)
        elif args.command == "undo":
            result = editor.undo(args.path)

        print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
